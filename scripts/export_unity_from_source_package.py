#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Unity-friendly OBJ package from a polygon source package "
            "and a directory of face textures."
        )
    )
    parser.add_argument("--source-package-dir", type=Path, required=True)
    parser.add_argument("--texture-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-height-m", type=float, default=2.7)
    parser.add_argument("--version-name", default="restart_fullface_native_from_raw_pipeline")
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_source_metadata(source_dir: Path) -> dict:
    for name in ("metadata.json", "manifest.json"):
        path = source_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(source_dir / "metadata.json")


def infer_faces(metadata: dict, texture_dir: Path) -> list[str]:
    faces: list[str] = []
    for item in metadata.get("faces", []):
        face = item["face"] if isinstance(item, dict) else str(item)
        if (texture_dir / f"{face}.png").exists():
            faces.append(face)
    if not faces:
        faces = sorted(path.stem for path in texture_dir.glob("*.png"))
    if "floor" in faces and "ceiling" in faces:
        walls = sorted(face for face in faces if face.startswith("wall_"))
        other = sorted(face for face in faces if face not in {"floor", "ceiling"} and not face.startswith("wall_"))
        return ["floor", "ceiling", *walls, *other]
    return faces


def scale_obj(source_obj: Path, out_obj: Path, scale: float) -> None:
    out_lines: list[str] = []
    for line in source_obj.read_text(encoding="utf-8").splitlines():
        if line.startswith("mtllib "):
            out_lines.append("mtllib room.mtl")
        elif line.startswith("v "):
            parts = line.split()
            xyz = [float(parts[1]) * scale, float(parts[2]) * scale, float(parts[3]) * scale]
            out_lines.append(f"v {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}")
        else:
            out_lines.append(line)
    out_obj.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def write_mtl(out_mtl: Path, faces: list[str]) -> None:
    lines: list[str] = []
    for face in faces:
        lines.extend(
            [
                f"newmtl {face}",
                "Kd 1.000000 1.000000 1.000000",
                f"map_Kd textures/{face}.png",
                "",
            ]
        )
    out_mtl.write_text("\n".join(lines), encoding="utf-8")


def copy_textures(texture_dir: Path, out_dir: Path, faces: list[str]) -> dict[str, list[int]]:
    out_tex = out_dir / "textures"
    out_tex.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, list[int]] = {}
    for face in faces:
        src = texture_dir / f"{face}.png"
        require(src)
        dst = out_tex / src.name
        shutil.copy2(src, dst)
        with Image.open(dst) as img:
            sizes[face] = [int(img.width), int(img.height)]
    return sizes


def write_preview(out_dir: Path, sizes: dict[str, list[int]], faces: list[str]) -> None:
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    rows = []
    for face in faces:
        img = Image.open(out_dir / "textures" / f"{face}.png").convert("RGB")
        thumb = img.copy()
        thumb.thumbnail((640, 220), Image.Resampling.LANCZOS)
        row = Image.new("RGB", (820, max(260, thumb.height + 48)), (245, 245, 245))
        draw = ImageDraw.Draw(row)
        draw.text((12, 12), f"{face}  {sizes[face][0]}x{sizes[face][1]}", fill=(0, 0, 0), font=font)
        row.paste(thumb, (150, 34))
        rows.append(row)
    if not rows:
        return
    sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), (245, 245, 245))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(preview_dir / "unity_texture_overview.jpg", quality=92)


def main() -> int:
    args = parse_args()
    require(args.source_package_dir)
    require(args.texture_dir)
    source_obj = args.source_package_dir / "room_empty.obj"
    require(source_obj)

    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    metadata = load_source_metadata(args.source_package_dir)
    source_height = float(metadata.get("height") or (metadata.get("ceiling_y", 0.0) - metadata.get("floor_y", 0.0)))
    if source_height <= 0:
        raise ValueError("source package metadata has no positive room height")
    scale = args.target_height_m / source_height
    faces = infer_faces(metadata, args.texture_dir)

    scale_obj(source_obj, args.out_dir / "room.obj", scale)
    write_mtl(args.out_dir / "room.mtl", faces)
    sizes = copy_textures(args.texture_dir, args.out_dir, faces)

    preview = args.source_package_dir / "structure_package_preview.png"
    if preview.exists():
        shutil.copy2(preview, args.out_dir / "structure_package_preview.png")
    write_preview(args.out_dir, sizes, faces)

    manifest = {
        "method": "unity_scaled_from_polygon_source_package_v1",
        "version_name": args.version_name,
        "entry_obj": "room.obj",
        "entry_mtl": "room.mtl",
        "textures_dir": "textures",
        "source_geometry_package": str(args.source_package_dir),
        "texture_source_dir": str(args.texture_dir),
        "source_da3_height_units": source_height,
        "target_unity_height_m": args.target_height_m,
        "vertex_scale_factor": scale,
        "faces": faces,
        "texture_sizes_wh": sizes,
    }
    (args.out_dir / "unity_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.zip:
        archive = shutil.make_archive(str(args.out_dir), "zip", args.out_dir.parent, args.out_dir.name)
        print(f"[export] wrote {archive}")
    print(f"[export] wrote {args.out_dir / 'room.obj'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
