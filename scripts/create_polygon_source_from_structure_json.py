#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a DA3 GLB-space floor/ceiling/walls structure JSON into the "
            "polygon source package expected by the v53/v90 projection/material pipeline."
        )
    )
    parser.add_argument("--structure-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scene-name", default="room_empty")
    parser.add_argument("--texture-ppm", type=float, default=900.0)
    parser.add_argument("--max-texture-size", type=int, default=3072)
    parser.add_argument("--min-texture-size", type=int, default=512)
    parser.add_argument(
        "--min-wall-texture-width",
        type=int,
        default=64,
        help=(
            "Independent minimum width for wall textures. Keeping this below the floor/height "
            "minimum preserves a consistent pixels-per-room-unit scale for narrow wall returns."
        ),
    )
    parser.add_argument("--copy-debug-image", type=Path, default=None)
    return parser.parse_args()


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def texture_shape(span_u: float, span_v: float, ppm: float, min_size: int, max_size: int) -> tuple[int, int]:
    w = max(min_size, int(math.ceil(span_u * ppm)))
    h = max(min_size, int(math.ceil(span_v * ppm)))
    scale = min(1.0, float(max_size) / max(float(w), float(h), 1.0))
    return max(8, int(round(w * scale))), max(8, int(round(h * scale)))


def wall_texture_shape(
    length: float,
    height: float,
    ppm: float,
    min_width: int,
    min_height: int,
    max_size: int,
) -> tuple[int, int]:
    w = max(int(min_width), int(math.ceil(length * ppm)))
    h = max(int(min_height), int(math.ceil(height * ppm)))
    scale = min(1.0, float(max_size) / max(float(w), float(h), 1.0))
    return max(8, int(round(w * scale))), max(8, int(round(h * scale)))


def polygon_mask_for_texture(poly: np.ndarray, bounds: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    span = np.maximum(bounds[1] - bounds[0], 1e-8)
    pts = []
    for p in poly:
        x = int(np.clip(round((p[0] - bounds[0, 0]) / span[0] * (w - 1)), 0, w - 1))
        y = int(np.clip(round((p[1] - bounds[0, 1]) / span[1] * (h - 1)), 0, h - 1))
        pts.append((x, y))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    return np.asarray(mask, dtype=np.uint8)


def point_in_triangle(point: np.ndarray, tri: np.ndarray) -> bool:
    a, b, c = tri
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -1e-8 and v >= -1e-8 and (u + v) <= 1.0 + 1e-8


def triangulate_earclip(poly: np.ndarray) -> list[list[int]]:
    if len(poly) < 3:
        return []
    work = np.asarray(poly, dtype=np.float64)
    if polygon_area(work) < 0:
        work = work[::-1].copy()
    indices = list(range(len(work)))
    tris: list[list[int]] = []
    guard = 0
    while len(indices) > 3 and guard < len(work) * len(work):
        guard += 1
        ear_found = False
        for j in range(len(indices)):
            i0 = indices[(j - 1) % len(indices)]
            i1 = indices[j]
            i2 = indices[(j + 1) % len(indices)]
            a, b, c = work[i0], work[i1], work[i2]
            if np.cross(b - a, c - b) <= 1e-10:
                continue
            tri = np.array([a, b, c], dtype=np.float64)
            if any(point_in_triangle(work[k], tri) for k in indices if k not in {i0, i1, i2}):
                continue
            tris.append([i0, i1, i2])
            indices.pop(j)
            ear_found = True
            break
        if not ear_found:
            break
    if len(indices) == 3:
        tris.append(indices[:])
    if not tris:
        tris = [[0, i, i + 1] for i in range(1, len(poly) - 1)]
    return tris


def write_obj(out_dir: Path, scene_name: str, poly: np.ndarray, floor_y: float, ceiling_y: float, bounds: np.ndarray) -> tuple[Path, Path]:
    obj_path = out_dir / f"{scene_name}.obj"
    mtl_path = out_dir / f"{scene_name}.mtl"
    room_h = float(ceiling_y - floor_y)
    origin = bounds[0]
    span = np.maximum(bounds[1] - bounds[0], 1e-8)

    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[tuple[str, list[int]]] = []

    def add_vertex(pos: tuple[float, float, float], uv: tuple[float, float]) -> int:
        verts.append(pos)
        uvs.append(uv)
        return len(verts)

    floor_ids: list[int] = []
    ceil_ids: list[int] = []
    for p in poly:
        local_x = float(p[0] - origin[0])
        local_z = float(p[1] - origin[1])
        u = float((p[0] - origin[0]) / span[0])
        v_img = float((p[1] - origin[1]) / span[1])
        uv = (u, 1.0 - v_img)
        floor_ids.append(add_vertex((local_x, 0.0, local_z), uv))
        ceil_ids.append(add_vertex((local_x, room_h, local_z), uv))

    tris = triangulate_earclip(poly)
    for tri in tris:
        faces.append(("floor", [floor_ids[i] for i in tri]))
        faces.append(("floor", [floor_ids[i] for i in tri[::-1]]))
        faces.append(("ceiling", [ceil_ids[i] for i in tri[::-1]]))
        faces.append(("ceiling", [ceil_ids[i] for i in tri]))

    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        ax, az = float(a[0] - origin[0]), float(a[1] - origin[1])
        bx, bz = float(b[0] - origin[0]), float(b[1] - origin[1])
        ids = [
            add_vertex((ax, 0.0, az), (0.0, 0.0)),
            add_vertex((bx, 0.0, bz), (1.0, 0.0)),
            add_vertex((bx, room_h, bz), (1.0, 1.0)),
            add_vertex((ax, room_h, az), (0.0, 1.0)),
        ]
        face = f"wall_{i:02d}"
        faces.append((face, ids))
        faces.append((face, ids[::-1]))

    with mtl_path.open("w", encoding="utf-8") as f:
        for name in ["floor", "ceiling", *[f"wall_{i:02d}" for i in range(len(poly))]]:
            f.write(f"newmtl {name}\n")
            f.write(f"map_Kd textures/{name}.png\n\n")

    with obj_path.open("w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        for x, y, z in verts:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for u, v in uvs:
            f.write(f"vt {u:.8f} {v:.8f}\n")
        current_mat = None
        for mat, ids in faces:
            if mat != current_mat:
                f.write(f"usemtl {mat}\n")
                current_mat = mat
            refs = " ".join(f"{idx}/{idx}" for idx in ids)
            f.write(f"f {refs}\n")

    return obj_path, mtl_path


def save_preview(out_dir: Path, poly: np.ndarray, bounds: np.ndarray, floor_y: float, ceiling_y: float) -> None:
    preview = Image.new("RGB", (900, 700), (248, 248, 248))
    draw = ImageDraw.Draw(preview)
    span = np.maximum(bounds[1] - bounds[0], 1e-8)
    pad = 70
    scale = min((preview.width - 2 * pad) / span[0], (preview.height - 2 * pad) / span[1])

    def xy(p: np.ndarray) -> tuple[int, int]:
        x = pad + (p[0] - bounds[0, 0]) * scale
        y = preview.height - pad - (p[1] - bounds[0, 1]) * scale
        return int(round(x)), int(round(y))

    pts = [xy(p) for p in poly]
    draw.polygon(pts, outline=(220, 40, 40), fill=(235, 235, 235))
    for i, p in enumerate(pts):
        r = 5
        draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(20, 90, 210))
        draw.text((p[0] + 7, p[1] - 13), str(i), fill=(20, 20, 20))
    draw.text(
        (18, 16),
        f"top-down {len(poly)}-corner source package; height={ceiling_y - floor_y:.4f}",
        fill=(20, 20, 20),
    )
    preview.save(out_dir / "structure_package_preview.png", quality=95)


def main() -> int:
    args = parse_args()
    data = json.loads(args.structure_json.read_text(encoding="utf-8"))
    world_vertices = np.asarray(data["vertices"], dtype=np.float64)
    use_room_basis = "room_vertices" in data and "world_from_room_matrix" in data
    vertices = np.asarray(data["room_vertices"], dtype=np.float64) if use_room_basis else world_vertices
    floor_face = next(face for face in data["faces"] if face.get("type") == "floor" or face.get("name") == "floor")
    floor_indices = [int(i) for i in floor_face["vertices"]]
    floor_vertices = vertices[floor_indices]
    poly = floor_vertices[:, [0, 2]].astype(np.float64)
    if polygon_area(poly) < 0:
        poly = poly[::-1].copy()
    bounds = np.stack([poly.min(axis=0), poly.max(axis=0)], axis=0).astype(np.float64)
    floor_y = float(np.min(vertices[:, 1]))
    ceiling_y = float(np.max(vertices[:, 1]))
    room_h = float(ceiling_y - floor_y)
    if room_h <= 1e-8:
        raise ValueError("Structure has degenerate vertical bounds")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tex_dir = args.out_dir / "textures"
    dbg_dir = args.out_dir / "debug"
    tex_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    span = bounds[1] - bounds[0]
    floor_size = texture_shape(float(span[0]), float(span[1]), args.texture_ppm, args.min_texture_size, args.max_texture_size)
    floor_mask = polygon_mask_for_texture(poly, bounds, floor_size)

    faces_meta: list[dict] = []
    for face in ("floor", "ceiling"):
        base = np.zeros((floor_size[1], floor_size[0], 3), dtype=np.uint8)
        base[floor_mask > 0] = np.array([180, 180, 180], dtype=np.uint8)
        Image.fromarray(base).save(tex_dir / f"{face}.png")
        Image.fromarray(floor_mask).save(dbg_dir / f"{face}_valid_mask.png")
        faces_meta.append(
            {
                "face": face,
                "type": "floor_ceiling_polygon",
                "texture_size": [int(floor_size[0]), int(floor_size[1])],
                "surface_point_count": 0,
                "observed_texels": 0,
                "valid_texels": int(np.count_nonzero(floor_mask)),
            }
        )

    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        length = float(np.linalg.norm(b - a))
        size = wall_texture_shape(
            length,
            room_h,
            args.texture_ppm,
            args.min_wall_texture_width,
            args.min_texture_size,
            args.max_texture_size,
        )
        face = f"wall_{i:02d}"
        base = np.full((size[1], size[0], 3), 180, dtype=np.uint8)
        mask = np.full((size[1], size[0]), 255, dtype=np.uint8)
        Image.fromarray(base).save(tex_dir / f"{face}.png")
        Image.fromarray(mask).save(dbg_dir / f"{face}_valid_mask.png")
        faces_meta.append(
            {
                "face": face,
                "type": "wall_edge",
                "edge_start": [float(a[0]), float(a[1])],
                "edge_end": [float(b[0]), float(b[1])],
                "length": length,
                "texture_size": [int(size[0]), int(size[1])],
                "surface_point_count": 0,
                "observed_texels": 0,
                "valid_texels": int(mask.size),
            }
        )

    obj_path, mtl_path = write_obj(args.out_dir, args.scene_name, poly, floor_y, ceiling_y, bounds)
    metadata = {
        "method": "polygon_source_package_from_da3_topdown_structure_json_v1",
        "structure_json": str(args.structure_json),
        "source_coordinate_space": "room_local_with_world_basis" if use_room_basis else data.get("coordinate_space", "unknown"),
        "up_axis": 1,
        "horizontal_axes": [0, 2],
        "manhattan_rotation_theta_rad": 0.0,
        "floor_y": floor_y,
        "ceiling_y": ceiling_y,
        "height": room_h,
        "min_wall_texture_width": int(args.min_wall_texture_width),
        "floorplan_polygon_uv": poly.tolist(),
        "floorplan_source": "da3_topdown_variable_corner_polygon_json",
        "selected_floorplan_source": "da3_topdown_variable_corner_polygon_json",
        "orthogonalized_floorplan": False,
        "bounds_uv": bounds.tolist(),
        "faces": faces_meta,
        "outputs": {
            "obj": str(obj_path),
            "mtl": str(mtl_path),
            "textures": str(tex_dir),
            "debug": str(dbg_dir),
        },
        "limitations": [
            "This package supplies geometry and valid atlas masks only; colors are filled by the photo projection stage.",
            "Coordinates are kept in the structure JSON coordinate space, normally DA3 exported GLB space.",
        ],
    }
    if use_room_basis:
        metadata["world_from_room_matrix"] = data["world_from_room_matrix"]
        metadata["room_from_world_matrix"] = data.get("room_from_world_matrix")
        metadata["room_basis_fit"] = data.get("room_basis_fit")
        metadata["world_vertices"] = world_vertices.tolist()
        metadata["room_vertices"] = vertices.tolist()
    for name in ("metadata.json", "manifest.json"):
        (args.out_dir / name).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    save_preview(args.out_dir, poly, bounds, floor_y, ceiling_y)
    if args.copy_debug_image and args.copy_debug_image.exists():
        shutil.copy2(args.copy_debug_image, args.out_dir / args.copy_debug_image.name)
    print(json.dumps({"out_dir": str(args.out_dir), "faces": [m["face"] for m in faces_meta]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
