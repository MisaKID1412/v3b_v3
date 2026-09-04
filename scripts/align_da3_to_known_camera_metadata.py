#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from pygltflib import GLTF2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace drifting DA3 poses with known same-center perspective-camera metadata, "
            "then rebuild the DA3 point cloud and numeric projection package."
        )
    )
    parser.add_argument("--da3-dir", type=Path, required=True)
    parser.add_argument("--camera-metadata-json", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--max-points", type=int, default=3000000)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def load_da3_names(da3_dir: Path, count: int) -> list[str]:
    meta = json.loads((da3_dir / "meta.json").read_text(encoding="utf-8"))
    names = [Path(str(item)).name for item in (meta.get("image_names") or meta.get("images") or [])]
    if len(names) != count:
        raise ValueError(f"Expected {count} DA3 image names, found {len(names)}")
    return names


def camera_arrays(camera_metadata: dict, names: list[str], depth_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    by_name = {Path(str(view["file_name"])).name: view for view in camera_metadata.get("views", [])}
    depth_h, depth_w = depth_shape
    output = camera_metadata.get("output_resolution", {})
    image_w = float(output.get("width", depth_w))
    image_h = float(output.get("height", depth_h))
    sx = float(depth_w) / max(image_w, 1.0)
    sy = float(depth_h) / max(image_h, 1.0)
    flip_cv_to_y_up = np.diag([1.0, -1.0, 1.0])
    intrinsics = []
    extrinsics = []
    for name in names:
        if name not in by_name:
            raise KeyError(f"No camera metadata for {name}")
        view = by_name[name]
        k = np.asarray(view["intrinsic_matrix_K"], dtype=np.float64).copy()
        k[0, :] *= sx
        k[1, :] *= sy
        r_camera_to_panorama = np.asarray(view["rotation_perspective_camera_to_panorama"], dtype=np.float64)
        r_camera_to_world_cv = r_camera_to_panorama @ flip_cv_to_y_up
        r_world_to_camera_cv = r_camera_to_world_cv.T
        e = np.concatenate([r_world_to_camera_cv, np.zeros((3, 1), dtype=np.float64)], axis=1)
        intrinsics.append(k)
        extrinsics.append(e)
    return np.stack(intrinsics).astype(np.float32), np.stack(extrinsics).astype(np.float32)


def resize_rgb(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.uint8)


def reconstruct_points(
    depth: np.ndarray,
    conf: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    names: list[str],
    image_dir: Path,
    stride: int,
    conf_percentile: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    point_chunks = []
    color_chunks = []
    per_view = []
    for idx, name in enumerate(names):
        zmap = np.asarray(depth[idx], dtype=np.float64)
        cmap = np.asarray(conf[idx], dtype=np.float64)
        h, w = zmap.shape
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        z = zmap[yy, xx]
        sampled_conf = cmap[yy, xx]
        finite_conf = sampled_conf[np.isfinite(sampled_conf)]
        threshold = float(np.percentile(finite_conf, conf_percentile)) if finite_conf.size else -np.inf
        keep = np.isfinite(z) & (z > 1e-6) & np.isfinite(sampled_conf) & (sampled_conf >= threshold)
        if not np.any(keep):
            per_view.append({"image": name, "points": 0, "conf_threshold": threshold})
            continue
        k = intrinsics[idx].astype(np.float64)
        x_cam = (xx[keep].astype(np.float64) - k[0, 2]) / max(k[0, 0], 1e-8) * z[keep]
        y_cam = (yy[keep].astype(np.float64) - k[1, 2]) / max(k[1, 1], 1e-8) * z[keep]
        points_camera = np.column_stack([x_cam, y_cam, z[keep]])
        e = extrinsics[idx].astype(np.float64)
        points_world = (points_camera - e[:, 3][None, :]) @ e[:, :3]
        rgb = resize_rgb(image_dir / name, (w, h))[yy[keep], xx[keep]]
        point_chunks.append(points_world.astype(np.float32))
        color_chunks.append(rgb)
        per_view.append({"image": name, "points": int(points_world.shape[0]), "conf_threshold": threshold})
    if not point_chunks:
        raise RuntimeError("No DA3 points survived known-camera reconstruction")
    return np.concatenate(point_chunks), np.concatenate(color_chunks), {"views": per_view}


def save_glb(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    rgba = np.concatenate([colors.astype(np.uint8), np.full((colors.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    scene = trimesh.Scene(trimesh.points.PointCloud(points.astype(np.float32), colors=rgba))
    path.write_bytes(scene.export(file_type="glb"))
    gltf = GLTF2().load_binary(str(path))
    scene_index = int(gltf.scene or 0)
    gltf.scenes[scene_index].extras = {"hf_alignment": np.eye(4, dtype=float).tolist()}
    gltf.save_binary(str(path))


def main() -> int:
    args = parse_args()
    depth = np.load(args.da3_dir / "depth.npy").astype(np.float32)
    conf = np.load(args.da3_dir / "conf.npy").astype(np.float32)
    if depth.ndim != 3 or conf.shape != depth.shape:
        raise ValueError(f"Invalid DA3 depth/conf shapes: {depth.shape}, {conf.shape}")
    names = load_da3_names(args.da3_dir, depth.shape[0])
    camera_metadata = json.loads(args.camera_metadata_json.read_text(encoding="utf-8"))
    intrinsics, extrinsics = camera_arrays(camera_metadata, names, tuple(depth.shape[1:3]))
    points, colors, reconstruction_meta = reconstruct_points(
        depth,
        conf,
        intrinsics,
        extrinsics,
        names,
        args.image_dir,
        max(1, int(args.stride)),
        float(args.conf_percentile),
    )
    if points.shape[0] > int(args.max_points):
        rng = np.random.default_rng(int(args.seed))
        indices = np.sort(rng.choice(points.shape[0], int(args.max_points), replace=False))
        points = points[indices]
        colors = colors[indices]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "depth.npy", depth)
    np.save(args.out_dir / "conf.npy", conf)
    np.save(args.out_dir / "intrinsics.npy", intrinsics)
    np.save(args.out_dir / "extrinsics.npy", extrinsics)
    save_glb(args.out_dir / "scene.glb", points, colors)

    source_meta = json.loads((args.da3_dir / "meta.json").read_text(encoding="utf-8"))
    meta = dict(source_meta)
    meta.update(
        {
            "source_da3_dir": str(args.da3_dir),
            "camera_metadata_json": str(args.camera_metadata_json),
            "image_dir": str(args.image_dir),
            "image_names": names,
            "num_images": len(names),
            "camera_pose_source": "known_same_center_camera_metadata",
            "extrinsics_convention": "w2c_opencv",
            "intrinsics_coordinate_space": "processed_images",
            "hf_alignment": np.eye(4, dtype=float).tolist(),
            "reconstructed_point_count": int(points.shape[0]),
            "reconstruction": reconstruction_meta,
        }
    )
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "points": int(points.shape[0]), "views": len(names)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
