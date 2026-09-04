#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from build_polygon_photo_source_from_colmap import (
    apply_depth_calibration,
    build_face_id_and_zbuffer,
    calibrate_da3_depth_to_colmap_zbuffer,
    face_meta_map,
    face_names,
    load_da3_hfalign_poses,
    load_da3_views,
    load_hf_alignment,
    rotate_points_to_floorplan,
    to_4x4,
)
from build_polygon_room_texture_from_da3_glb import points_in_polygon
from polygon_projection_utils import face_id_to_rgb, face_palette, image_keys

RESAMPLE_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-view room-face ids by back-projecting DA3 depth pixels "
            "to 3D and assigning each observed point to the nearest valid "
            "polygon room face."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--da3-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--faces", default=None)
    parser.add_argument(
        "--coordinate-space",
        choices=["raw", "hfalign"],
        default="raw",
        help="raw for structures fitted in DA3 numeric world; hfalign for scene.glb structures.",
    )
    parser.add_argument("--surface-distance-tol", type=float, default=0.075)
    parser.add_argument("--inside-margin", type=float, default=0.04)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--min-conf", type=float, default=0.0)
    parser.add_argument("--close-px", type=int, default=1)
    parser.add_argument("--dilate-px", type=int, default=1)
    parser.add_argument(
        "--calibrate-depth-to-polygon-zbuffer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fit a robust per-view linear/inverse DA3-depth transform to the known "
            "polygon-shell z-buffer before assigning observed pixels to room faces. "
            "This keeps separately sampled low-resolution perspective views on the "
            "same metric room shell without replacing their DA3 depth discontinuities."
        ),
    )
    parser.add_argument(
        "--require-depth-calibration",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Leave a view unassigned when the requested robust depth calibration fails.",
    )
    parser.add_argument("--zbuffer-stride", type=int, default=2)
    return parser.parse_args()


def read_manifest(source_dir: Path) -> dict:
    path = source_dir / "metadata.json"
    if not path.exists():
        path = source_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def da3_image_names(da3_dir: Path, count: int) -> list[str]:
    meta = json.loads((da3_dir / "meta.json").read_text(encoding="utf-8"))
    raw = meta.get("image_names") or meta.get("images") or meta.get("image_paths") or []
    names = [Path(str(x)).name for x in raw]
    if len(names) == count:
        return names
    return [f"{idx:06d}.png" for idx in range(count)]


def intrinsic_params(k: np.ndarray) -> tuple[float, float, float, float, float]:
    k = np.asarray(k, dtype=np.float64)
    if k.shape == (4,):
        fx, fy, cx, cy = map(float, k)
        return fx, fy, cx, cy, 0.0
    if k.shape == (3, 3):
        return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), float(k[0, 1])
    raise ValueError(f"Unsupported DA3 intrinsic shape: {k.shape}")


def unproject_da3_depth_points(depth: np.ndarray, intrinsic: np.ndarray, extrinsic_w2c: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    z = depth.reshape(-1).astype(np.float64)
    xpix = xx.reshape(-1)
    ypix = yy.reshape(-1)
    fx, fy, cx, cy, skew = intrinsic_params(intrinsic)
    y = (ypix - cy) / max(fy, 1e-8) * z
    x = (xpix - cx - skew * (y / np.maximum(z, 1e-8))) / max(fx, 1e-8) * z
    cam = np.stack([x, y, z], axis=1)
    e = to_4x4(extrinsic_w2c)
    r = e[:3, :3]
    t = e[:3, 3]
    return (cam - t[None, :]) @ r


def face_inside_and_distance(points: np.ndarray, face: str, manifest: dict, metas: dict, margin: float) -> tuple[np.ndarray, np.ndarray]:
    horiz, heights = rotate_points_to_floorplan(points, manifest)
    floor_y = float(manifest["floor_y"])
    ceiling_y = float(manifest["ceiling_y"])
    if face in {"floor", "ceiling"}:
        plane_y = floor_y if face == "floor" else ceiling_y
        dist = np.abs(heights - plane_y)
        poly = np.asarray(manifest["floorplan_polygon_uv"], dtype=np.float32)
        inside = points_in_polygon(horiz.astype(np.float32), poly)
        if margin > 0:
            bounds = np.asarray(manifest["bounds_uv"], dtype=np.float64)
            inside &= np.all(horiz >= bounds[0][None, :] - margin, axis=1)
            inside &= np.all(horiz <= bounds[1][None, :] + margin, axis=1)
        return inside, dist
    meta = metas[face]
    a = np.asarray(meta["edge_start"], dtype=np.float64)
    b = np.asarray(meta["edge_end"], dtype=np.float64)
    edge = b - a
    length = max(float(np.linalg.norm(edge)), 1e-8)
    edir = edge / length
    rel = horiz - a[None, :]
    along = rel @ edir
    perp = np.abs(rel[:, 0] * edir[1] - rel[:, 1] * edir[0])
    inside = (along >= -margin) & (along <= length + margin)
    inside &= heights >= floor_y - margin
    inside &= heights <= ceiling_y + margin
    return inside, perp


def smooth_face_id(face_id: np.ndarray, faces: list[str], close_px: int, dilate_px: int) -> np.ndarray:
    out = face_id.copy()
    for idx in range(len(faces)):
        mask = out == idx
        if not np.any(mask):
            continue
        if close_px > 0:
            k = np.ones((2 * close_px + 1, 2 * close_px + 1), dtype=np.uint8)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=1).astype(bool)
        if dilate_px > 0:
            k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=np.uint8)
            grown = cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)
            fill = (out == 255) & grown
            out[fill] = idx
    return out


def main() -> int:
    args = parse_args()
    manifest = read_manifest(args.source_dir)
    metas = face_meta_map(manifest)
    faces = face_names(manifest, args.faces)
    depth = np.load(args.da3_dir / "depth.npy").astype(np.float32)
    conf_path = args.da3_dir / "conf.npy"
    conf = np.load(conf_path).astype(np.float32) if conf_path.exists() else None
    intrinsics = np.load(args.da3_dir / "intrinsics.npy").astype(np.float64)
    extrinsics = np.load(args.da3_dir / "extrinsics.npy").astype(np.float64)
    names = da3_image_names(args.da3_dir, depth.shape[0])
    hf_scene = args.da3_dir / "scene.glb"
    if not hf_scene.exists():
        hf_scene = args.dataset_dir / "scene.glb"
    hf = load_hf_alignment(hf_scene) if args.coordinate_space == "hfalign" else None
    image_dir = args.dataset_dir / "input_images"
    view_dir = args.out_dir / "view_face_masks"
    view_dir.mkdir(parents=True, exist_ok=True)
    palette = face_palette(faces)
    calibration_poses = {}
    calibration_views = {}
    calibration_similarity = None
    if args.calibrate_depth_to_polygon_zbuffer:
        poses, calibration_similarity, _ = load_da3_hfalign_poses(args.dataset_dir, args.da3_dir)
        calibration_poses = {int(pose.image_id): pose for pose in poses}
        calibration_views = load_da3_views(args.da3_dir, poses)
    stats = []
    for idx, name in enumerate(names):
        image_path = image_dir / Path(name).name
        if not image_path.exists():
            continue
        with Image.open(image_path) as im:
            image_size = im.size
        d = depth[idx]
        calibration = None
        if args.calibrate_depth_to_polygon_zbuffer:
            pose = calibration_poses.get(int(idx))
            view = calibration_views.get(int(idx))
            if pose is not None and view is not None and calibration_similarity is not None:
                _, polygon_zbuffer = build_face_id_and_zbuffer(
                    pose,
                    faces,
                    args.source_dir,
                    calibration_similarity,
                    manifest,
                    metas,
                    max(1, int(args.zbuffer_stride)),
                )
                calibration = calibrate_da3_depth_to_colmap_zbuffer(view, pose, polygon_zbuffer)
                if calibration is not None:
                    d = apply_depth_calibration(d, calibration)
        valid = np.isfinite(d.reshape(-1)) & (d.reshape(-1) > args.min_depth)
        if args.require_depth_calibration and calibration is None:
            valid[:] = False
        if conf is not None and args.min_conf > 0:
            valid &= np.isfinite(conf[idx].reshape(-1)) & (conf[idx].reshape(-1) >= args.min_conf)
        pts = unproject_da3_depth_points(d, intrinsics[idx], extrinsics[idx])
        if hf is not None:
            homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
            moved = homog @ hf.T
            pts = moved[:, :3] / np.maximum(np.abs(moved[:, 3:4]), 1e-8)
        best_dist = np.full(pts.shape[0], np.inf, dtype=np.float32)
        best_id = np.full(pts.shape[0], 255, dtype=np.uint16)
        for face_idx, face in enumerate(faces):
            inside, dist = face_inside_and_distance(pts, face, manifest, metas, args.inside_margin)
            ok = valid & inside & np.isfinite(dist) & (dist <= args.surface_distance_tol) & (dist < best_dist)
            best_dist[ok] = dist[ok].astype(np.float32)
            best_id[ok] = face_idx
        face_depth = best_id.reshape(d.shape)
        face_depth = smooth_face_id(face_depth, faces, args.close_px, args.dilate_px)
        face_img = Image.fromarray(face_depth.astype(np.uint16))
        if face_img.size != image_size:
            face_img = face_img.resize(image_size, RESAMPLE_NEAREST)
        face_id = np.asarray(face_img, dtype=np.uint16)
        rgb = face_id_to_rgb(face_id, faces, palette)
        for key in image_keys(type("PoseKey", (), {"image_id": idx, "name": Path(name).name})()):
            Image.fromarray(rgb).save(view_dir / f"{key}_faces.png")
            np.save(view_dir / f"{key}_face_id.npy", face_id.astype(np.uint16))
        stats.append(
            {
                "image_id": int(idx),
                "name": Path(name).name,
                "shape_hw": [int(image_size[1]), int(image_size[0])],
                "covered_pixels": int(np.count_nonzero(face_id != 255)),
                "depth_calibration": (
                    {
                        "mode": calibration.mode,
                        "scale": float(calibration.scale),
                        "shift": float(calibration.shift),
                        "median_abs_error": float(calibration.median_abs_error),
                        "p80_abs_error": float(calibration.p80_abs_error),
                        "samples": int(calibration.samples),
                    }
                    if calibration is not None
                    else None
                ),
            }
        )
        print(f"[da3-face] {Path(name).name}: {stats[-1]['covered_pixels']} px", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "method": "da3_depth_point_to_polygon_room_face_v1",
                "dataset_dir": str(args.dataset_dir),
                "source_dir": str(args.source_dir),
                "da3_dir": str(args.da3_dir),
                "coordinate_space": args.coordinate_space,
                "faces": faces,
                "surface_distance_tol": args.surface_distance_tol,
                "inside_margin": args.inside_margin,
                "calibrate_depth_to_polygon_zbuffer": bool(args.calibrate_depth_to_polygon_zbuffer),
                "require_depth_calibration": bool(args.require_depth_calibration),
                "zbuffer_stride": int(args.zbuffer_stride),
                "views": stats,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
