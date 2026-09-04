#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from build_polygon_photo_source_from_colmap import depth_normal_map_world, load_hf_alignment, to_4x4
from diagnose_da3_pose_hf_candidates import da3_names
from fit_manhattan_l_room_from_da3_glb import room_to_world_points, structure_lines, world_to_room_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit a variable-corner Manhattan room shell from unmasked DA3 depth observations."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--da3-dir", type=Path, required=True)
    parser.add_argument("--structure-json", type=Path, required=True)
    parser.add_argument("--reject-mask-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--view-stride", type=int, default=1)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--wall-band", type=float, default=0.14)
    parser.add_argument("--floor-band", type=float, default=0.08)
    parser.add_argument("--ceiling-band", type=float, default=0.08)
    parser.add_argument("--height-margin", type=float, default=0.08)
    parser.add_argument("--along-margin", type=float, default=0.08)
    parser.add_argument("--wall-percentile", type=float, default=8.0)
    parser.add_argument("--floor-percentile", type=float, default=12.0)
    parser.add_argument("--ceiling-percentile", type=float, default=88.0)
    parser.add_argument("--max-wall-shift", type=float, default=0.18)
    parser.add_argument(
        "--max-local-wall-shift-ratio",
        type=float,
        default=0.0,
        help=(
            "Reject a wall-plane update when its absolute shift exceeds this fraction of the "
            "smallest length among the wall and its two neighbours. Zero preserves the legacy "
            "unconstrained per-wall refit. This guards short variable-topology returns without "
            "changing the robust percentile target used by v3b."
        ),
    )
    parser.add_argument("--max-y-shift", type=float, default=0.08)
    parser.add_argument("--min-wall-points", type=int, default=1200)
    parser.add_argument("--min-floor-points", type=int, default=2000)
    parser.add_argument(
        "--normal-guided-vertical-bounds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Initialize floor/ceiling heights from dominant horizontal DA3 depth-normal "
            "modes before applying the original one-step robust percentile refit."
        ),
    )
    parser.add_argument("--horizontal-normal-min-cos", type=float, default=0.90)
    parser.add_argument("--horizontal-mode-min-fraction", type=float, default=0.015)
    parser.add_argument("--plot-points", type=int, default=250000)
    return parser.parse_args()


def polygon_area(poly: np.ndarray) -> float:
    return 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - poly[:, 1] * np.roll(poly[:, 0], -1)))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if polygon_area(poly) >= 0.0 else poly[::-1].copy()


def polygon_contains(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    contour = poly.astype(np.float32).reshape(-1, 1, 2)
    out = np.zeros(pts.shape[0], dtype=bool)
    for i, p in enumerate(pts.astype(np.float32)):
        out[i] = cv2.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= -1e-6
    return out


def edge_outward_normal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    edge = b - a
    tangent = edge / max(float(np.linalg.norm(edge)), 1e-8)
    return np.array([tangent[1], -tangent[0]], dtype=np.float64)


def load_reject_mask(mask_dir: Path | None, idx: int, name: str, depth_shape: tuple[int, int]) -> np.ndarray:
    if mask_dir is None:
        return np.zeros(depth_shape, dtype=bool)
    stem = Path(name).stem
    keys = [
        f"{stem}_object_mask.png",
        f"view_{idx:03d}_object_mask.png",
        f"view_{idx:06d}_object_mask.png",
        f"{idx:03d}_object_mask.png",
        f"{idx:06d}_object_mask.png",
    ]
    path = next((mask_dir / key for key in keys if (mask_dir / key).exists()), None)
    if path is None:
        return np.zeros(depth_shape, dtype=bool)
    image = Image.open(path).convert("L").resize((depth_shape[1], depth_shape[0]), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def backproject_depth_to_glb(
    depth: np.ndarray,
    k: np.ndarray,
    e_w2c: np.ndarray,
    hf: np.ndarray,
    stride: int,
    keep_mask: np.ndarray,
) -> np.ndarray:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[yy, xx].reshape(-1).astype(np.float64)
    u = xx.reshape(-1).astype(np.float64)
    v = yy.reshape(-1).astype(np.float64)
    keep = keep_mask[yy, xx].reshape(-1)
    valid = keep & np.isfinite(z) & (z > 1e-6)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    z = z[valid]
    u = u[valid]
    v = v[valid]
    if k.shape == (4,):
        fx, fy, cx, cy = map(float, k)
        skew = 0.0
    else:
        fx = float(k[0, 0])
        skew = float(k[0, 1])
        fy = float(k[1, 1])
        cx = float(k[0, 2])
        cy = float(k[1, 2])
    y_cam = (v - cy) / max(fy, 1e-8) * z
    x_cam = (u - cx - skew * (y_cam / np.maximum(z, 1e-8))) / max(fx, 1e-8) * z
    cam = np.column_stack([x_cam, y_cam, z])
    e = to_4x4(e_w2c)
    r = e[:3, :3]
    t = e[:3, 3]
    raw_world = (cam - t[None, :]) @ r
    homog = np.concatenate([raw_world, np.ones((raw_world.shape[0], 1), dtype=np.float64)], axis=1)
    glb = homog @ hf.T
    return glb[:, :3]


def collect_observed_room_points(args: argparse.Namespace, seed: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    depth = np.load(args.da3_dir / "depth.npy").astype(np.float32)
    conf_path = args.da3_dir / "conf.npy"
    conf = np.load(conf_path).astype(np.float32) if conf_path.exists() else None
    intrinsics = np.load(args.da3_dir / "intrinsics.npy").astype(np.float64)
    extrinsics = np.load(args.da3_dir / "extrinsics.npy").astype(np.float64)
    names = da3_names(args.da3_dir, depth.shape[0])
    hf = load_hf_alignment(args.da3_dir / "scene.glb")
    room_from_world = np.asarray(seed["room_from_world_matrix"], dtype=np.float64)
    chunks: list[np.ndarray] = []
    horizontal_chunks: list[np.ndarray] = []
    stats: list[dict] = []
    for idx in range(0, depth.shape[0], max(1, int(args.view_stride))):
        keep = np.ones(depth[idx].shape, dtype=bool)
        if conf is not None:
            keep &= conf[idx] >= float(args.min_conf)
        reject = load_reject_mask(args.reject_mask_dir, idx, names[idx], depth[idx].shape)
        keep &= ~reject
        pts_glb = backproject_depth_to_glb(depth[idx], intrinsics[idx], extrinsics[idx], hf, max(1, args.stride), keep)
        if pts_glb.size:
            chunks.append(world_to_room_points(pts_glb, room_from_world))
        horizontal_samples = 0
        if args.normal_guided_vertical_bounds:
            c2w = np.linalg.inv(to_4x4(extrinsics[idx]))
            normals_raw, normals_valid = depth_normal_map_world(depth[idx], intrinsics[idx], c2w)
            linear = room_from_world[:3, :3] @ hf[:3, :3]
            normals_room = normals_raw.astype(np.float64) @ linear.T
            normals_room /= np.maximum(np.linalg.norm(normals_room, axis=2, keepdims=True), 1e-8)
            horizontal_keep = (
                keep
                & (normals_valid >= 0.5)
                & (np.abs(normals_room[..., 1]) >= float(args.horizontal_normal_min_cos))
            )
            horizontal_glb = backproject_depth_to_glb(
                depth[idx],
                intrinsics[idx],
                extrinsics[idx],
                hf,
                max(1, args.stride),
                horizontal_keep,
            )
            if horizontal_glb.size:
                horizontal_room = world_to_room_points(horizontal_glb, room_from_world)
                horizontal_chunks.append(horizontal_room)
                horizontal_samples = int(horizontal_room.shape[0])
        stats.append(
            {
                "view": int(idx),
                "image": Path(names[idx]).name,
                "kept_samples": int(pts_glb.shape[0]),
                "horizontal_normal_samples": horizontal_samples,
                "reject_fraction": float(np.count_nonzero(reject) / max(1, reject.size)),
            }
        )
    if not chunks:
        raise RuntimeError("No observed DA3 depth points survived mask/conf filters")
    horizontal = np.concatenate(horizontal_chunks, axis=0) if horizontal_chunks else np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0), horizontal, stats


def dominant_height_mode(
    values: np.ndarray,
    bin_width: float,
    minimum_count: int,
) -> tuple[float | None, dict]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < minimum_count:
        return None, {"samples": int(values.size), "reason": "insufficient_samples"}
    low = float(np.percentile(values, 0.5))
    high = float(np.percentile(values, 99.5))
    if high <= low + 1e-8:
        return float(np.median(values)), {"samples": int(values.size), "support": int(values.size)}
    bins = max(16, int(np.ceil((high - low) / max(bin_width, 1e-5))))
    hist, edges = np.histogram(values, bins=bins, range=(low, high))
    peak = int(np.argmax(hist))
    center = 0.5 * float(edges[peak] + edges[peak + 1])
    radius = max(float(bin_width) * 1.5, float(edges[peak + 1] - edges[peak]) * 1.5)
    support = values[np.abs(values - center) <= radius]
    if support.size < minimum_count:
        return None, {
            "samples": int(values.size),
            "support": int(support.size),
            "reason": "insufficient_mode_support",
        }
    return float(np.median(support)), {
        "samples": int(values.size),
        "support": int(support.size),
        "histogram_peak_center": center,
        "histogram_bin_width": float(edges[1] - edges[0]),
        "mode_median": float(np.median(support)),
    }


def initialize_vertical_bounds_from_horizontal_normals(
    seed: dict,
    horizontal_points_room: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, dict]:
    result = dict(seed)
    result["vertical_bounds"] = dict(seed["vertical_bounds"])
    if not args.normal_guided_vertical_bounds or horizontal_points_room.size == 0:
        return result, {"enabled": bool(args.normal_guided_vertical_bounds), "applied": False}

    room_from_world = np.asarray(seed["room_from_world_matrix"], dtype=np.float64)
    camera_room_y = float(world_to_room_points(np.zeros((1, 3), dtype=np.float64), room_from_world)[0, 1])
    heights = horizontal_points_room[:, 1]
    seed_floor = float(seed["vertical_bounds"]["floor_y"])
    seed_ceiling = float(seed["vertical_bounds"]["ceiling_y"])
    seed_height = max(seed_ceiling - seed_floor, 1e-6)
    separation = 0.12 * seed_height
    below = heights[heights < camera_room_y - separation]
    above = heights[heights > camera_room_y + separation]
    minimum_count = max(128, int(round(float(args.horizontal_mode_min_fraction) * max(heights.size, 1))))
    bin_width = max(0.005, 0.012 * seed_height)
    floor_mode, floor_stats = dominant_height_mode(below, bin_width, minimum_count)
    ceiling_mode, ceiling_stats = dominant_height_mode(above, bin_width, minimum_count)
    applied = bool(
        floor_mode is not None
        and ceiling_mode is not None
        and floor_mode < camera_room_y - separation
        and ceiling_mode > camera_room_y + separation
        and ceiling_mode - floor_mode >= 0.55 * seed_height
    )
    if applied:
        result["vertical_bounds"]["floor_y"] = float(floor_mode)
        result["vertical_bounds"]["ceiling_y"] = float(ceiling_mode)
        result["vertical_bounds"]["height"] = float(ceiling_mode - floor_mode)
    return result, {
        "enabled": True,
        "applied": applied,
        "method": "dominant_horizontal_DA3_depth_normal_modes_then_original_robust_refit",
        "camera_room_y": camera_room_y,
        "seed_floor_y": seed_floor,
        "seed_ceiling_y": seed_ceiling,
        "horizontal_samples": int(heights.size),
        "minimum_mode_support": int(minimum_count),
        "floor_mode": floor_mode,
        "ceiling_mode": ceiling_mode,
        "floor_mode_stats": floor_stats,
        "ceiling_mode_stats": ceiling_stats,
    }


def robust_coord(vals: np.ndarray, outward_sign: float, pct: float) -> float:
    if outward_sign < 0:
        return float(np.percentile(vals, pct))
    return float(np.percentile(vals, 100.0 - pct))


def refit(seed: dict, points_room: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float, float, list[dict]]:
    poly = ensure_ccw(np.asarray(seed["floor_polygon_xz"], dtype=np.float64))
    floor_y = float(seed["vertical_bounds"]["floor_y"])
    ceiling_y = float(seed["vertical_bounds"]["ceiling_y"])
    xz = points_room[:, [0, 2]]
    y = points_room[:, 1]
    inside = polygon_contains(poly, xz)
    stats: list[dict] = []

    floor_pts = y[inside & (np.abs(y - floor_y) <= float(args.floor_band))]
    if floor_pts.size >= int(args.min_floor_points):
        new_floor = float(np.percentile(floor_pts, float(args.floor_percentile)))
        floor_shift = float(np.clip(new_floor - floor_y, -args.max_y_shift, args.max_y_shift))
        floor_y += floor_shift
    else:
        floor_shift = 0.0
    stats.append({"face": "floor", "points": int(floor_pts.size), "shift": floor_shift, "y": floor_y})

    ceil_pts = y[inside & (np.abs(y - ceiling_y) <= float(args.ceiling_band))]
    if ceil_pts.size >= int(args.min_floor_points):
        new_ceil = float(np.percentile(ceil_pts, float(args.ceiling_percentile)))
        ceil_shift = float(np.clip(new_ceil - ceiling_y, -args.max_y_shift, args.max_y_shift))
        ceiling_y += ceil_shift
    else:
        ceil_shift = 0.0
    stats.append({"face": "ceiling", "points": int(ceil_pts.size), "shift": ceil_shift, "y": ceiling_y})

    out = poly.copy()
    edge_lengths = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    mid_height = (y >= floor_y + args.height_margin) & (y <= ceiling_y - args.height_margin)
    for i, (a, b) in enumerate(zip(poly, np.roll(poly, -1, axis=0))):
        edge = b - a
        length = max(float(np.linalg.norm(edge)), 1e-8)
        tangent = edge / length
        normal = edge_outward_normal(a, b)
        rel = xz - a[None, :]
        along = rel @ tangent
        signed = rel @ normal
        band = (
            mid_height
            & (along >= -float(args.along_margin))
            & (along <= length + float(args.along_margin))
            & (np.abs(signed) <= float(args.wall_band))
        )
        pts = xz[band]
        local_scale = float(
            min(
                edge_lengths[(i - 1) % len(poly)],
                edge_lengths[i],
                edge_lengths[(i + 1) % len(poly)],
            )
        )
        item = {
            "face": f"wall_{i:02d}",
            "points": int(pts.shape[0]),
            "shift": 0.0,
            "axis": None,
            "edge_length": length,
            "local_scale": local_scale,
        }
        if pts.shape[0] < int(args.min_wall_points):
            stats.append(item)
            continue
        axis: int | None = None
        axis_name: str | None = None
        if abs(edge[0]) < abs(edge[1]) * 0.08:
            old = float(a[0])
            target = robust_coord(pts[:, 0], float(normal[0]), float(args.wall_percentile))
            axis = 0
            axis_name = "x"
        elif abs(edge[1]) < abs(edge[0]) * 0.08:
            old = float(a[1])
            target = robust_coord(pts[:, 1], float(normal[1]), float(args.wall_percentile))
            axis = 1
            axis_name = "z"
        if axis is not None and axis_name is not None:
            proposed_shift = float(np.clip(target - old, -args.max_wall_shift, args.max_wall_shift))
            local_shift_ratio = abs(proposed_shift) / max(local_scale, 1e-8)
            accepted = bool(
                float(args.max_local_wall_shift_ratio) <= 0.0
                or local_shift_ratio <= float(args.max_local_wall_shift_ratio)
            )
            shift = proposed_shift if accepted else 0.0
            out[[i, (i + 1) % len(poly)], axis] += shift
            item.update(
                {
                    "axis": axis_name,
                    "old": old,
                    "target": target,
                    "proposed_shift": proposed_shift,
                    "local_shift_ratio": local_shift_ratio,
                    "accepted": accepted,
                    "rejection_reason": None if accepted else "local_shift_ratio_exceeded",
                    "shift": shift,
                }
            )
        stats.append(item)
    return ensure_ccw(out), floor_y, ceiling_y, stats


def rebuild_structure(seed: dict, poly: np.ndarray, floor_y: float, ceiling_y: float) -> dict:
    world_from_room = np.asarray(seed["world_from_room_matrix"], dtype=np.float64)
    room_from_world = np.asarray(seed["room_from_world_matrix"], dtype=np.float64)
    floor = np.column_stack([poly[:, 0], np.full(len(poly), floor_y), poly[:, 1]])
    ceil = np.column_stack([poly[:, 0], np.full(len(poly), ceiling_y), poly[:, 1]])
    room_vertices = np.vstack([floor, ceil])
    vertices = room_to_world_points(room_vertices, world_from_room)
    n = len(poly)
    faces = [
        {"name": "floor", "type": "floor", "vertices": list(range(n))},
        {"name": "ceiling", "type": "ceiling", "vertices": list(range(2 * n - 1, n - 1, -1))},
    ]
    for i in range(n):
        j = (i + 1) % n
        faces.append({"name": f"wall_{i:02d}", "type": "wall", "vertices": [i, j, j + n, i + n]})
    out = dict(seed)
    out.update(
        {
            "method": "da3_observed_unmasked_surface_refit_manhattan_polygon_v2",
            "vertices": vertices.tolist(),
            "room_vertices": room_vertices.tolist(),
            "faces": faces,
            "floor_corner_count": int(n),
            "ceiling_corner_count": int(n),
            "floor_polygon_xz": poly.tolist(),
            "vertical_bounds": {
                "floor_y": float(floor_y),
                "ceiling_y": float(ceiling_y),
                "percentiles": seed.get("vertical_bounds", {}).get("percentiles", [2.0, 98.0]),
            },
            "world_from_room_matrix": world_from_room.tolist(),
            "room_from_world_matrix": room_from_world.tolist(),
        }
    )
    return out


def write_obj(path: Path, structure: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# DA3 observed-surface refit variable-corner Manhattan room structure\n")
        for v in structure["vertices"]:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in structure["faces"]:
            f.write(f"o {face['name']}\n")
            f.write("f " + " ".join(str(int(i) + 1) for i in face["vertices"]) + "\n")


def plot(path: Path, points_room: np.ndarray, before: dict, after: dict, max_points: int) -> None:
    sample = points_room
    if sample.shape[0] > max_points:
        idx = np.linspace(0, sample.shape[0] - 1, max_points).astype(np.int64)
        sample = sample[idx]
    bpoly = np.asarray(before["floor_polygon_xz"], dtype=float)
    apoly = np.asarray(after["floor_polygon_xz"], dtype=float)
    after_room = dict(after)
    after_room["vertices"] = after["room_vertices"]
    fig = plt.figure(figsize=(16, 7), dpi=170)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.scatter(sample[:, 0], sample[:, 2], s=0.08, c="#666666", alpha=0.14)
    ax1.plot(np.r_[bpoly[:, 0], bpoly[0, 0]], np.r_[bpoly[:, 1], bpoly[0, 1]], "-o", c="#e53935", lw=1.2, ms=2, label="before")
    ax1.plot(np.r_[apoly[:, 0], apoly[0, 0]], np.r_[apoly[:, 1], apoly[0, 1]], "-o", c="#14a44d", lw=1.8, ms=2, label="after")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Top x-z unmasked DA3 observed-surface refit")
    ax1.legend(loc="best")
    ax2.scatter(sample[:, 2], sample[:, 1], s=0.08, c="#666666", alpha=0.13)
    for line in structure_lines(after_room):
        ax2.plot(line[:, 2], line[:, 1], c="#14a44d", lw=1.4)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Leveled side z-y after observed-surface refit")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed = json.loads(args.structure_json.read_text(encoding="utf-8"))
    points_room, horizontal_points_room, view_stats = collect_observed_room_points(args, seed)
    initialized_seed, vertical_initialization = initialize_vertical_bounds_from_horizontal_normals(
        seed,
        horizontal_points_room,
        args,
    )
    poly, floor_y, ceiling_y, stats = refit(initialized_seed, points_room, args)
    refined = rebuild_structure(initialized_seed, poly, floor_y, ceiling_y)
    refined["refit_stats"] = stats
    refined["vertical_bound_initialization"] = vertical_initialization
    refined["view_sample_stats"] = view_stats
    refined["refit_params"] = {
        "stride": int(args.stride),
        "view_stride": int(args.view_stride),
        "min_conf": float(args.min_conf),
        "wall_band": float(args.wall_band),
        "floor_band": float(args.floor_band),
        "ceiling_band": float(args.ceiling_band),
        "wall_percentile": float(args.wall_percentile),
        "max_local_wall_shift_ratio": float(args.max_local_wall_shift_ratio),
        "normal_guided_vertical_bounds": bool(args.normal_guided_vertical_bounds),
        "horizontal_normal_min_cos": float(args.horizontal_normal_min_cos),
        "horizontal_mode_min_fraction": float(args.horizontal_mode_min_fraction),
        "reject_mask_dir": None if args.reject_mask_dir is None else str(args.reject_mask_dir),
    }
    out_json = args.out_dir / "structure_da3_manhattan_polygon_observed_refit.json"
    out_json.write_text(json.dumps(refined, indent=2, ensure_ascii=False), encoding="utf-8")
    write_obj(args.out_dir / "room_structure_da3_manhattan_polygon_observed_refit.obj", refined)
    plot(args.out_dir / "structure_da3_manhattan_polygon_observed_refit_comparison.png", points_room, seed, refined, args.plot_points)
    print(json.dumps({"out_json": str(out_json), "point_count": int(points_room.shape[0]), "stats": stats}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
