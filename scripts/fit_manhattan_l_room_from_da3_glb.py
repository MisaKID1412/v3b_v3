#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an orthogonal 6-corner L-shaped room footprint from a DA3 exported "
            "GLB point cloud. This is intentionally more constrained than contour "
            "approximation: walls are Manhattan axis-aligned and the footprint is a "
            "rectangle with one missing corner."
        )
    )
    parser.add_argument("--scene-glb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=720)
    parser.add_argument("--bounds-percentiles", type=float, nargs=2, default=[0.6, 99.4])
    parser.add_argument("--vertical-percentiles", type=float, nargs=2, default=[2.0, 98.0])
    parser.add_argument("--occupancy-threshold-quantile", type=float, default=0.42)
    parser.add_argument("--close-iterations", type=int, default=3)
    parser.add_argument("--dilate-iterations", type=int, default=1)
    parser.add_argument("--notch-samples", type=int, default=94)
    parser.add_argument("--min-notch-frac", type=float, default=0.18)
    parser.add_argument("--max-notch-frac", type=float, default=0.82)
    parser.add_argument("--plot-points", type=int, default=240000)
    parser.add_argument("--scene-name", default="room_empty")
    parser.add_argument("--level-grid", type=int, default=84)
    parser.add_argument("--floor-cell-percentile", type=float, default=6.0)
    parser.add_argument("--ceiling-cell-percentile", type=float, default=94.0)
    parser.add_argument("--min-cell-points", type=int, default=18)
    parser.add_argument("--plane-trim-percentile", type=float, default=68.0)
    parser.add_argument("--plane-fit-iters", type=int, default=7)
    parser.add_argument("--average-ceiling-normal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--forced-up-axis-world",
        type=float,
        nargs=3,
        default=None,
        help="Optional known gravity/up vector in the DA3 world frame.",
    )
    return parser.parse_args()


def load_point_cloud(path: Path) -> np.ndarray:
    scene = trimesh.load(path, force="scene")
    geoms = []
    for geom in scene.geometry.values():
        if hasattr(geom, "vertices"):
            vertices = np.asarray(geom.vertices)
            if vertices.ndim == 2 and vertices.shape[1] == 3 and vertices.shape[0] > 1000:
                geoms.append(vertices)
    if not geoms:
        raise RuntimeError(f"No point cloud geometry found in {path}")
    geoms.sort(key=lambda arr: arr.shape[0], reverse=True)
    return geoms[0].astype(np.float64)


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    z = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - z * np.roll(x, -1)))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if polygon_area(poly) >= 0.0 else poly[::-1].copy()


def start_top_left(poly: np.ndarray) -> np.ndarray:
    order = np.lexsort((poly[:, 0], -poly[:, 1]))
    return np.roll(poly, -int(order[0]), axis=0)


def fit_plane_y_xz(samples: np.ndarray, trim_percentile: float, iters: int) -> tuple[np.ndarray, dict]:
    """Fit y = a*x + b*z + c with iterative residual trimming."""
    if samples.shape[0] < 12:
        raise RuntimeError("Not enough samples for plane fit")
    x = samples[:, 0]
    y = samples[:, 1]
    z = samples[:, 2]
    a_mat = np.column_stack([x, z, np.ones_like(x)])
    keep = np.ones(samples.shape[0], dtype=bool)
    coef = np.linalg.lstsq(a_mat, y, rcond=None)[0]
    history = []
    for _ in range(max(1, int(iters))):
        pred = a_mat @ coef
        resid = np.abs(pred - y)
        thresh = float(np.percentile(resid[keep], trim_percentile)) if np.any(keep) else float(np.percentile(resid, trim_percentile))
        keep = resid <= max(thresh, 1e-8)
        if int(np.count_nonzero(keep)) < 12:
            keep = resid <= np.percentile(resid, 90.0)
        coef = np.linalg.lstsq(a_mat[keep], y[keep], rcond=None)[0]
        history.append(
            {
                "inliers": int(np.count_nonzero(keep)),
                "threshold": thresh,
                "median_abs_residual": float(np.median(resid[keep])) if np.any(keep) else float(np.median(resid)),
            }
        )
    pred = a_mat @ coef
    resid = np.abs(pred - y)
    return coef.astype(np.float64), {
        "samples": int(samples.shape[0]),
        "inliers": int(np.count_nonzero(keep)),
        "median_abs_residual": float(np.median(resid[keep])),
        "p90_abs_residual": float(np.percentile(resid[keep], 90.0)),
        "history": history,
    }


def cell_extreme_samples(
    points: np.ndarray,
    grid: int,
    percentile: float,
    min_points: int,
    bounds_percentiles: tuple[float, float],
) -> np.ndarray:
    xz = points[:, [0, 2]]
    lo = np.percentile(xz, bounds_percentiles[0], axis=0)
    hi = np.percentile(xz, bounds_percentiles[1], axis=0)
    span = np.maximum(hi - lo, 1e-8)
    uv = (xz - lo[None, :]) / span[None, :]
    valid = np.all((uv >= 0.0) & (uv <= 1.0), axis=1)
    ij = np.floor(uv[valid] * (grid - 1)).astype(np.int32)
    src = points[valid]
    buckets: dict[tuple[int, int], list[int]] = {}
    for row, col in enumerate(ij):
        key = (int(col[0]), int(col[1]))
        buckets.setdefault(key, []).append(row)
    out = []
    for (ix, iz), rows in buckets.items():
        if len(rows) < min_points:
            continue
        pts = src[np.asarray(rows, dtype=np.int64)]
        y = float(np.percentile(pts[:, 1], percentile))
        # Use the cell center in x/z and the robust extreme y for a smoother
        # floor/ceiling plane estimate.
        x = float(lo[0] + (ix + 0.5) / grid * span[0])
        z = float(lo[1] + (iz + 0.5) / grid * span[1])
        out.append([x, y, z])
    if not out:
        raise RuntimeError("No cell samples for plane fitting")
    return np.asarray(out, dtype=np.float64)


def plane_normal_from_y_xz(coef: np.ndarray) -> np.ndarray:
    a, b, _ = map(float, coef)
    normal = np.array([-a, 1.0, -b], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    if normal[1] < 0:
        normal *= -1.0
    return normal


def estimate_room_basis(points: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    floor_samples = cell_extreme_samples(
        points,
        int(args.level_grid),
        float(args.floor_cell_percentile),
        int(args.min_cell_points),
        (float(args.bounds_percentiles[0]), float(args.bounds_percentiles[1])),
    )
    ceil_samples = cell_extreme_samples(
        points,
        int(args.level_grid),
        float(args.ceiling_cell_percentile),
        int(args.min_cell_points),
        (float(args.bounds_percentiles[0]), float(args.bounds_percentiles[1])),
    )
    floor_coef, floor_stats = fit_plane_y_xz(floor_samples, float(args.plane_trim_percentile), int(args.plane_fit_iters))
    ceil_coef, ceil_stats = fit_plane_y_xz(ceil_samples, float(args.plane_trim_percentile), int(args.plane_fit_iters))
    floor_n = plane_normal_from_y_xz(floor_coef)
    ceil_n = plane_normal_from_y_xz(ceil_coef)
    forced_up_raw = getattr(args, "forced_up_axis_world", None)
    forced_up = None if forced_up_raw is None else np.asarray(forced_up_raw, dtype=np.float64)
    if forced_up is not None:
        if forced_up.shape != (3,) or not np.all(np.isfinite(forced_up)) or np.linalg.norm(forced_up) < 1e-8:
            raise ValueError("forced_up_axis_world must be a finite non-zero 3-vector")
        up = forced_up / np.linalg.norm(forced_up)
        if up[1] < 0:
            up *= -1.0
        up_source = "known_camera_metadata"
    elif bool(args.average_ceiling_normal) and float(np.dot(floor_n, ceil_n)) > 0.985:
        up = floor_n + ceil_n
        up /= max(float(np.linalg.norm(up)), 1e-8)
        up_source = "averaged_floor_ceiling_plane_fit"
    else:
        up = floor_n
        up_source = "floor_plane_fit"

    world_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = world_x - np.dot(world_x, up) * up
    if np.linalg.norm(x_axis) < 1e-6:
        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x_axis = np.cross(up, world_z)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-8)
    z_axis = np.cross(x_axis, up)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-8)
    if z_axis[2] < 0:
        x_axis *= -1.0
        z_axis *= -1.0

    if forced_up is not None:
        center = np.median(points, axis=0)
        floor_level = float(np.median(floor_samples @ up))
        origin = center + (floor_level - float(center @ up)) * up
    else:
        center_xz = np.median(points[:, [0, 2]], axis=0)
        a, b, c = map(float, floor_coef)
        origin = np.array([center_xz[0], a * center_xz[0] + b * center_xz[1] + c, center_xz[1]], dtype=np.float64)
    rot = np.column_stack([x_axis, up, z_axis])
    world_from_room = np.eye(4, dtype=np.float64)
    world_from_room[:3, :3] = rot
    world_from_room[:3, 3] = origin
    room_from_world = np.linalg.inv(world_from_room)
    meta = {
        "floor_plane_y_eq_ax_bz_c": [float(x) for x in floor_coef],
        "ceiling_plane_y_eq_ax_bz_c": [float(x) for x in ceil_coef],
        "floor_plane_stats": floor_stats,
        "ceiling_plane_stats": ceil_stats,
        "up_axis_world": up.tolist(),
        "up_axis_source": up_source,
        "x_axis_world": x_axis.tolist(),
        "z_axis_world": z_axis.tolist(),
        "origin_world": origin.tolist(),
        "floor_ceiling_normal_dot": float(np.dot(floor_n, ceil_n)),
    }
    return world_from_room, room_from_world, meta


def world_to_room_points(points: np.ndarray, room_from_world: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    out = homog @ room_from_world.T
    return out[:, :3] / np.maximum(np.abs(out[:, 3:4]), 1e-8)


def room_to_world_points(points: np.ndarray, world_from_room: np.ndarray) -> np.ndarray:
    homog = np.concatenate([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    out = homog @ world_from_room.T
    return out[:, :3] / np.maximum(np.abs(out[:, 3:4]), 1e-8)


def build_density_grid(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: int) -> np.ndarray:
    xz = points[:, [0, 2]]
    valid = np.all((xz >= lo[None, :]) & (xz <= hi[None, :]), axis=1)
    uv = (xz[valid] - lo[None, :]) / np.maximum(hi - lo, 1e-8)[None, :]
    ij = np.floor(uv * (size - 1)).astype(np.int32)
    ij = np.clip(ij, 0, size - 1)
    grid = np.zeros((size, size), dtype=np.float32)
    np.add.at(grid, (ij[:, 1], ij[:, 0]), 1.0)
    return grid


def occupancy_from_grid(grid: np.ndarray, q: float, close_iters: int, dilate_iters: int) -> np.ndarray:
    positive = grid[grid > 0]
    if positive.size == 0:
        raise RuntimeError("Empty DA3 top-down density grid")
    threshold = max(1.0, float(np.quantile(positive, q)))
    occ = (grid >= threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    if close_iters > 0:
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))
    if dilate_iters > 0:
        occ = cv2.dilate(occ, kernel, iterations=int(dilate_iters))
    return occ


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return (labels == label).astype(np.uint8) * 255


def contour_bbox(mask: np.ndarray, margin_px: int = 2) -> tuple[int, int, int, int]:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No occupancy contour")
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    h_img, w_img = mask.shape
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(w_img - 1, x + w - 1 + margin_px)
    y1 = min(h_img - 1, y + h - 1 + margin_px)
    return x0, x1, y0, y1


def l_polygon_px(kind: str, x0: int, x1: int, y0: int, y1: int, xn: int, yn: int) -> np.ndarray:
    # Pixel y increases downward. The returned polygon is in image/grid pixels.
    if kind == "missing_top_right":
        poly = [(x0, y0), (xn, y0), (xn, yn), (x1, yn), (x1, y1), (x0, y1)]
    elif kind == "missing_top_left":
        poly = [(x0, yn), (xn, yn), (xn, y0), (x1, y0), (x1, y1), (x0, y1)]
    elif kind == "missing_bottom_right":
        poly = [(x0, y0), (x1, y0), (x1, yn), (xn, yn), (xn, y1), (x0, y1)]
    elif kind == "missing_bottom_left":
        poly = [(x0, y0), (x1, y0), (x1, y1), (xn, y1), (xn, yn), (x0, yn)]
    else:
        raise ValueError(kind)
    return np.asarray(poly, dtype=np.int32)


def raster_poly(shape: tuple[int, int], poly: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(out, [poly.reshape(-1, 1, 2)], 255)
    return out


def score_mask(candidate: np.ndarray, occ: np.ndarray, edge_weight: np.ndarray) -> dict:
    cand = candidate > 0
    obs = occ > 0
    obs_count = max(1, int(np.count_nonzero(obs)))
    cand_count = max(1, int(np.count_nonzero(cand)))
    outside = obs & ~cand
    empty = cand & ~obs
    outside_frac = float(np.count_nonzero(outside) / obs_count)
    empty_frac = float(np.count_nonzero(empty) / cand_count)
    boundary = cv2.morphologyEx(candidate, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if np.any(boundary):
        edge_cost = float(np.mean(edge_weight[boundary]))
        edge_p85 = float(np.percentile(edge_weight[boundary], 85.0))
    else:
        edge_cost = 1.0
        edge_p85 = 1.0
    # Outside observed room evidence is expensive; empty interior is cheaper
    # because furniture and glossy surfaces create holes in top-down occupancy.
    total = 7.5 * outside_frac + 1.25 * empty_frac + 2.0 * edge_cost + 0.7 * edge_p85
    return {
        "score": float(total),
        "outside_observed_frac": outside_frac,
        "empty_candidate_frac": empty_frac,
        "edge_cost": edge_cost,
        "edge_p85": edge_p85,
        "candidate_texels": int(cand_count),
        "observed_texels": int(obs_count),
    }


def fit_l_mask(occ: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    occ = largest_component(occ)
    x0, x1, y0, y1 = contour_bbox(occ, margin_px=max(2, args.grid_size // 220))
    # Distance to observed boundary: low near actual room boundary, high away.
    boundary = cv2.morphologyEx(occ, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    inv_boundary = (boundary == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(inv_boundary, cv2.DIST_L2, 3)
    dist = dist / max(float(np.percentile(dist, 95.0)), 1e-6)
    dist = np.clip(dist, 0.0, 1.0).astype(np.float32)

    minf = float(args.min_notch_frac)
    maxf = float(args.max_notch_frac)
    xs = np.linspace(x0 + (x1 - x0) * minf, x0 + (x1 - x0) * maxf, int(args.notch_samples)).astype(np.int32)
    ys = np.linspace(y0 + (y1 - y0) * minf, y0 + (y1 - y0) * maxf, int(args.notch_samples)).astype(np.int32)
    kinds = ["missing_top_right", "missing_top_left", "missing_bottom_right", "missing_bottom_left"]
    best: tuple[float, np.ndarray, dict] | None = None
    for kind in kinds:
        for xn in xs:
            for yn in ys:
                # Keep the missing rectangle large enough to represent a real L.
                if kind == "missing_top_right":
                    missing_area = max(0, x1 - xn) * max(0, yn - y0)
                elif kind == "missing_top_left":
                    missing_area = max(0, xn - x0) * max(0, yn - y0)
                elif kind == "missing_bottom_right":
                    missing_area = max(0, x1 - xn) * max(0, y1 - yn)
                else:
                    missing_area = max(0, xn - x0) * max(0, y1 - yn)
                bbox_area = max(1, (x1 - x0) * (y1 - y0))
                if missing_area / bbox_area < 0.035:
                    continue
                poly = l_polygon_px(kind, x0, x1, y0, y1, int(xn), int(yn))
                cand = raster_poly(occ.shape, poly)
                item = score_mask(cand, occ, dist)
                item.update(
                    {
                        "kind": kind,
                        "x0": int(x0),
                        "x1": int(x1),
                        "y0": int(y0),
                        "y1": int(y1),
                        "xn": int(xn),
                        "yn": int(yn),
                        "missing_area_frac": float(missing_area / bbox_area),
                    }
                )
                candidate = (item["score"], poly, item)
                if best is None or candidate[0] < best[0]:
                    best = candidate
    if best is None:
        raise RuntimeError("Could not fit an L-shaped room mask")
    return best[1], best[2]


def px_to_xz(poly_px: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: int) -> np.ndarray:
    x = lo[0] + (poly_px[:, 0].astype(np.float64) + 0.5) / size * (hi[0] - lo[0])
    # Pixel y increases downward; grid z increases upward in Matplotlib origin=lower,
    # and our occupancy image uses row index directly from z-bin. Therefore y maps
    # directly to z.
    z = lo[1] + (poly_px[:, 1].astype(np.float64) + 0.5) / size * (hi[1] - lo[1])
    return start_top_left(ensure_ccw(np.column_stack([x, z]).astype(np.float64)))


def make_structure(
    poly_xz: np.ndarray,
    points_room: np.ndarray,
    y_pct: tuple[float, float],
    method: str,
    world_from_room: np.ndarray,
    room_from_world: np.ndarray,
) -> dict:
    y0, y1 = np.percentile(points_room[:, 1], [float(y_pct[0]), float(y_pct[1])])
    floor = np.column_stack([poly_xz[:, 0], np.full(poly_xz.shape[0], y0), poly_xz[:, 1]])
    ceil = np.column_stack([poly_xz[:, 0], np.full(poly_xz.shape[0], y1), poly_xz[:, 1]])
    n = int(poly_xz.shape[0])
    room_vertices = np.vstack([floor, ceil])
    vertices = room_to_world_points(room_vertices, world_from_room)
    faces = [
        {"name": "floor", "type": "floor", "vertices": list(range(n))},
        {"name": "ceiling", "type": "ceiling", "vertices": list(range(2 * n - 1, n - 1, -1))},
    ]
    for i in range(n):
        j = (i + 1) % n
        faces.append({"name": f"wall_{i:02d}", "type": "wall", "vertices": [i, j, j + n, i + n]})
    return {
        "method": method,
        "coordinate_space": "da3_exported_glb",
        "vertices": vertices.tolist(),
        "room_vertices": room_vertices.tolist(),
        "faces": faces,
        "floor_corner_count": n,
        "ceiling_corner_count": n,
        "floor_polygon_xz": poly_xz.tolist(),
        "vertical_bounds": {
            "floor_y": float(y0),
            "ceiling_y": float(y1),
            "percentiles": [float(y_pct[0]), float(y_pct[1])],
        },
        "world_from_room_matrix": world_from_room.tolist(),
        "room_from_world_matrix": room_from_world.tolist(),
    }


def structure_lines(structure: dict) -> list[np.ndarray]:
    vertices = np.asarray(structure["vertices"], dtype=float)
    lines = []
    seen = set()
    for face in structure["faces"]:
        idx = [int(i) for i in face["vertices"]]
        for a, b in zip(idx, idx[1:] + idx[:1]):
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            lines.append(vertices[[a, b]])
    return lines


def write_obj(path: Path, structure: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Manhattan L-shaped DA3 room structure\n")
        for v in structure["vertices"]:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in structure["faces"]:
            f.write(f"o {face['name']}\n")
            f.write("f " + " ".join(str(int(i) + 1) for i in face["vertices"]) + "\n")


def save_mask_preview(path: Path, grid: np.ndarray, occ: np.ndarray, cand: np.ndarray, best: dict) -> None:
    h, w = occ.shape
    img = Image.new("RGB", (w, h), (0, 0, 0))
    log_grid = np.log1p(grid)
    if np.max(log_grid) > 0:
        gray = np.clip(log_grid / np.max(log_grid) * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(np.dstack([gray, gray, gray]), "RGB")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cand_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    cand_rgba[cand > 0] = np.array([40, 180, 80, 70], dtype=np.uint8)
    occ_edge = cv2.morphologyEx(occ, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    cand_edge = cv2.morphologyEx(cand, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    cand_rgba[occ_edge > 0] = np.array([0, 210, 255, 230], dtype=np.uint8)
    cand_rgba[cand_edge > 0] = np.array([255, 45, 45, 245], dtype=np.uint8)
    overlay = Image.fromarray(cand_rgba, "RGBA")
    out = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    out = out.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    draw = ImageDraw.Draw(out)
    draw.text((12, 12), f"best={best['kind']} score={best['score']:.4f}", fill=(255, 255, 255))
    draw.text(
        (12, 30),
        f"outside={best['outside_observed_frac']:.3f} empty={best['empty_candidate_frac']:.3f} edge={best['edge_cost']:.3f}",
        fill=(255, 255, 255),
    )
    out.save(path, quality=95)


def plot_structure(path: Path, points: np.ndarray, structure: dict, max_points: int) -> None:
    sample = points
    if sample.shape[0] > max_points:
        idx = np.linspace(0, sample.shape[0] - 1, max_points).astype(np.int64)
        sample = sample[idx]
    poly = np.asarray(structure["floor_polygon_xz"], dtype=float)
    closed = np.vstack([poly, poly[:1]])
    fig = plt.figure(figsize=(18, 8), dpi=170)
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    ax1.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.08, c="#666666", alpha=0.13)
    for line in structure_lines(structure):
        ax1.plot(line[:, 0], line[:, 1], line[:, 2], c="#e53935", lw=1.6)
    ax1.set_title("3D DA3 GLB + Manhattan L structure")
    ax1.view_init(elev=18, azim=-62)
    lo = np.percentile(sample, 2, axis=0)
    hi = np.percentile(sample, 98, axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.max(hi - lo) * 0.58), 1e-3)
    ax1.set_xlim(center[0] - radius, center[0] + radius)
    ax1.set_ylim(center[1] - radius, center[1] + radius)
    ax1.set_zlim(center[2] - radius, center[2] + radius)

    ax2.scatter(sample[:, 0], sample[:, 2], s=0.08, c="#666666", alpha=0.18)
    ax2.plot(closed[:, 0], closed[:, 1], "-o", c="#e53935", lw=2.0, ms=3)
    for i, p in enumerate(poly):
        ax2.text(p[0], p[1], str(i), fontsize=8)
    ax2.set_title("Top view x-z")
    ax2.set_aspect("equal", adjustable="box")

    ax3.scatter(sample[:, 0], sample[:, 1], s=0.08, c="#666666", alpha=0.18)
    for line in structure_lines(structure):
        ax3.plot(line[:, 0], line[:, 1], c="#e53935", lw=1.5)
    ax3.set_title("Side x-y")
    ax3.set_aspect("equal", adjustable="box")

    ax4.scatter(sample[:, 2], sample[:, 1], s=0.08, c="#666666", alpha=0.18)
    for line in structure_lines(structure):
        ax4.plot(line[:, 2], line[:, 1], c="#e53935", lw=1.5)
    ax4.set_title("Side z-y")
    ax4.set_aspect("equal", adjustable="box")

    edge_lengths = [float(np.linalg.norm(b - a)) for a, b in zip(poly, np.roll(poly, -1, axis=0))]
    ax5.bar(np.arange(len(edge_lengths)), edge_lengths, color="#c62828")
    ax5.set_title("Floor edge lengths")
    ax5.set_xlabel("edge")
    ax5.set_ylabel("length")

    vertices = np.asarray(structure["vertices"], dtype=float)
    ax6.axis("off")
    rows = [
        f"floor_y={structure['vertical_bounds']['floor_y']:.4f}",
        f"ceiling_y={structure['vertical_bounds']['ceiling_y']:.4f}",
        f"height={structure['vertical_bounds']['ceiling_y'] - structure['vertical_bounds']['floor_y']:.4f}",
        f"vertices={len(vertices)} faces={len(structure['faces'])}",
    ]
    ax6.text(0.02, 0.96, "\n".join(rows), va="top", ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points_world = load_point_cloud(args.scene_glb)
    world_from_room, room_from_world, basis_meta = estimate_room_basis(points_world, args)
    points = world_to_room_points(points_world, room_from_world)
    lo = np.percentile(points[:, [0, 2]], float(args.bounds_percentiles[0]), axis=0).astype(np.float64)
    hi = np.percentile(points[:, [0, 2]], float(args.bounds_percentiles[1]), axis=0).astype(np.float64)
    grid = build_density_grid(points, lo, hi, int(args.grid_size))
    occ = occupancy_from_grid(grid, float(args.occupancy_threshold_quantile), args.close_iterations, args.dilate_iterations)
    occ = largest_component(occ)
    poly_px, best = fit_l_mask(occ, args)
    cand = raster_poly(occ.shape, poly_px)
    poly_xz = px_to_xz(poly_px, lo, hi, int(args.grid_size))
    structure = make_structure(
        poly_xz,
        points,
        (float(args.vertical_percentiles[0]), float(args.vertical_percentiles[1])),
        "da3_glb_leveled_manhattan_l_room_rect_minus_one_corner_v1",
        world_from_room,
        room_from_world,
    )
    structure["source_scene_glb"] = str(args.scene_glb)
    structure["point_count"] = int(points_world.shape[0])
    structure["room_basis_fit"] = basis_meta
    structure["fit_params"] = {
        "grid_size": int(args.grid_size),
        "bounds_percentiles": [float(args.bounds_percentiles[0]), float(args.bounds_percentiles[1])],
        "bounds_lo_xz": lo.tolist(),
        "bounds_hi_xz": hi.tolist(),
        "occupancy_threshold_quantile": float(args.occupancy_threshold_quantile),
        "close_iterations": int(args.close_iterations),
        "dilate_iterations": int(args.dilate_iterations),
    }
    structure["fit_score"] = best
    (args.out_dir / "structure_da3_manhattan_l.json").write_text(
        json.dumps(structure, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_obj(args.out_dir / "room_structure_da3_manhattan_l.obj", structure)
    save_mask_preview(args.out_dir / "structure_da3_manhattan_l_mask_fit.png", grid, occ, cand, best)
    leveled_structure = dict(structure)
    leveled_structure["vertices"] = structure["room_vertices"]
    plot_structure(args.out_dir / "structure_da3_manhattan_l_leveled_comparison.png", points, leveled_structure, args.plot_points)
    plot_structure(args.out_dir / "structure_da3_manhattan_l_world_comparison.png", points_world, structure, args.plot_points)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "structure_json": str(args.out_dir / "structure_da3_manhattan_l.json"),
                "leveled_preview": str(args.out_dir / "structure_da3_manhattan_l_leveled_comparison.png"),
                "world_preview": str(args.out_dir / "structure_da3_manhattan_l_world_comparison.png"),
                "best": best,
                "basis": basis_meta,
                "floor_polygon_xz": structure["floor_polygon_xz"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
