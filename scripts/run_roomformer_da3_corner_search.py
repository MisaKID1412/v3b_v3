#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

from fit_manhattan_l_room_from_da3_glb import (
    build_density_grid,
    estimate_room_basis,
    load_point_cloud,
    occupancy_from_grid,
    room_to_world_points,
    world_to_room_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search for a variable-corner Manhattan room footprint from the latest DA3 point cloud "
            "using RoomFormer candidates plus DA3 top-down geometric scoring."
        )
    )
    parser.add_argument("--roomformer-dir", type=Path, required=True)
    parser.add_argument("--scene-glb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--score-grid-size", type=int, default=760)
    parser.add_argument("--bounds-percentiles", type=float, nargs=2, default=[0.6, 99.4])
    parser.add_argument("--density-modes", nargs="+", default=["log", "sqrt", "binary"])
    parser.add_argument("--corner-thresholds", type=float, nargs="+", default=[0.28, 0.34, 0.42, 0.50])
    parser.add_argument("--min-corners", type=int, default=4)
    parser.add_argument("--max-corners", type=int, default=20)
    parser.add_argument(
        "--target-corners",
        type=int,
        default=0,
        help="Optional fixed corner count for legacy/reproduction runs; 0 keeps the count data-driven.",
    )
    parser.add_argument(
        "--candidate-source-policy",
        choices=("all", "roomformer_only"),
        default="all",
        help=(
            "Select which base polygon families enter DA3 scoring. "
            "'all' preserves the current hybrid search. "
            "'roomformer_only' admits only RoomFormer-derived variable-corner "
            "proposals while retaining DA3 occupancy/radial/vertical evidence "
            "for scoring and topology-preserving refit."
        ),
    )
    parser.add_argument("--plot-points", type=int, default=230000)
    parser.add_argument("--point-sample", type=int, default=160000)
    parser.add_argument("--coarse-point-sample", type=int, default=20000)
    parser.add_argument("--max-base-candidates", type=int, default=240)
    parser.add_argument("--max-expanded-candidates", type=int, default=240)
    parser.add_argument(
        "--corner-penalty",
        type=float,
        default=0.12,
        help="Small model-complexity penalty per corner beyond four to prevent noisy contour overfitting.",
    )
    parser.add_argument(
        "--axis-weight",
        type=float,
        default=1.9,
        help=(
            "Weight of the Manhattan-axis consistency term. Use a stronger value when known camera metadata "
            "provides a trusted room-aligned frame; this does not prescribe a corner count."
        ),
    )
    parser.add_argument("--top-k-preview", type=int, default=12)
    parser.add_argument("--occupancy-threshold-quantile", type=float, default=0.42)
    parser.add_argument("--close-iterations", type=int, default=3)
    parser.add_argument("--dilate-iterations", type=int, default=1)
    parser.add_argument("--notch-samples", type=int, default=110)
    parser.add_argument(
        "--radial-footprint-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate variable-corner free-space candidates from height-consistent radial DA3 evidence. "
            "This preserves narrow wall returns and attached bays that disappear in a filled top-down density map."
        ),
    )
    parser.add_argument("--radial-angle-bins", type=int, default=1440)
    parser.add_argument("--radial-height-bins", type=int, default=8)
    parser.add_argument("--radial-height-range", type=float, nargs=2, default=[0.15, 0.85])
    parser.add_argument("--radial-distance-quantile", type=float, default=0.94)
    parser.add_argument("--radial-min-points-per-bin", type=int, default=3)
    parser.add_argument("--radial-min-height-bins", type=int, default=3)
    parser.add_argument("--radial-smooth-radius", type=int, default=3)
    parser.add_argument("--radial-score-mask-dilate-px", type=int, default=8)
    parser.add_argument("--radial-step-min-depth-frac", type=float, default=0.025)
    parser.add_argument("--radial-step-max-depth-frac", type=float, default=0.20)
    parser.add_argument("--radial-step-min-length-frac", type=float, default=0.06)
    parser.add_argument("--radial-step-max-length-frac", type=float, default=0.88)
    parser.add_argument("--radial-step-profile-smooth-frac", type=float, default=0.012)
    parser.add_argument("--radial-bounds-percentiles", type=float, nargs=2, default=[0.05, 99.95])
    parser.add_argument(
        "--radial-max-bounds-expansion",
        type=float,
        default=0.14,
        help="Maximum radial-evidence bounds expansion on each side, as a fraction of the robust room span.",
    )
    parser.add_argument("--wall-evidence-height-bins", type=int, default=8)
    parser.add_argument("--wall-evidence-min-height-bins", type=int, default=4)
    parser.add_argument("--wall-evidence-dilate-px", type=int, default=2)
    parser.add_argument("--wall-edge-distance-frac", type=float, default=0.018)
    parser.add_argument("--wall-edge-support-target", type=float, default=0.32)
    parser.add_argument("--wall-edge-support-hard-min", type=float, default=0.05)
    parser.add_argument(
        "--wall-edge-supported-perimeter-hard-min",
        type=float,
        default=0.45,
        help=(
            "Minimum fraction of the candidate perimeter whose nearby multi-height wall evidence reaches "
            "--wall-edge-support-hard-min. A perimeter-level gate permits a genuinely enclosing wall to be "
            "partly unobserved behind a large opening or occluder without accepting mostly unsupported shells."
        ),
    )
    parser.add_argument(
        "--wall-edge-mean-support-hard-min",
        type=float,
        default=0.08,
        help="Minimum length-weighted multi-height wall support over the complete candidate perimeter.",
    )
    parser.add_argument("--wall-edge-support-weight", type=float, default=4.0)
    parser.add_argument(
        "--opening-aware-enclosure-override",
        action="store_true",
        help=(
            "Enable the strong-wall enclosure fallback for a run whose depth rays escape through large openings. "
            "Disabled by default so ordinary rooms retain the generalized candidate ranking."
        ),
    )
    parser.add_argument("--wall-enclosure-override-min-support", type=float, default=0.55)
    parser.add_argument("--wall-enclosure-override-mean-support", type=float, default=0.75)
    parser.add_argument(
        "--dominant-wall-enclosure-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When opening-aware selection starts from a four-wall outer envelope, prefer a moderately smaller "
            "four-wall envelope recovered directly from dominant multi-height wall lines if its complete perimeter "
            "is strongly supported. This rejects window-depth leakage without collapsing onto a small interior bay."
        ),
    )
    parser.add_argument("--dominant-wall-enclosure-guard-min-support", type=float, default=0.30)
    parser.add_argument("--dominant-wall-enclosure-guard-mean-support", type=float, default=0.74)
    parser.add_argument("--dominant-wall-enclosure-guard-min-area-ratio", type=float, default=0.45)
    parser.add_argument("--dominant-wall-enclosure-guard-max-area-ratio", type=float, default=0.85)
    parser.add_argument(
        "--high-confidence-learned-topology-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preserve a high-confidence learned non-rectangular topology when it explains the occupied/radial "
            "footprint substantially better than a rectangular envelope. The learned proposal keeps its inferred "
            "corner count; no room shape or corner count is prescribed."
        ),
    )
    parser.add_argument("--learned-topology-min-probability", type=float, default=0.90)
    parser.add_argument("--learned-topology-max-outside-occupancy", type=float, default=0.03)
    parser.add_argument("--learned-topology-max-radial-outside", type=float, default=0.05)
    parser.add_argument("--learned-topology-min-empty-inside-improvement", type=float, default=0.10)
    parser.add_argument("--learned-topology-min-area-ratio", type=float, default=0.60)
    parser.add_argument("--learned-topology-max-area-ratio", type=float, default=0.90)
    parser.add_argument("--learned-topology-min-supported-perimeter", type=float, default=0.80)
    parser.add_argument("--learned-topology-min-mean-wall-support", type=float, default=0.60)
    parser.add_argument(
        "--supported-topology-preservation-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preserve additional data-driven wall turns when every added segment has persistent multi-height "
            "wall support and the richer boundary explains the radial free-space evidence substantially better. "
            "This guard is independent of room shape and does not prescribe a corner count."
        ),
    )
    parser.add_argument("--supported-topology-min-area-ratio", type=float, default=0.75)
    parser.add_argument("--supported-topology-max-area-ratio", type=float, default=0.99)
    parser.add_argument("--supported-topology-min-edge-support", type=float, default=0.55)
    parser.add_argument("--supported-topology-min-edge-support-gain", type=float, default=0.08)
    parser.add_argument("--supported-topology-min-mean-support", type=float, default=0.70)
    parser.add_argument("--supported-topology-min-mean-support-gain", type=float, default=0.04)
    parser.add_argument("--supported-topology-max-radial-edge-p90-ratio", type=float, default=0.60)
    parser.add_argument("--supported-topology-min-radial-empty-improvement", type=float, default=0.015)
    parser.add_argument("--supported-topology-max-outside-occupancy-increase", type=float, default=0.012)
    parser.add_argument("--supported-topology-max-radial-outside-increase", type=float, default=0.012)
    parser.add_argument(
        "--dominant-wall-single-opening-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Permit a camera-containing four-wall dominant enclosure when three sides have strong multi-height "
            "wall support and the fourth retains weaker but nonzero support consistent with one large opening."
        ),
    )
    parser.add_argument("--dominant-wall-single-opening-supported-edge-min", type=float, default=0.65)
    parser.add_argument("--dominant-wall-single-opening-edge-min", type=float, default=0.18)
    parser.add_argument("--dominant-wall-single-opening-mean-support", type=float, default=0.65)
    parser.add_argument("--dominant-wall-single-opening-min-area-ratio", type=float, default=0.35)
    parser.add_argument("--dominant-wall-single-opening-max-area-ratio", type=float, default=0.70)
    parser.add_argument(
        "--dominant-base-from-wall-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the strongest enclosing vertical-wall lines around the camera as the base rectangle before "
            "adding data-driven bays or recesses. This prevents small distant point sets from defining all room bounds."
        ),
    )
    parser.add_argument("--disable-yaw-align", action="store_true")
    parser.add_argument(
        "--forced-up-axis-world",
        type=float,
        nargs=3,
        default=None,
        help="Optional known gravity/up vector in the DA3 world frame.",
    )
    return parser.parse_args()


def polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    return float(0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - poly[:, 1] * np.roll(poly[:, 0], -1)))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if polygon_area(poly) >= 0.0 else poly[::-1].copy()


def start_top_left(poly: np.ndarray) -> np.ndarray:
    order = np.lexsort((poly[:, 0], -poly[:, 1]))
    return np.roll(poly, -int(order[0]), axis=0)


def normalize_polygon_vertices(poly: np.ndarray, relative_tolerance: float = 1e-7) -> np.ndarray:
    """Remove zero-length edges and redundant collinear vertices.

    Data-driven radial candidates can contain repeated samples at the end of a
    step or an extra sample on an otherwise straight wall.  Those samples do
    not define additional room corners and would otherwise export degenerate
    wall faces.  The tolerance is scaled by the candidate span and does not
    impose a topology or preferred corner count.
    """
    points = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] < 3:
        return points.copy()
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    distance_tolerance = float(relative_tolerance) * span

    deduplicated = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - deduplicated[-1])) > distance_tolerance:
            deduplicated.append(point)
    if len(deduplicated) > 1 and float(np.linalg.norm(deduplicated[-1] - deduplicated[0])) <= distance_tolerance:
        deduplicated.pop()

    cleaned = np.asarray(deduplicated, dtype=np.float64)
    changed = True
    while changed and cleaned.shape[0] >= 3:
        changed = False
        keep = np.ones(cleaned.shape[0], dtype=bool)
        for index in range(cleaned.shape[0]):
            previous = cleaned[(index - 1) % cleaned.shape[0]]
            current = cleaned[index]
            following = cleaned[(index + 1) % cleaned.shape[0]]
            incoming = current - previous
            outgoing = following - current
            incoming_length = float(np.linalg.norm(incoming))
            outgoing_length = float(np.linalg.norm(outgoing))
            if incoming_length <= distance_tolerance or outgoing_length <= distance_tolerance:
                keep[index] = False
                changed = True
                continue
            normalized_cross = abs(float(np.cross(incoming, outgoing))) / max(
                incoming_length * outgoing_length, 1e-12
            )
            if normalized_cross <= relative_tolerance and float(np.dot(incoming, outgoing)) > 0.0:
                keep[index] = False
                changed = True
        if changed:
            cleaned = cleaned[keep]
    return cleaned


def rotate_xz(xz: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    out = np.empty_like(xz, dtype=np.float64)
    out[:, 0] = c * xz[:, 0] - s * xz[:, 1]
    out[:, 1] = s * xz[:, 0] + c * xz[:, 1]
    return out


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return (labels == label).astype(np.uint8) * 255


def estimate_manhattan_yaw(points_room: np.ndarray, args: argparse.Namespace) -> tuple[float, dict]:
    lo = np.percentile(points_room[:, [0, 2]], args.bounds_percentiles[0], axis=0).astype(np.float64)
    hi = np.percentile(points_room[:, [0, 2]], args.bounds_percentiles[1], axis=0).astype(np.float64)
    grid = build_density_grid(points_room, lo, hi, int(args.score_grid_size))
    occ = occupancy_from_grid(
        grid,
        float(args.occupancy_threshold_quantile),
        int(args.close_iterations),
        int(args.dilate_iterations),
    )
    occ = largest_component(occ)
    edges = cv2.Canny(occ, 40, 140)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720.0,
        threshold=max(24, int(args.score_grid_size * 0.045)),
        minLineLength=max(26, int(args.score_grid_size * 0.075)),
        maxLineGap=max(6, int(args.score_grid_size * 0.018)),
    )
    angles = []
    weights = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            angles.append(math.atan2(dy, dx))
            weights.append(length)

    if not angles:
        contours, _ = cv2.findContours(occ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
            for a, b in zip(contour, np.roll(contour, -1, axis=0)):
                edge = b - a
                length = float(np.linalg.norm(edge))
                if length >= max(8.0, args.score_grid_size * 0.015):
                    angles.append(math.atan2(float(edge[1]), float(edge[0])))
                    weights.append(length)

    if not angles:
        return 0.0, {"method": "none", "line_count": 0, "yaw_degrees": 0.0}

    a = np.asarray(angles, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    z = np.sum(w * np.exp(1j * 4.0 * a))
    yaw = 0.25 * math.atan2(float(z.imag), float(z.real))
    while yaw <= -math.pi / 4.0:
        yaw += math.pi / 2.0
    while yaw > math.pi / 4.0:
        yaw -= math.pi / 2.0

    strength = float(abs(z) / max(float(np.sum(w)), 1e-8))
    return yaw, {
        "method": "hough_or_contour_four_angle_mean",
        "line_count": int(len(angles)),
        "yaw_degrees": float(math.degrees(yaw)),
        "strength": strength,
    }


def compose_yaw_basis(world_from_room: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    base_from_aligned = np.eye(4, dtype=np.float64)
    base_from_aligned[:3, :3] = np.array(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=np.float64,
    )
    world_from_aligned = world_from_room @ base_from_aligned
    aligned_from_world = np.linalg.inv(world_from_aligned)
    return world_from_aligned, aligned_from_world


def basis_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        bounds_percentiles=args.bounds_percentiles,
        level_grid=84,
        floor_cell_percentile=6.0,
        ceiling_cell_percentile=94.0,
        min_cell_points=18,
        plane_trim_percentile=68.0,
        plane_fit_iters=7,
        average_ceiling_normal=False,
        forced_up_axis_world=args.forced_up_axis_world,
    )


def normalize_density(grid: np.ndarray, occ: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log":
        img = np.log1p(grid)
    elif mode == "sqrt":
        img = np.sqrt(np.maximum(grid, 0.0))
    elif mode == "binary":
        img = (occ > 0).astype(np.float32)
        img = cv2.GaussianBlur(img, (3, 3), 0.0)
    else:
        raise ValueError(f"Unknown density mode: {mode}")
    max_v = float(np.max(img))
    if max_v > 0:
        img = img / max_v
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def roomformer_args(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        backbone="resnet50",
        dilation=False,
        position_embedding="sine",
        position_embedding_scale=2 * np.pi,
        num_feature_levels=4,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_queries=800,
        num_polys=20,
        dec_n_points=4,
        enc_n_points=4,
        query_pos_type="sine",
        with_poly_refine=True,
        masked_attn=False,
        semantic_classes=-1,
        aux_loss=True,
        lr_backbone=0.0,
    )


def build_roomformer(roomformer_dir: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(roomformer_dir))
    sys.path.insert(0, str(roomformer_dir / "diff_ras"))
    sys.path.insert(0, str(roomformer_dir / "models" / "ops"))
    from models import build_model

    model = build_model(roomformer_args(str(device)), train=False)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [x for x in unexpected if not (x.endswith("total_params") or x.endswith("total_ops"))]
    model.to(device)
    model.eval()
    return model, {"missing_keys": len(missing), "unexpected_keys": len(unexpected)}


def sanitize_poly(poly: np.ndarray, min_dist: float = 1e-4) -> np.ndarray | None:
    poly = np.asarray(poly, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 4:
        return None
    poly = np.clip(poly, 0.0, 1.0)
    keep = [0]
    for i in range(1, poly.shape[0]):
        if float(np.linalg.norm(poly[i] - poly[keep[-1]])) >= min_dist:
            keep.append(i)
    poly = poly[np.asarray(keep, dtype=np.int64)]
    if poly.shape[0] > 2 and float(np.linalg.norm(poly[0] - poly[-1])) < min_dist:
        poly = poly[:-1]
    if poly.shape[0] < 4 or abs(polygon_area(poly)) < 2e-5:
        return None
    return ensure_ccw(poly)


def approx_to_target(poly: np.ndarray, target: int) -> np.ndarray | None:
    if poly.shape[0] == target:
        return poly.copy()
    pts = (poly * 255.0).astype(np.float32).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(pts, True))
    if perimeter <= 0:
        return None
    best = None
    for epsf in np.linspace(0.0006, 0.08, 260):
        approx = cv2.approxPolyDP(pts, float(epsf * perimeter), True)[:, 0, :] / 255.0
        if approx.shape[0] < 4:
            continue
        key = (abs(int(approx.shape[0]) - target), abs(float(epsf) - 0.012))
        if best is None or key < best[0]:
            best = (key, approx)
        if approx.shape[0] == target:
            return ensure_ccw(approx.astype(np.float64))
    if best is None:
        return None
    approx = best[1].astype(np.float64)
    if approx.shape[0] < target:
        return None
    return greedy_reduce_to_target(approx, target)


def greedy_reduce_to_target(poly: np.ndarray, target: int) -> np.ndarray | None:
    out = ensure_ccw(poly.copy())
    while out.shape[0] > target:
        costs = []
        for i in range(out.shape[0]):
            prev_p = out[(i - 1) % out.shape[0]]
            curr_p = out[i]
            next_p = out[(i + 1) % out.shape[0]]
            costs.append(abs(np.cross(curr_p - prev_p, next_p - curr_p)))
        remove_i = int(np.argmin(np.asarray(costs)))
        out = np.delete(out, remove_i, axis=0)
        if out.shape[0] < 4:
            return None
    if out.shape[0] != target or abs(polygon_area(out)) < 2e-5:
        return None
    return ensure_ccw(out)


def normalized_to_xz(poly: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x = lo[0] + poly[:, 0] * (hi[0] - lo[0])
    z = lo[1] + poly[:, 1] * (hi[1] - lo[1])
    return ensure_ccw(np.column_stack([x, z]).astype(np.float64))


def grid_poly_to_xz(poly_px: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: int) -> np.ndarray:
    x = lo[0] + (poly_px[:, 0].astype(np.float64) + 0.5) / size * (hi[0] - lo[0])
    z = lo[1] + (poly_px[:, 1].astype(np.float64) + 0.5) / size * (hi[1] - lo[1])
    return ensure_ccw(np.column_stack([x, z]).astype(np.float64))


def d4_matrices() -> list[tuple[str, np.ndarray]]:
    mats = []
    base = [
        ("identity", np.array([[1, 0], [0, 1]], dtype=float)),
        ("rot90", np.array([[0, -1], [1, 0]], dtype=float)),
        ("rot180", np.array([[-1, 0], [0, -1]], dtype=float)),
        ("rot270", np.array([[0, 1], [-1, 0]], dtype=float)),
    ]
    mirror = np.array([[-1, 0], [0, 1]], dtype=float)
    for name, mat in base:
        mats.append((name, mat))
        mats.append((f"mirror_x_{name}", mat @ mirror))
    return mats


def transform_in_bounds(poly: np.ndarray, mat: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    center = 0.5 * (lo + hi)
    half = np.maximum(0.5 * (hi - lo), 1e-6)
    norm = (poly - center[None, :]) / half[None, :]
    out = norm @ mat.T
    return ensure_ccw(out * half[None, :] + center[None, :])


def corner_count_allowed(count: int, min_corners: int, max_corners: int, target_corners: int = 0) -> bool:
    if target_corners > 0:
        return count == target_corners
    return min_corners <= count <= max_corners


def approx_in_corner_range(
    poly: np.ndarray,
    min_corners: int,
    max_corners: int,
    target_corners: int = 0,
) -> list[tuple[str, np.ndarray]]:
    if target_corners > 0:
        approx = approx_to_target(poly, target_corners)
        return [] if approx is None else [(f"approx{target_corners}", approx)]

    variants: list[tuple[str, np.ndarray]] = []
    seen_counts: set[int] = set()

    def add(kind: str, candidate: np.ndarray) -> None:
        candidate = sanitize_poly(candidate)
        if candidate is None or not corner_count_allowed(candidate.shape[0], min_corners, max_corners):
            return
        count = int(candidate.shape[0])
        if count in seen_counts:
            return
        seen_counts.add(count)
        variants.append((kind, candidate))

    add(f"raw{poly.shape[0]}", poly)
    pts = (poly * 255.0).astype(np.float32).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(pts, True))
    if perimeter > 0:
        for epsf in np.linspace(0.001, 0.055, 64):
            approx = cv2.approxPolyDP(pts, float(epsf * perimeter), True)[:, 0, :] / 255.0
            add(f"approx{approx.shape[0]}_eps{epsf:.4f}", approx.astype(np.float64))
    return variants


def snap_manhattan_polygon(poly: np.ndarray) -> np.ndarray | None:
    """Project an arbitrary even-sided footprint to alternating horizontal/vertical edge lines."""
    poly = start_top_left(ensure_ccw(np.asarray(poly, dtype=np.float64)))
    n = int(poly.shape[0])
    if n < 4 or n % 2:
        return None
    edges = np.roll(poly, -1, axis=0) - poly
    orientations = np.where(np.abs(edges[:, 0]) >= np.abs(edges[:, 1]), 0, 1)  # 0: horizontal, 1: vertical
    if np.any(orientations == np.roll(orientations, 1)):
        return None
    line_values = np.where(
        orientations == 0,
        0.5 * (poly[:, 1] + np.roll(poly[:, 1], -1)),
        0.5 * (poly[:, 0] + np.roll(poly[:, 0], -1)),
    )
    out = np.empty_like(poly)
    for i in range(n):
        previous = (i - 1) % n
        horizontal = previous if orientations[previous] == 0 else i
        vertical = previous if orientations[previous] == 1 else i
        out[i] = [line_values[vertical], line_values[horizontal]]
    out = start_top_left(ensure_ccw(out))
    geom = Polygon(out)
    if not geom.is_valid or geom.area <= 1e-6 or np.min(np.linalg.norm(np.roll(out, -1, axis=0) - out, axis=1)) <= 1e-5:
        return None
    return out


def contour_polygon_candidates(occ: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: int, args: argparse.Namespace) -> list[dict]:
    occ = largest_component(occ)
    contours, _ = cv2.findContours((occ > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    margin = max(1, size // 300)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(size - 1, x + w - 1 + margin), min(size - 1, y + h - 1 + margin)
    bbox = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
    out: list[dict] = [{"source": "occupancy_axis_aligned_envelope_4", "poly_xz": grid_poly_to_xz(bbox, lo, hi, size)}]
    seen_counts: set[int] = set()
    for epsf in np.linspace(0.001, 0.06, 96):
        approx = cv2.approxPolyDP(contour, float(epsf * perimeter), True)[:, 0, :]
        count = int(approx.shape[0])
        if not corner_count_allowed(count, args.min_corners, args.max_corners, args.target_corners):
            continue
        if count in seen_counts:
            continue
        seen_counts.add(count)
        out.append({"source": f"occupancy_contour_{count}_eps{epsf:.4f}", "poly_xz": grid_poly_to_xz(approx, lo, hi, size)})
    return out


def circular_nanmedian(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    if values.size == 0:
        return out
    offsets = np.arange(-max(0, int(radius)), max(0, int(radius)) + 1, dtype=np.int64)
    for i in range(values.size):
        ids = (i + offsets) % values.size
        finite = values[ids][np.isfinite(values[ids])]
        if finite.size:
            out[i] = float(np.median(finite))
    return out


def radial_evidence_bounds(points_room: np.ndarray, base_lo: np.ndarray, base_hi: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    """Expand robust x-z bounds only where middle-height points provide persistent room evidence."""
    floor_y, ceiling_y = np.percentile(points_room[:, 1], [2.0, 98.0])
    height = max(float(ceiling_y - floor_y), 1e-8)
    yn = (points_room[:, 1] - float(floor_y)) / height
    lower, upper = [float(x) for x in args.radial_height_range]
    slab = points_room[(yn >= lower) & (yn <= upper)]
    if slab.shape[0] < 100:
        return base_lo, base_hi, {"enabled": False, "reason": "insufficient_middle_height_points"}
    qlo, qhi = [float(x) for x in args.radial_bounds_percentiles]
    slab_lo = np.percentile(slab[:, [0, 2]], qlo, axis=0).astype(np.float64)
    slab_hi = np.percentile(slab[:, [0, 2]], qhi, axis=0).astype(np.float64)
    base_span = np.maximum(base_hi - base_lo, 1e-8)
    limit = max(0.0, float(args.radial_max_bounds_expansion)) * base_span
    clipped_lo = np.maximum(slab_lo, base_lo - limit)
    clipped_hi = np.minimum(slab_hi, base_hi + limit)
    expanded_lo = np.minimum(base_lo, clipped_lo)
    expanded_hi = np.maximum(base_hi, clipped_hi)
    return expanded_lo, expanded_hi, {
        "enabled": True,
        "floor_y": float(floor_y),
        "ceiling_y": float(ceiling_y),
        "height_range": [lower, upper],
        "slab_point_count": int(slab.shape[0]),
        "slab_percentiles": [qlo, qhi],
        "slab_lo_xz": slab_lo.tolist(),
        "slab_hi_xz": slab_hi.tolist(),
        "base_lo_xz": base_lo.tolist(),
        "base_hi_xz": base_hi.tolist(),
        "expanded_lo_xz": expanded_lo.tolist(),
        "expanded_hi_xz": expanded_hi.tolist(),
    }


def build_radial_free_space_mask(
    points_room: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    size: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict]:
    """Build a star-shaped visible-room mask using wall depths that repeat over several height bands."""
    if not bool(args.radial_footprint_candidates):
        return None, {"enabled": False, "reason": "disabled"}
    floor_y, ceiling_y = np.percentile(points_room[:, 1], [2.0, 98.0])
    height = max(float(ceiling_y - floor_y), 1e-8)
    lower, upper = [float(x) for x in args.radial_height_range]
    yn = (points_room[:, 1] - float(floor_y)) / height
    keep = (yn >= lower) & (yn <= upper)
    slab = points_room[keep]
    slab_yn = yn[keep]
    if slab.shape[0] < 100:
        return None, {"enabled": False, "reason": "insufficient_middle_height_points"}

    xz = slab[:, [0, 2]]
    radii = np.linalg.norm(xz, axis=1)
    angles = np.mod(np.arctan2(xz[:, 1], xz[:, 0]), 2.0 * np.pi)
    angle_bins = max(180, int(args.radial_angle_bins))
    height_bins = max(3, int(args.radial_height_bins))
    angle_ids = np.floor(angles / (2.0 * np.pi) * angle_bins).astype(np.int32)
    angle_ids = np.clip(angle_ids, 0, angle_bins - 1)
    height_ids = np.floor((slab_yn - lower) / max(upper - lower, 1e-8) * height_bins).astype(np.int32)
    height_ids = np.clip(height_ids, 0, height_bins - 1)
    quantile = float(np.clip(args.radial_distance_quantile, 0.50, 0.995))
    min_points = max(1, int(args.radial_min_points_per_bin))
    per_band = np.full((height_bins, angle_bins), np.nan, dtype=np.float64)
    for band in range(height_bins):
        band_keep = height_ids == band
        band_angles = angle_ids[band_keep]
        band_radii = radii[band_keep]
        for angle_i in np.unique(band_angles):
            vals = band_radii[band_angles == angle_i]
            if vals.size >= min_points:
                per_band[band, int(angle_i)] = float(np.quantile(vals, quantile))

    radial = np.full(angle_bins, np.nan, dtype=np.float64)
    band_support = np.sum(np.isfinite(per_band), axis=0).astype(np.int32)
    required_bands = max(1, int(args.radial_min_height_bins))
    for angle_i in range(angle_bins):
        vals = per_band[:, angle_i]
        vals = vals[np.isfinite(vals)]
        if vals.size >= required_bands:
            radial[angle_i] = float(np.median(vals))
    radial = circular_nanmedian(radial, max(1, int(args.radial_smooth_radius) // 2))
    finite = np.isfinite(radial)
    if int(np.count_nonzero(finite)) < max(24, angle_bins // 8):
        return None, {
            "enabled": False,
            "reason": "insufficient_height_consistent_angles",
            "finite_angles": int(np.count_nonzero(finite)),
        }
    ids = np.arange(angle_bins, dtype=np.float64)
    radial[~finite] = np.interp(ids[~finite], ids[finite], radial[finite], period=angle_bins)
    radial = circular_nanmedian(radial, max(0, int(args.radial_smooth_radius)))

    theta = (np.arange(angle_bins, dtype=np.float64) + 0.5) / angle_bins * (2.0 * np.pi)
    boundary = np.column_stack([radial * np.cos(theta), radial * np.sin(theta)])
    px = np.empty_like(boundary)
    px[:, 0] = (boundary[:, 0] - lo[0]) / max(float(hi[0] - lo[0]), 1e-8) * (size - 1)
    px[:, 1] = (boundary[:, 1] - lo[1]) / max(float(hi[1] - lo[1]), 1e-8) * (size - 1)
    px = np.clip(px, 0.0, float(size - 1))
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return mask, {
        "enabled": True,
        "method": "height_consistent_radial_free_space_v1",
        "floor_y": float(floor_y),
        "ceiling_y": float(ceiling_y),
        "height_range": [lower, upper],
        "slab_point_count": int(slab.shape[0]),
        "angle_bins": angle_bins,
        "height_bins": height_bins,
        "distance_quantile": quantile,
        "minimum_height_bins": required_bands,
        "finite_angle_count_before_interpolation": int(np.count_nonzero(finite)),
        "height_band_support_percentiles": np.percentile(band_support, [0, 10, 50, 90, 100]).tolist(),
        "radial_distance_percentiles": np.percentile(radial, [0, 10, 50, 90, 100]).tolist(),
    }


def radial_polygon_candidates(mask: np.ndarray | None, lo: np.ndarray, hi: np.ndarray, args: argparse.Namespace) -> list[dict]:
    if mask is None:
        return []
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(contour, True))
    out: list[dict] = []
    seen_counts: set[int] = set()
    for epsf in np.linspace(0.001, 0.045, 96):
        approx = cv2.approxPolyDP(contour, float(epsf * perimeter), True)[:, 0, :]
        count = int(approx.shape[0])
        if count in seen_counts or not corner_count_allowed(count, args.min_corners, args.max_corners, args.target_corners):
            continue
        seen_counts.add(count)
        out.append(
            {
                "source": f"occupancy_radial_free_space_{count}_eps{epsf:.4f}",
                "poly_xz": grid_poly_to_xz(approx, lo, hi, int(mask.shape[0])),
            }
        )
    return out


def _smooth_profile(profile: np.ndarray, kernel: int) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(profile)
    if not np.any(finite):
        return profile
    ids = np.arange(profile.size, dtype=np.float64)
    filled = profile.copy()
    filled[~finite] = np.interp(ids[~finite], ids[finite], profile[finite])
    kernel = max(3, int(kernel) | 1)
    kernel = min(kernel, (profile.size - 1) | 1)
    radius = kernel // 2
    padded = np.pad(filled, (radius, radius), mode="edge")
    return np.asarray([np.median(padded[i : i + kernel]) for i in range(profile.size)], dtype=np.float64)


def _true_runs(active: np.ndarray) -> list[tuple[int, int]]:
    active = np.asarray(active, dtype=bool)
    padded = np.concatenate([[False], active, [False]]).astype(np.int8)
    transitions = np.diff(padded)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0] - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def radial_step_candidates(
    mask: np.ndarray | None,
    base_lo: np.ndarray,
    base_hi: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    args: argparse.Namespace,
) -> list[dict]:
    """Fit generic Manhattan side steps from radial boundary profiles.

    An internal side interval produces an eight-edge footprint: the original
    side is split and two narrow return faces plus one offset face are added.
    If the interval reaches a room corner, the equivalent six-edge candidate
    is produced.  No side, position, or corner count is fixed in advance.
    """
    if mask is None:
        return []
    observed = mask > 0
    size = int(mask.shape[0])
    room_span = np.maximum(base_hi - base_lo, 1e-8)
    smooth_kernel = max(3, int(round(float(args.radial_step_profile_smooth_frac) * size)) | 1)
    close_kernel = max(3, int(round(0.012 * size)) | 1)
    min_depth_frac = max(0.0, float(args.radial_step_min_depth_frac))
    max_depth_frac = max(min_depth_frac, float(args.radial_step_max_depth_frac))
    min_length_frac = max(0.0, float(args.radial_step_min_length_frac))
    max_length_frac = min(1.0, max(min_length_frac, float(args.radial_step_max_length_frac)))

    profiles: list[tuple[str, np.ndarray, float, float]] = []
    row_profiles = []
    for row in range(size):
        cols = np.where(observed[row])[0]
        row_profiles.append((float(np.min(cols)), float(np.max(cols))) if cols.size else (np.nan, np.nan))
    col_profiles = []
    for col in range(size):
        rows = np.where(observed[:, col])[0]
        col_profiles.append((float(np.min(rows)), float(np.max(rows))) if rows.size else (np.nan, np.nan))
    row_profiles = np.asarray(row_profiles, dtype=np.float64)
    col_profiles = np.asarray(col_profiles, dtype=np.float64)

    def x_to_px(x: float) -> float:
        return (x - lo[0]) / max(float(hi[0] - lo[0]), 1e-8) * (size - 1)

    def z_to_px(z: float) -> float:
        return (z - lo[1]) / max(float(hi[1] - lo[1]), 1e-8) * (size - 1)

    profiles.extend(
        [
            ("right", row_profiles[:, 1], x_to_px(float(base_hi[0])), float(room_span[0])),
            ("left", row_profiles[:, 0], x_to_px(float(base_lo[0])), float(room_span[0])),
            ("top", col_profiles[:, 1], z_to_px(float(base_hi[1])), float(room_span[1])),
            ("bottom", col_profiles[:, 0], z_to_px(float(base_lo[1])), float(room_span[1])),
        ]
    )

    out: list[dict] = []
    seen: set[tuple[float, ...]] = set()
    for side, raw_profile, baseline_px, physical_span in profiles:
        profile = _smooth_profile(raw_profile, smooth_kernel)
        px_per_unit = (size - 1) / max(physical_span, 1e-8)
        min_depth_px = min_depth_frac * physical_span * px_per_unit
        max_depth_px = max_depth_frac * physical_span * px_per_unit
        for direction in ("outward", "inward"):
            if side in ("right", "top"):
                outward_delta = profile - baseline_px
            else:
                outward_delta = baseline_px - profile
            delta = outward_delta if direction == "outward" else -outward_delta
            active = np.isfinite(delta) & (delta >= min_depth_px) & (delta <= max_depth_px)
            active_u8 = active.astype(np.uint8).reshape(1, -1) * 255
            active_u8 = cv2.morphologyEx(
                active_u8, cv2.MORPH_CLOSE, np.ones((1, close_kernel), np.uint8), iterations=1
            )
            active_u8 = cv2.morphologyEx(
                active_u8, cv2.MORPH_OPEN, np.ones((1, max(3, close_kernel // 2 | 1)), np.uint8), iterations=1
            )
            candidates = []
            for start, end in _true_runs(active_u8.reshape(-1) > 0):
                length_frac = float(end - start + 1) / size
                if length_frac < min_length_frac or length_frac > max_length_frac:
                    continue
                step_px = float(np.median(profile[start : end + 1]))
                depth_px = float(np.median(delta[start : end + 1]))
                candidates.append((depth_px * length_frac, start, end, step_px, length_frac))
            for _, start, end, step_px, length_frac in sorted(candidates, reverse=True)[:2]:
                touch = max(2, int(round(0.025 * size)))
                touches_low = start <= touch
                touches_high = end >= size - 1 - touch
                if touches_low and touches_high:
                    continue
                x0, z0 = [float(x) for x in base_lo]
                x1, z1 = [float(x) for x in base_hi]
                if side in ("right", "left"):
                    seg0 = lo[1] + (start + 0.5) / size * (hi[1] - lo[1])
                    seg1 = lo[1] + (end + 0.5) / size * (hi[1] - lo[1])
                    seg0 = float(np.clip(seg0, z0, z1))
                    seg1 = float(np.clip(seg1, z0, z1))
                    step = lo[0] + (step_px + 0.5) / size * (hi[0] - lo[0])
                    step = float(step)
                    if side == "right":
                        if touches_low:
                            poly = [[x0, z1], [x0, z0], [step, z0], [step, seg1], [x1, seg1], [x1, z1]]
                        elif touches_high:
                            poly = [[x0, z1], [x0, z0], [x1, z0], [x1, seg0], [step, seg0], [step, z1]]
                        else:
                            poly = [[x0, z1], [x0, z0], [x1, z0], [x1, seg0], [step, seg0], [step, seg1], [x1, seg1], [x1, z1]]
                    else:
                        if touches_low:
                            poly = [[x0, z1], [x0, seg1], [step, seg1], [step, z0], [x1, z0], [x1, z1]]
                        elif touches_high:
                            poly = [[step, z1], [step, seg0], [x0, seg0], [x0, z0], [x1, z0], [x1, z1]]
                        else:
                            poly = [[x0, z1], [x0, seg1], [step, seg1], [step, seg0], [x0, seg0], [x0, z0], [x1, z0], [x1, z1]]
                else:
                    seg0 = lo[0] + (start + 0.5) / size * (hi[0] - lo[0])
                    seg1 = lo[0] + (end + 0.5) / size * (hi[0] - lo[0])
                    seg0 = float(np.clip(seg0, x0, x1))
                    seg1 = float(np.clip(seg1, x0, x1))
                    step = lo[1] + (step_px + 0.5) / size * (hi[1] - lo[1])
                    step = float(step)
                    if side == "top":
                        if touches_low:
                            poly = [[x0, step], [seg1, step], [seg1, z1], [x1, z1], [x1, z0], [x0, z0]]
                        elif touches_high:
                            poly = [[x0, z1], [x0, z0], [x1, z0], [x1, step], [seg0, step], [seg0, z1]]
                        else:
                            poly = [[x0, z1], [x0, z0], [x1, z0], [x1, z1], [seg1, z1], [seg1, step], [seg0, step], [seg0, z1]]
                    else:
                        if touches_low:
                            poly = [[x0, z1], [x0, step], [seg1, step], [seg1, z0], [x1, z0], [x1, z1]]
                        elif touches_high:
                            poly = [[x0, z1], [x0, z0], [seg0, z0], [seg0, step], [x1, step], [x1, z1]]
                        else:
                            poly = [[x0, z1], [x0, z0], [seg0, z0], [seg0, step], [seg1, step], [seg1, z0], [x1, z0], [x1, z1]]
                poly_arr = start_top_left(ensure_ccw(np.asarray(poly, dtype=np.float64)))
                if not corner_count_allowed(poly_arr.shape[0], args.min_corners, args.max_corners, args.target_corners):
                    continue
                geom = Polygon(poly_arr)
                if not geom.is_valid or geom.area <= 1e-6:
                    continue
                key = tuple(np.round(poly_arr.reshape(-1), 4).tolist())
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "source": f"occupancy_radial_step_{side}_{direction}_{poly_arr.shape[0]}",
                        "poly_xz": poly_arr,
                        "radial_step": {
                            "side": side,
                            "direction": direction,
                            "profile_start": int(start),
                            "profile_end": int(end),
                            "length_fraction": length_frac,
                            "step_coordinate": step,
                        },
                    }
                )
    return out


def filter_points_to_radial_mask(point_xz: np.ndarray, support: dict, dilate_px: int) -> np.ndarray:
    mask = support.get("radial_mask")
    if mask is None or point_xz.size == 0:
        return point_xz
    dilate_px = max(0, int(dilate_px))
    if dilate_px:
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    lo = support["lo_xz"]
    hi = support["hi_xz"]
    size = int(mask.shape[0])
    uv = (point_xz - lo[None, :]) / np.maximum(hi - lo, 1e-8)[None, :]
    ij = np.floor(uv * (size - 1)).astype(np.int32)
    valid = np.all((ij >= 0) & (ij < size), axis=1)
    keep = np.zeros(point_xz.shape[0], dtype=bool)
    keep[valid] = mask[ij[valid, 1], ij[valid, 0]] > 0
    return point_xz[keep]


def build_vertical_wall_evidence(
    points_room: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    size: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep x-z cells that recur through several independent room-height bands.

    Floor and ceiling samples occupy only one or two bands, while a structural wall
    normally returns at the same x-z location over much of the room height.  The
    result is intentionally independent of any dataset semantic labels.
    """
    floor_y, ceiling_y = np.percentile(points_room[:, 1], [2.0, 98.0])
    height = max(float(ceiling_y - floor_y), 1e-8)
    lower, upper = [float(x) for x in args.radial_height_range]
    yn = (points_room[:, 1] - float(floor_y)) / height
    keep = (yn >= lower) & (yn <= upper)
    slab = points_room[keep]
    slab_yn = yn[keep]
    height_bins = max(3, int(args.wall_evidence_height_bins))
    required = int(np.clip(int(args.wall_evidence_min_height_bins), 1, height_bins))
    band_ids = np.floor((slab_yn - lower) / max(upper - lower, 1e-8) * height_bins).astype(np.int32)
    band_ids = np.clip(band_ids, 0, height_bins - 1)
    support_count = np.zeros((size, size), dtype=np.uint8)
    dilate_px = max(0, int(args.wall_evidence_dilate_px))
    for band in range(height_bins):
        band_xz = slab[band_ids == band][:, [0, 2]]
        if band_xz.size == 0:
            continue
        uv = (band_xz - lo[None, :]) / np.maximum(hi - lo, 1e-8)[None, :]
        ij = np.floor(uv * (size - 1)).astype(np.int32)
        valid = np.all((ij >= 0) & (ij < size), axis=1)
        ij = ij[valid]
        band_mask = np.zeros((size, size), dtype=np.uint8)
        if ij.size:
            band_mask[ij[:, 1], ij[:, 0]] = 255
        if dilate_px:
            band_mask = cv2.dilate(
                band_mask,
                np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8),
                iterations=1,
            )
        support_count += (band_mask > 0).astype(np.uint8)
    evidence = (support_count >= required).astype(np.uint8) * 255
    evidence = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    yy, xx = np.where(evidence > 0)
    evidence_xz = np.column_stack(
        [
            lo[0] + (xx.astype(np.float64) + 0.5) / size * (hi[0] - lo[0]),
            lo[1] + (yy.astype(np.float64) + 0.5) / size * (hi[1] - lo[1]),
        ]
    )
    return evidence, evidence_xz, {
        "method": "multi_height_vertical_wall_evidence_v1",
        "height_range": [lower, upper],
        "height_bins": height_bins,
        "minimum_height_bins": required,
        "dilate_px": dilate_px,
        "slab_points": int(slab.shape[0]),
        "evidence_cells": int(evidence_xz.shape[0]),
        "support_percentiles": np.percentile(support_count, [0, 50, 90, 99, 100]).tolist(),
    }


def dominant_enclosing_axis_bounds(
    evidence: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    camera_xz: np.ndarray,
    fallback_lo: np.ndarray,
    fallback_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Find the strongest enclosing Manhattan wall line on each side of the camera."""
    size = int(evidence.shape[0])
    x_profile = np.sum(evidence > 0, axis=0).astype(np.float64)
    z_profile = np.sum(evidence > 0, axis=1).astype(np.float64)
    x_profile = cv2.GaussianBlur(x_profile.reshape(1, -1), (0, 0), sigmaX=2.0).reshape(-1)
    z_profile = cv2.GaussianBlur(z_profile.reshape(1, -1), (0, 0), sigmaX=2.0).reshape(-1)
    x_coords = lo[0] + (np.arange(size, dtype=np.float64) + 0.5) / size * (hi[0] - lo[0])
    z_coords = lo[1] + (np.arange(size, dtype=np.float64) + 0.5) / size * (hi[1] - lo[1])

    def choose(profile: np.ndarray, coords: np.ndarray, side: str, center: float, fallback: float) -> tuple[float, dict]:
        span = max(float(coords[-1] - coords[0]), 1e-8)
        if side == "low":
            allowed = np.where(coords < center - 0.015 * span)[0]
        else:
            allowed = np.where(coords > center + 0.015 * span)[0]
        if allowed.size == 0:
            return float(fallback), {"fallback": True, "reason": "no_cells_on_side"}
        side_max = float(np.max(profile[allowed]))
        threshold = max(3.0, 0.16 * side_max)
        ordered = allowed[np.argsort(profile[allowed])[::-1]]
        selected = None
        for index in ordered:
            if float(profile[index]) < threshold:
                break
            left = max(0, int(index) - 2)
            right = min(profile.size, int(index) + 3)
            if float(profile[index]) >= float(np.max(profile[left:right])) - 1e-8:
                selected = int(index)
                break
        if selected is None:
            return float(fallback), {"fallback": True, "reason": "no_strong_peak", "side_max": side_max}
        return float(coords[selected]), {
            "fallback": False,
            "coordinate": float(coords[selected]),
            "profile_strength": float(profile[selected]),
            "side_max": side_max,
            "distance_from_camera": float(abs(coords[selected] - center)),
        }

    left, left_meta = choose(x_profile, x_coords, "low", float(camera_xz[0]), float(fallback_lo[0]))
    right, right_meta = choose(x_profile, x_coords, "high", float(camera_xz[0]), float(fallback_hi[0]))
    bottom, bottom_meta = choose(z_profile, z_coords, "low", float(camera_xz[1]), float(fallback_lo[1]))
    top, top_meta = choose(z_profile, z_coords, "high", float(camera_xz[1]), float(fallback_hi[1]))
    out_lo = np.asarray([left, bottom], dtype=np.float64)
    out_hi = np.asarray([right, top], dtype=np.float64)
    if np.any(out_hi <= out_lo) or not np.all((camera_xz > out_lo) & (camera_xz < out_hi)):
        return fallback_lo.copy(), fallback_hi.copy(), {
            "enabled": False,
            "reason": "dominant_bounds_do_not_enclose_camera",
            "sides": {"left": left_meta, "right": right_meta, "bottom": bottom_meta, "top": top_meta},
        }
    return out_lo, out_hi, {
        "enabled": True,
        "camera_xz": camera_xz.tolist(),
        "dominant_lo_xz": out_lo.tolist(),
        "dominant_hi_xz": out_hi.tolist(),
        "sides": {"left": left_meta, "right": right_meta, "bottom": bottom_meta, "top": top_meta},
    }


def extract_roomformer_candidates(
    outputs: dict,
    thresholds: list[float],
    lo: np.ndarray,
    hi: np.ndarray,
    checkpoint_name: str,
    density_mode: str,
    min_corners: int,
    max_corners: int,
    target_corners: int,
) -> list[dict]:
    logits = outputs["pred_logits"][0].detach().float().cpu().numpy()
    coords = outputs["pred_coords"][0].detach().float().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    candidates = []
    for room_i in range(logits.shape[0]):
        for threshold in thresholds:
            keep = probs[room_i] >= threshold
            if int(np.count_nonzero(keep)) < 4:
                continue
            raw = sanitize_poly(coords[room_i][keep])
            if raw is None:
                continue
            variants = approx_in_corner_range(raw, min_corners, max_corners, target_corners)
            for kind, poly_norm in variants:
                poly_xz = normalized_to_xz(poly_norm, lo, hi)
                candidates.append(
                    {
                        "source": f"roomformer_{checkpoint_name}_{density_mode}_room{room_i:02d}_thr{threshold:.2f}_{kind}",
                        "poly_xz": poly_xz,
                        "corner_probs": probs[room_i][keep].astype(float).tolist(),
                        "valid_corner_count": int(np.count_nonzero(keep)),
                        "mean_prob": float(np.mean(probs[room_i][keep])),
                    }
                )
    return candidates


def build_scoring_support(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, args: argparse.Namespace) -> dict:
    grid = build_density_grid(points, lo, hi, int(args.score_grid_size))
    occ = occupancy_from_grid(
        grid,
        float(args.occupancy_threshold_quantile),
        int(args.close_iterations),
        int(args.dilate_iterations),
    )
    radial_mask, radial_meta = build_radial_free_space_mask(points, lo, hi, int(args.score_grid_size), args)
    if radial_mask is not None:
        dilate_px = max(0, int(args.radial_score_mask_dilate_px))
        allowed = radial_mask
        if dilate_px:
            allowed = cv2.dilate(
                allowed,
                np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8),
                iterations=1,
            )
        occ = cv2.bitwise_and(occ, allowed)
    occ = largest_component(occ)
    contours, _ = cv2.findContours((occ > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)[:, 0, :]
        bx = contour[:, 0]
        by = contour[:, 1]
    else:
        by, bx = np.where(cv2.morphologyEx(occ, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
    oy, ox = np.where(occ > 0)
    if len(bx) == 0:
        by, bx = oy, ox
    boundary_xz = np.column_stack(
        [
            lo[0] + (bx.astype(np.float64) + 0.5) / args.score_grid_size * (hi[0] - lo[0]),
            lo[1] + (by.astype(np.float64) + 0.5) / args.score_grid_size * (hi[1] - lo[1]),
        ]
    )
    occ_xz = np.column_stack(
        [
            lo[0] + (ox.astype(np.float64) + 0.5) / args.score_grid_size * (hi[0] - lo[0]),
            lo[1] + (oy.astype(np.float64) + 0.5) / args.score_grid_size * (hi[1] - lo[1]),
        ]
    )
    radial_boundary_xz = np.empty((0, 2), dtype=np.float64)
    radial_boundary_tree = None
    if radial_mask is not None:
        radial_contours, _ = cv2.findContours(
            (radial_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if radial_contours:
            radial_contour = max(radial_contours, key=cv2.contourArea)[:, 0, :]
            radial_boundary_xz = np.column_stack(
                [
                    lo[0]
                    + (radial_contour[:, 0].astype(np.float64) + 0.5)
                    / args.score_grid_size
                    * (hi[0] - lo[0]),
                    lo[1]
                    + (radial_contour[:, 1].astype(np.float64) + 0.5)
                    / args.score_grid_size
                    * (hi[1] - lo[1]),
                ]
            )
            radial_boundary_tree = cKDTree(radial_boundary_xz)
    wall_evidence_mask, wall_evidence_xz, wall_evidence_meta = build_vertical_wall_evidence(
        points, lo, hi, int(args.score_grid_size), args
    )
    room_span = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 1e-8)
    return {
        "grid": grid,
        "occupancy": occ,
        "boundary_xz": boundary_xz.astype(np.float64),
        "occupancy_xz": occ_xz.astype(np.float64),
        "boundary_tree": cKDTree(boundary_xz.astype(np.float64)),
        "occupancy_tree": cKDTree(occ_xz.astype(np.float64)) if len(occ_xz) else None,
        "radial_mask": radial_mask,
        "radial_meta": radial_meta,
        "radial_boundary_xz": radial_boundary_xz,
        "radial_boundary_tree": radial_boundary_tree,
        "wall_evidence_mask": wall_evidence_mask,
        "wall_evidence_xz": wall_evidence_xz,
        "wall_evidence_tree": cKDTree(wall_evidence_xz) if len(wall_evidence_xz) else None,
        "wall_evidence_meta": wall_evidence_meta,
        "wall_edge_distance_tol": max(0.018, float(args.wall_edge_distance_frac) * room_span),
        "wall_edge_support_target": float(args.wall_edge_support_target),
        "wall_edge_support_hard_min": float(args.wall_edge_support_hard_min),
        "wall_edge_supported_perimeter_hard_min": float(args.wall_edge_supported_perimeter_hard_min),
        "wall_edge_mean_support_hard_min": float(args.wall_edge_mean_support_hard_min),
        "wall_edge_support_weight": float(args.wall_edge_support_weight),
        "lo_xz": lo.astype(np.float64),
        "hi_xz": hi.astype(np.float64),
        "grid_size": int(args.score_grid_size),
        "corner_penalty": float(args.corner_penalty),
        "area_grid_size": 256,
        "occupancy_area_mask": cv2.resize(occ, (256, 256), interpolation=cv2.INTER_NEAREST),
    }


def refine_axis_aligned_by_boundary(poly: np.ndarray, boundary_xz: np.ndarray) -> np.ndarray | None:
    snapped = snap_manhattan_polygon(poly)
    if snapped is None or len(boundary_xz) < 20:
        return None
    span = max(float(np.ptp(boundary_xz[:, 0])), float(np.ptp(boundary_xz[:, 1])), 1e-6)
    tol = float(np.clip(0.038 * span, 0.035, 0.11))
    edges = np.roll(snapped, -1, axis=0) - snapped
    orientations = np.where(np.abs(edges[:, 0]) >= np.abs(edges[:, 1]), 0, 1)
    values = np.where(
        orientations == 0,
        0.5 * (snapped[:, 1] + np.roll(snapped[:, 1], -1)),
        0.5 * (snapped[:, 0] + np.roll(snapped[:, 0], -1)),
    )
    updated = values.copy()
    for i, center in enumerate(values):
        samples = boundary_xz[:, 1] if orientations[i] == 0 else boundary_xz[:, 0]
        distance = np.abs(samples - center)
        keep = distance <= tol
        if int(np.count_nonzero(keep)) < 16:
            keep = distance <= tol * 1.8
        if int(np.count_nonzero(keep)) >= 16:
            updated[i] = float(np.median(samples[keep]))
    out = np.empty_like(snapped)
    for i in range(snapped.shape[0]):
        previous = (i - 1) % snapped.shape[0]
        horizontal = previous if orientations[previous] == 0 else i
        vertical = previous if orientations[previous] == 1 else i
        out[i] = [updated[vertical], updated[horizontal]]
    return snap_manhattan_polygon(out)


def sample_edges(poly: np.ndarray, step: float = 0.012) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    normals = []
    poly = ensure_ccw(poly)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-8:
            continue
        n = max(8, int(math.ceil(length / step)))
        ts = (np.arange(n) + 0.5) / n
        epts = a[None, :] * (1.0 - ts[:, None]) + b[None, :] * ts[:, None]
        tangent = edge / length
        outward = np.array([tangent[1], -tangent[0]], dtype=np.float64)
        pts.append(epts)
        normals.append(np.tile(outward[None, :], (n, 1)))
    return np.concatenate(pts, axis=0), np.concatenate(normals, axis=0)


def axis_penalty(poly: np.ndarray) -> float:
    devs = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-8:
            continue
        unit = np.abs(edge / length)
        devs.append(float(min(unit[0], unit[1])))
    if not devs:
        return 1.0
    return float(np.percentile(devs, 80.0))


def score_poly(
    poly: np.ndarray,
    point_xz: np.ndarray,
    support: dict,
    min_corners: int,
    max_corners: int,
    target_corners: int,
    axis_weight: float,
    enforce_wall_edge_hard_min: bool = True,
) -> dict:
    poly = start_top_left(ensure_ccw(poly))
    geom = Polygon(poly)
    if not corner_count_allowed(poly.shape[0], min_corners, max_corners, target_corners) or not geom.is_valid or geom.area <= 1e-6:
        return {"score": 1e9, "invalid": True}
    camera_xz = support.get("camera_xz")
    if camera_xz is not None and not geom.covers(Point(np.asarray(camera_xz, dtype=np.float64))):
        return {"score": 1e9, "invalid": True, "invalid_reason": "candidate_does_not_enclose_camera"}
    path = MplPath(poly)
    inside_points = path.contains_points(point_xz)
    outside_point_frac = float(1.0 - np.mean(inside_points))

    occ_xz = support["occupancy_xz"]
    inside_occ = path.contains_points(occ_xz) if len(occ_xz) else np.array([], dtype=bool)
    outside_occ_frac = float(1.0 - np.mean(inside_occ)) if inside_occ.size else 1.0

    area = float(abs(polygon_area(poly)))
    occ_cell_area = 1.0
    if len(occ_xz) > 1:
        occ_cell_area = area / max(float(len(occ_xz)), 1.0)
    grid_size = int(support["area_grid_size"])
    lo_xz = support["lo_xz"]
    hi_xz = support["hi_xz"]
    span_xz = np.maximum(hi_xz - lo_xz, 1e-8)
    poly_px = np.empty_like(poly)
    poly_px[:, 0] = (poly[:, 0] - lo_xz[0]) / span_xz[0] * (grid_size - 1)
    poly_px[:, 1] = (poly[:, 1] - lo_xz[1]) / span_xz[1] * (grid_size - 1)
    candidate_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    cv2.fillPoly(candidate_mask, [np.round(poly_px).astype(np.int32).reshape(-1, 1, 2)], 255)
    candidate = candidate_mask > 0
    observed = support["occupancy_area_mask"] > 0
    empty_inside_frac = float(np.count_nonzero(candidate & ~observed) / max(1, np.count_nonzero(candidate)))

    radial_mask = support.get("radial_mask")
    radial_outside_frac = 0.0
    radial_empty_inside_frac = 0.0
    if radial_mask is not None:
        radial_observed = cv2.resize(radial_mask, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST) > 0
        radial_outside_frac = float(
            np.count_nonzero(radial_observed & ~candidate) / max(1, np.count_nonzero(radial_observed))
        )
        radial_empty_inside_frac = float(
            np.count_nonzero(candidate & ~radial_observed) / max(1, np.count_nonzero(candidate))
        )

    edge_pts, normals = sample_edges(poly)
    boundary_tree = support["boundary_tree"]
    edge_dists, _ = boundary_tree.query(edge_pts, k=1, workers=-1)
    occ_tree = support["occupancy_tree"]
    if occ_tree is not None:
        offset = max(0.035, math.sqrt(max(occ_cell_area, 1e-8)) * 2.0)
        din, _ = occ_tree.query(edge_pts - normals * offset, k=1, workers=-1)
        dout, _ = occ_tree.query(edge_pts + normals * offset, k=1, workers=-1)
        outside_empty_margin = float(np.mean(np.clip(dout - din, -0.25, 0.25)))
    else:
        outside_empty_margin = -1.0

    radial_edge_median_dist = 0.0
    radial_edge_p90_dist = 0.0
    radial_tree = support.get("radial_boundary_tree")
    if radial_tree is not None:
        radial_edge_dists, _ = radial_tree.query(edge_pts, k=1, workers=-1)
        radial_edge_median_dist = float(np.median(radial_edge_dists))
        radial_edge_p90_dist = float(np.percentile(radial_edge_dists, 90.0))

    wall_edge_support_fractions: list[float] = []
    wall_edge_median_distances: list[float] = []
    wall_edge_lengths: list[float] = []
    wall_tree = support.get("wall_evidence_tree")
    wall_tol = float(support.get("wall_edge_distance_tol", 0.05))
    if wall_tree is not None:
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            edge = b - a
            length = float(np.linalg.norm(edge))
            if length < 1e-8:
                continue
            count = max(12, int(math.ceil(length / max(0.5 * wall_tol, 0.006))))
            ts = (np.arange(count, dtype=np.float64) + 0.5) / count
            samples = a[None, :] * (1.0 - ts[:, None]) + b[None, :] * ts[:, None]
            distances, _ = wall_tree.query(samples, k=1, workers=-1)
            wall_edge_support_fractions.append(float(np.mean(distances <= wall_tol)))
            wall_edge_median_distances.append(float(np.median(distances)))
            wall_edge_lengths.append(length)
    if wall_edge_support_fractions:
        edge_support = np.asarray(wall_edge_support_fractions, dtype=np.float64)
        edge_lengths_arr = np.asarray(wall_edge_lengths, dtype=np.float64)
        wall_edge_min_support = float(np.min(edge_support))
        wall_edge_mean_support = float(np.average(edge_support, weights=np.maximum(edge_lengths_arr, 1e-8)))
        wall_target = float(support.get("wall_edge_support_target", 0.32))
        wall_hard_min = float(support.get("wall_edge_support_hard_min", 0.05))
        wall_supported_perimeter_fraction = float(
            np.sum(edge_lengths_arr[edge_support >= wall_hard_min])
            / max(float(np.sum(edge_lengths_arr)), 1e-8)
        )
        wall_supported_perimeter_hard_min = float(
            support.get("wall_edge_supported_perimeter_hard_min", 0.45)
        )
        wall_mean_support_hard_min = float(support.get("wall_edge_mean_support_hard_min", 0.08))
        wall_support_penalty = float(support.get("wall_edge_support_weight", 4.0)) * (
            max(0.0, wall_target - wall_edge_mean_support)
            + 1.8 * max(0.0, wall_target - wall_edge_min_support)
        )
        if enforce_wall_edge_hard_min and (
            wall_supported_perimeter_fraction < wall_supported_perimeter_hard_min
            or wall_edge_mean_support < wall_mean_support_hard_min
        ):
            return {
                "score": 1e9,
                "invalid": True,
                "invalid_reason": "insufficient_supported_wall_perimeter",
                "wall_edge_support_fractions": wall_edge_support_fractions,
                "wall_edge_median_distances": wall_edge_median_distances,
                "wall_edge_min_support": wall_edge_min_support,
                "wall_edge_mean_support": wall_edge_mean_support,
                "wall_edge_distance_tol": wall_tol,
                "wall_edge_supported_perimeter_fraction": wall_supported_perimeter_fraction,
                "wall_edge_supported_perimeter_hard_min": wall_supported_perimeter_hard_min,
                "wall_edge_mean_support_hard_min": wall_mean_support_hard_min,
            }
    else:
        wall_edge_min_support = 0.0
        wall_edge_mean_support = 0.0
        wall_supported_perimeter_fraction = 0.0
        wall_supported_perimeter_hard_min = float(
            support.get("wall_edge_supported_perimeter_hard_min", 0.45)
        )
        wall_mean_support_hard_min = float(support.get("wall_edge_mean_support_hard_min", 0.08))
        wall_support_penalty = 0.0

    axis = axis_penalty(poly)
    extra_corners = max(0, int(poly.shape[0]) - 4)
    if radial_tree is not None:
        room_span = max(
            float(np.ptp(poly[:, 0])),
            float(np.ptp(poly[:, 1])),
            1e-6,
        )
        radial_support_factor = float(np.clip(radial_edge_p90_dist / (0.035 * room_span), 0.12, 1.0))
    else:
        radial_support_factor = 1.0
    effective_corner_penalty = float(support["corner_penalty"]) * extra_corners * radial_support_factor
    total = (
        8.5 * outside_point_frac
        + 5.5 * outside_occ_frac
        + 0.45 * empty_inside_frac
        + 6.5 * radial_outside_frac
        + 0.65 * radial_empty_inside_frac
        + 4.3 * float(np.median(edge_dists))
        + 1.25 * float(np.percentile(edge_dists, 90.0))
        + 2.2 * radial_edge_median_dist
        + 0.75 * radial_edge_p90_dist
        + float(axis_weight) * axis
        - 0.9 * outside_empty_margin
        + effective_corner_penalty
        + wall_support_penalty
    )
    return {
        "score": float(total),
        "outside_point_frac": outside_point_frac,
        "outside_occupancy_frac": outside_occ_frac,
        "empty_inside_frac": empty_inside_frac,
        "radial_outside_frac": radial_outside_frac,
        "radial_empty_inside_frac": radial_empty_inside_frac,
        "edge_median_dist": float(np.median(edge_dists)),
        "edge_p90_dist": float(np.percentile(edge_dists, 90.0)),
        "radial_edge_median_dist": radial_edge_median_dist,
        "radial_edge_p90_dist": radial_edge_p90_dist,
        "outside_empty_margin": outside_empty_margin,
        "axis_penalty": axis,
        "axis_weight": float(axis_weight),
        "corner_penalty": effective_corner_penalty,
        "corner_penalty_support_factor": radial_support_factor,
        "wall_edge_support_fractions": wall_edge_support_fractions,
        "wall_edge_median_distances": wall_edge_median_distances,
        "wall_edge_min_support": wall_edge_min_support,
        "wall_edge_mean_support": wall_edge_mean_support,
        "wall_edge_supported_perimeter_fraction": wall_supported_perimeter_fraction,
        "wall_edge_supported_perimeter_hard_min": wall_supported_perimeter_hard_min,
        "wall_edge_mean_support_hard_min": wall_mean_support_hard_min,
        "wall_edge_distance_tol": wall_tol,
        "wall_edge_support_penalty": wall_support_penalty,
        "corner_count": int(poly.shape[0]),
        "area": area,
        "invalid": False,
    }


def expand_and_score_candidates(
    candidates: list[dict], point_xz: np.ndarray, support: dict, lo: np.ndarray, hi: np.ndarray, args: argparse.Namespace
) -> list[dict]:
    coarse_scored = []
    seen = set()
    if point_xz.shape[0] > int(args.coarse_point_sample):
        coarse_indices = np.linspace(0, point_xz.shape[0] - 1, int(args.coarse_point_sample)).astype(np.int64)
        coarse_points = point_xz[coarse_indices]
    else:
        coarse_points = point_xz
    unique_candidates = []
    seen_base = set()
    for candidate in candidates:
        poly = start_top_left(ensure_ccw(np.asarray(candidate["poly_xz"], dtype=np.float64)))
        key = tuple(np.round(poly.reshape(-1), 4).tolist())
        if key in seen_base:
            continue
        seen_base.add(key)
        unique_candidates.append(candidate)

    def candidate_priority(candidate: dict) -> tuple[float, float, str]:
        source = str(candidate.get("source", ""))
        # Keep every evidence family represented before the base-candidate
        # limit is applied.  Dense contour extraction can otherwise fill the
        # entire budget and silently remove the enclosing-wall prior and all
        # learned RoomFormer proposals.
        if source.startswith("vertical_wall_dominant_"):
            source_rank = 0.0
        elif "radial" in source or "axis_aligned_envelope" in source:
            source_rank = 1.0
        elif source.startswith("roomformer_"):
            source_rank = 2.0
        elif source.startswith("occupancy_contour_"):
            source_rank = 3.0
        else:
            source_rank = 4.0
        confidence_rank = -float(candidate.get("mean_prob", 0.0))
        return (source_rank, confidence_rank, source)

    unique_candidates.sort(key=candidate_priority)
    unique_candidates = unique_candidates[: max(1, int(args.max_base_candidates))]
    for cand in unique_candidates:
        base = start_top_left(ensure_ccw(np.asarray(cand["poly_xz"], dtype=np.float64)))
        variants = [(f"{cand['source']}__as_is", base)]
        snapped = snap_manhattan_polygon(base)
        if snapped is not None:
            variants.append((f"{cand['source']}__manhattan_snap", snapped))
            refined = refine_axis_aligned_by_boundary(snapped, support["boundary_xz"])
            if refined is not None:
                variants.append((f"{cand['source']}__edge_refine", refined))
        for d4_name, mat in d4_matrices():
            transformed = transform_in_bounds(base, mat, lo, hi)
            variants.append((f"{cand['source']}__{d4_name}", transformed))
            snap_t = snap_manhattan_polygon(transformed)
            if snap_t is not None:
                variants.append((f"{cand['source']}__{d4_name}_manhattan_snap", snap_t))
                refined_t = refine_axis_aligned_by_boundary(snap_t, support["boundary_xz"])
                if refined_t is not None:
                    variants.append((f"{cand['source']}__{d4_name}_edge_refine", refined_t))
        for name, poly in variants:
            poly = normalize_polygon_vertices(poly)
            if poly.shape[0] < int(args.min_corners) or poly.shape[0] > int(args.max_corners):
                continue
            poly = start_top_left(ensure_ccw(poly))
            key = tuple(np.round(poly.reshape(-1), 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            metrics = score_poly(
                poly,
                coarse_points,
                support,
                args.min_corners,
                args.max_corners,
                args.target_corners,
                args.axis_weight,
            )
            if metrics.get("invalid"):
                continue
            item = {
                "source": name,
                "score": metrics["score"],
                "metrics": metrics,
                "polygon_xz": poly.tolist(),
            }
            for k in ("mean_prob", "valid_corner_count", "raw_score", "radial_step"):
                if k in cand:
                    item[k] = cand[k]
            coarse_scored.append(item)
    coarse_scored.sort(key=lambda x: float(x["score"]))
    finalists = coarse_scored[: max(1, int(args.max_expanded_candidates))]
    # Always retain the best enclosing-wall variant for full-resolution
    # scoring.  This is essential for rooms with large door/window openings,
    # where rays through the opening can make an outer contour look better at
    # the coarse stage even though its edges have no enclosing-wall support.
    finalist_sources = {str(item["source"]) for item in finalists}
    dominant_items = [
        item for item in coarse_scored if str(item["source"]).startswith("vertical_wall_dominant_")
    ]
    if dominant_items and not any(source.startswith("vertical_wall_dominant_") for source in finalist_sources):
        finalists.append(dominant_items[0])
        finalist_sources.add(str(dominant_items[0]["source"]))

    # Coarse point support can unfairly suppress the Manhattan projection of a
    # RoomFormer topology when one long wall is noisy or sparsely observed.
    # Retain a bounded set of those topology-preserving alternatives for the
    # full-resolution pass; this does not impose any fixed room shape or corner
    # count, it only prevents the coarse shortlist from discarding them.
    roomformer_manhattan_items = [
        item
        for item in coarse_scored
        if str(item["source"]).startswith("roomformer_")
        and ("__manhattan_snap" in str(item["source"]) or "__edge_refine" in str(item["source"]))
    ]
    for item in roomformer_manhattan_items[:64]:
        source = str(item["source"])
        if source not in finalist_sources:
            finalists.append(item)
            finalist_sources.add(source)
    scored = []
    for item in finalists:
        poly = np.asarray(item["polygon_xz"], dtype=np.float64)
        metrics = score_poly(
            poly,
            point_xz,
            support,
            args.min_corners,
            args.max_corners,
            args.target_corners,
            args.axis_weight,
        )
        if metrics.get("invalid"):
            continue
        item = dict(item)
        item["coarse_score"] = float(item["score"])
        item["score"] = float(metrics["score"])
        item["metrics"] = metrics
        scored.append(item)
    scored.sort(key=lambda x: float(x["score"]))
    return scored


def select_supported_topology_candidate(
    scored: list[dict], args: argparse.Namespace
) -> tuple[list[dict], dict]:
    """Keep observed wall turns that a simpler envelope would flatten.

    The ordinary objective includes a complexity prior, which is useful for
    suppressing noisy contour wiggles.  It can, however, overrule a real inset
    even when all of the inset walls recur across height bands.  This guard
    compares candidates only by evidence: a richer topology must improve both
    multi-height wall support and radial boundary agreement, exclude a
    meaningful unsupported pocket, and retain nearly the same room area.  No
    source, room shape, or corner count is assumed.
    """
    if not args.supported_topology_preservation_guard:
        return scored, {"applied": False, "reason": "supported_topology_guard_disabled"}
    if not scored:
        return scored, {"applied": False, "reason": "no_scored_candidates"}

    ordinary = scored[0]
    ordinary_metrics = ordinary.get("metrics", {})
    ordinary_area = float(ordinary_metrics.get("area", 0.0))
    ordinary_corners = int(ordinary_metrics.get("corner_count", 0))
    if ordinary_area <= 0.0 or ordinary_corners <= 0:
        return scored, {"applied": False, "reason": "ordinary_candidate_missing_geometry"}

    ordinary_edge_min = float(ordinary_metrics.get("wall_edge_min_support", 0.0))
    ordinary_edge_mean = float(ordinary_metrics.get("wall_edge_mean_support", 0.0))
    ordinary_radial_p90 = float(ordinary_metrics.get("radial_edge_p90_dist", 0.0))
    ordinary_radial_empty = float(ordinary_metrics.get("radial_empty_inside_frac", 0.0))
    ordinary_outside_occ = float(ordinary_metrics.get("outside_occupancy_frac", 0.0))
    ordinary_radial_outside = float(ordinary_metrics.get("radial_outside_frac", 0.0))
    ordinary_axis = float(ordinary_metrics.get("axis_penalty", 0.0))

    qualified = []
    for item in scored[1:]:
        metrics = item.get("metrics", {})
        corner_count = int(metrics.get("corner_count", 0))
        if corner_count <= ordinary_corners:
            continue
        area_ratio = float(metrics.get("area", 0.0)) / ordinary_area
        edge_min = float(metrics.get("wall_edge_min_support", 0.0))
        edge_mean = float(metrics.get("wall_edge_mean_support", 0.0))
        radial_p90 = float(metrics.get("radial_edge_p90_dist", float("inf")))
        radial_empty_improvement = ordinary_radial_empty - float(
            metrics.get("radial_empty_inside_frac", 1.0)
        )
        outside_occ_increase = float(metrics.get("outside_occupancy_frac", 1.0)) - ordinary_outside_occ
        radial_outside_increase = float(metrics.get("radial_outside_frac", 1.0)) - ordinary_radial_outside
        if (
            float(args.supported_topology_min_area_ratio)
            <= area_ratio
            <= float(args.supported_topology_max_area_ratio)
            and edge_min
            >= max(
                float(args.supported_topology_min_edge_support),
                ordinary_edge_min + float(args.supported_topology_min_edge_support_gain),
            )
            and edge_mean
            >= max(
                float(args.supported_topology_min_mean_support),
                ordinary_edge_mean + float(args.supported_topology_min_mean_support_gain),
            )
            and ordinary_radial_p90 > 0.0
            and radial_p90
            <= float(args.supported_topology_max_radial_edge_p90_ratio) * ordinary_radial_p90
            and radial_empty_improvement
            >= float(args.supported_topology_min_radial_empty_improvement)
            and outside_occ_increase
            <= float(args.supported_topology_max_outside_occupancy_increase)
            and radial_outside_increase
            <= float(args.supported_topology_max_radial_outside_increase)
            and float(metrics.get("axis_penalty", float("inf"))) <= max(0.02, ordinary_axis + 0.01)
        ):
            qualified.append(
                (
                    float(item["score"]),
                    -edge_min,
                    -edge_mean,
                    radial_p90,
                    str(item.get("source", "")),
                    area_ratio,
                    radial_empty_improvement,
                    outside_occ_increase,
                    radial_outside_increase,
                    item,
                )
            )

    if not qualified:
        return scored, {
            "applied": False,
            "reason": "no_richer_topology_met_persistent_wall_and_radial_evidence",
            "ordinary_best_source": ordinary.get("source"),
        }
    qualified.sort(key=lambda row: row[:5])
    (
        _,
        _,
        _,
        _,
        _,
        area_ratio,
        radial_empty_improvement,
        outside_occ_increase,
        radial_outside_increase,
        selected,
    ) = qualified[0]
    reordered = [selected] + [item for item in scored if item is not selected]
    return reordered, {
        "applied": selected is not ordinary,
        "method": "persistent_multi_height_wall_turn_preservation_guard",
        "ordinary_best_source": ordinary.get("source"),
        "ordinary_corner_count": ordinary_corners,
        "selected_source": selected.get("source"),
        "selected_corner_count": int(selected["metrics"]["corner_count"]),
        "selected_area_ratio_to_ordinary": float(area_ratio),
        "selected_min_wall_support": float(selected["metrics"]["wall_edge_min_support"]),
        "selected_mean_wall_support": float(selected["metrics"]["wall_edge_mean_support"]),
        "selected_radial_edge_p90": float(selected["metrics"]["radial_edge_p90_dist"]),
        "selected_radial_empty_improvement": float(radial_empty_improvement),
        "selected_outside_occupancy_increase": float(outside_occ_increase),
        "selected_radial_outside_increase": float(radial_outside_increase),
        "qualified_candidate_count": int(len(qualified)),
    }


def select_opening_aware_candidate(scored: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Prefer a strongly supported enclosure when door/window rays escape it.

    Ordinary occupancy scoring assumes every observed point belongs inside the
    room shell.  That is false for open doors and windows: distant geometry is
    visible through an otherwise well-supported enclosing wall plane.  A
    candidate only enters this override when *every* edge and the
    length-weighted mean have strong multi-height vertical-wall support.
    """
    if not args.opening_aware_enclosure_override:
        return scored, {"applied": False, "reason": "opening_aware_enclosure_override_disabled"}
    if not scored:
        return scored, {"applied": False, "reason": "no_scored_candidates"}
    minimum = float(args.wall_enclosure_override_min_support)
    mean_minimum = float(args.wall_enclosure_override_mean_support)
    ordinary = scored[0]
    ordinary_metrics = ordinary.get("metrics", {})
    ordinary_area = float(ordinary_metrics.get("area", 0.0))
    guard_candidates = []
    if (
        args.dominant_wall_enclosure_guard
        and int(ordinary_metrics.get("corner_count", 0)) == 4
        and ordinary_area > 0.0
    ):
        for item in scored:
            source = str(item.get("source", ""))
            metrics = item.get("metrics", {})
            area_ratio = float(metrics.get("area", 0.0)) / ordinary_area
            if (
                source.startswith("vertical_wall_dominant_envelope_4__edge_refine")
                and float(metrics.get("wall_edge_min_support", 0.0))
                >= float(args.dominant_wall_enclosure_guard_min_support)
                and float(metrics.get("wall_edge_mean_support", 0.0))
                >= float(args.dominant_wall_enclosure_guard_mean_support)
                and float(args.dominant_wall_enclosure_guard_min_area_ratio)
                <= area_ratio
                <= float(args.dominant_wall_enclosure_guard_max_area_ratio)
            ):
                guard_candidates.append((float(metrics["score"]), area_ratio, item))
    if guard_candidates:
        guard_candidates.sort(key=lambda row: (row[0], -row[2]["metrics"]["wall_edge_mean_support"], str(row[2]["source"])))
        _, area_ratio, selected = guard_candidates[0]
        reordered = [selected] + [item for item in scored if item is not selected]
        return reordered, {
            "applied": selected is not ordinary,
            "method": "dominant_multi_height_wall_enclosure_guard",
            "ordinary_best_source": ordinary["source"],
            "selected_source": selected["source"],
            "selected_edge_support": float(selected["metrics"]["wall_edge_min_support"]),
            "selected_mean_support": float(selected["metrics"]["wall_edge_mean_support"]),
            "selected_area_ratio_to_ordinary": float(area_ratio),
            "guard_minimum_edge_support": float(args.dominant_wall_enclosure_guard_min_support),
            "guard_minimum_mean_support": float(args.dominant_wall_enclosure_guard_mean_support),
            "guard_area_ratio_range": [
                float(args.dominant_wall_enclosure_guard_min_area_ratio),
                float(args.dominant_wall_enclosure_guard_max_area_ratio),
            ],
        }
    learned_topology_candidates = []
    if (
        args.high_confidence_learned_topology_guard
        and int(ordinary_metrics.get("corner_count", 0)) == 4
        and ordinary_area > 0.0
    ):
        ordinary_empty_inside = float(ordinary_metrics.get("empty_inside_frac", 1.0))
        ordinary_corner_count = int(ordinary_metrics.get("corner_count", 0))
        for item in scored:
            source = str(item.get("source", ""))
            metrics = item.get("metrics", {})
            area_ratio = float(metrics.get("area", 0.0)) / ordinary_area
            empty_inside_improvement = ordinary_empty_inside - float(
                metrics.get("empty_inside_frac", 1.0)
            )
            if (
                source.startswith("roomformer_")
                and int(metrics.get("corner_count", 0)) > ordinary_corner_count
                and float(item.get("mean_prob", 0.0))
                >= float(args.learned_topology_min_probability)
                and float(metrics.get("outside_occupancy_frac", 1.0))
                <= float(args.learned_topology_max_outside_occupancy)
                and float(metrics.get("radial_outside_frac", 1.0))
                <= float(args.learned_topology_max_radial_outside)
                and empty_inside_improvement
                >= float(args.learned_topology_min_empty_inside_improvement)
                and float(args.learned_topology_min_area_ratio)
                <= area_ratio
                <= float(args.learned_topology_max_area_ratio)
                and float(metrics.get("wall_edge_supported_perimeter_fraction", 0.0))
                >= float(args.learned_topology_min_supported_perimeter)
                and float(metrics.get("wall_edge_mean_support", 0.0))
                >= float(args.learned_topology_min_mean_wall_support)
            ):
                learned_topology_candidates.append(
                    (
                        float(metrics["score"]),
                        -float(item.get("mean_prob", 0.0)),
                        str(item.get("source", "")),
                        area_ratio,
                        empty_inside_improvement,
                        item,
                    )
                )
    if learned_topology_candidates:
        learned_topology_candidates.sort(key=lambda row: row[:3])
        _, _, _, area_ratio, empty_inside_improvement, selected = learned_topology_candidates[0]
        reordered = [selected] + [item for item in scored if item is not selected]
        return reordered, {
            "applied": selected is not ordinary,
            "method": "high_confidence_learned_topology_preservation_guard",
            "ordinary_best_source": ordinary["source"],
            "selected_source": selected["source"],
            "selected_corner_count": int(selected["metrics"]["corner_count"]),
            "selected_mean_probability": float(selected.get("mean_prob", 0.0)),
            "selected_area_ratio_to_ordinary": float(area_ratio),
            "selected_empty_inside_improvement": float(empty_inside_improvement),
            "selected_outside_occupancy_fraction": float(
                selected["metrics"].get("outside_occupancy_frac", 1.0)
            ),
            "selected_radial_outside_fraction": float(
                selected["metrics"].get("radial_outside_frac", 1.0)
            ),
            "selected_supported_perimeter_fraction": float(
                selected["metrics"].get("wall_edge_supported_perimeter_fraction", 0.0)
            ),
            "selected_mean_wall_support": float(
                selected["metrics"].get("wall_edge_mean_support", 0.0)
            ),
        }
    single_opening_candidates = []
    if (
        args.dominant_wall_single_opening_guard
        and int(ordinary_metrics.get("corner_count", 0)) == 4
        and ordinary_area > 0.0
    ):
        for item in scored:
            source = str(item.get("source", ""))
            metrics = item.get("metrics", {})
            edge_support = sorted(
                [float(value) for value in metrics.get("wall_edge_support_fractions", [])],
                reverse=True,
            )
            area_ratio = float(metrics.get("area", 0.0)) / ordinary_area
            if (
                source.startswith("vertical_wall_dominant_envelope_4__")
                and len(edge_support) == 4
                and edge_support[2]
                >= float(args.dominant_wall_single_opening_supported_edge_min)
                and edge_support[3] >= float(args.dominant_wall_single_opening_edge_min)
                and float(metrics.get("wall_edge_mean_support", 0.0))
                >= float(args.dominant_wall_single_opening_mean_support)
                and float(args.dominant_wall_single_opening_min_area_ratio)
                <= area_ratio
                <= float(args.dominant_wall_single_opening_max_area_ratio)
            ):
                single_opening_candidates.append(
                    (
                        -edge_support[2],
                        -float(metrics.get("wall_edge_mean_support", 0.0)),
                        float(metrics["score"]),
                        str(item.get("source", "")),
                        area_ratio,
                        edge_support,
                        item,
                    )
                )
    if single_opening_candidates:
        single_opening_candidates.sort(key=lambda row: row[:4])
        _, _, _, _, area_ratio, edge_support, selected = single_opening_candidates[0]
        reordered = [selected] + [item for item in scored if item is not selected]
        return reordered, {
            "applied": selected is not ordinary,
            "method": "dominant_three_wall_single_opening_enclosure_guard",
            "ordinary_best_source": ordinary["source"],
            "selected_source": selected["source"],
            "selected_edge_support_descending": edge_support,
            "selected_mean_support": float(selected["metrics"]["wall_edge_mean_support"]),
            "selected_area_ratio_to_ordinary": float(area_ratio),
            "guard_supported_edge_minimum": float(
                args.dominant_wall_single_opening_supported_edge_min
            ),
            "guard_opening_edge_minimum": float(args.dominant_wall_single_opening_edge_min),
            "guard_mean_support_minimum": float(args.dominant_wall_single_opening_mean_support),
            "guard_area_ratio_range": [
                float(args.dominant_wall_single_opening_min_area_ratio),
                float(args.dominant_wall_single_opening_max_area_ratio),
            ],
        }
    qualified = []
    for item in scored:
        metrics = item.get("metrics", {})
        if (
            float(metrics.get("wall_edge_min_support", 0.0)) >= minimum
            and float(metrics.get("wall_edge_mean_support", 0.0)) >= mean_minimum
        ):
            # Ignore only the terms that count evidence beyond the candidate
            # as an error.  Boundary fit, empty interior, axis alignment,
            # complexity, and wall support continue to decide among valid
            # enclosures.
            opening_aware_score = float(metrics["score"])
            opening_aware_score -= 8.5 * float(metrics.get("outside_point_frac", 0.0))
            opening_aware_score -= 5.5 * float(metrics.get("outside_occupancy_frac", 0.0))
            opening_aware_score -= 6.5 * float(metrics.get("radial_outside_frac", 0.0))
            qualified.append((opening_aware_score, item))
    if not qualified:
        return scored, {
            "applied": False,
            "reason": "no_candidate_met_strong_enclosure_support",
            "minimum_edge_support": minimum,
            "minimum_mean_support": mean_minimum,
            "ordinary_best_source": scored[0]["source"],
        }
    qualified.sort(key=lambda pair: (pair[0], float(pair[1]["score"]), str(pair[1]["source"])))
    opening_score, selected = qualified[0]
    reordered = [selected] + [item for item in scored if item is not selected]
    return reordered, {
        "applied": selected is not scored[0],
        "method": "strong_multi_height_wall_enclosure_ignoring_only_outside_ray_terms",
        "minimum_edge_support": minimum,
        "minimum_mean_support": mean_minimum,
        "qualified_candidates": int(len(qualified)),
        "ordinary_best_source": scored[0]["source"],
        "selected_source": selected["source"],
        "selected_opening_aware_score": float(opening_score),
        "selected_edge_support": float(selected["metrics"]["wall_edge_min_support"]),
        "selected_mean_support": float(selected["metrics"]["wall_edge_mean_support"]),
    }


def stabilize_selected_topology(
    scored: list[dict],
    point_xz: np.ndarray,
    support: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    """Straighten the selected topology without changing its corner count.

    A long wall seen mainly through doors or windows may have enough evidence to
    establish the RoomFormer topology while its raw endpoints remain slightly
    diagonal.  Refit only that already-selected topology to horizontal/vertical
    boundary lines, and accept it only when the displacement is small, the
    length-weighted wall support is retained, and the full score improves.  The
    sparse-edge hard gate is relaxed only for this one topology-preserving test;
    its ordinary wall-support penalty remains in the score.
    """
    if not scored:
        return scored, {"applied": False, "reason": "no_scored_candidates"}
    selected = scored[0]
    poly = start_top_left(ensure_ccw(np.asarray(selected["polygon_xz"], dtype=np.float64)))
    if poly.shape[0] < 4 or poly.shape[0] % 2:
        return scored, {"applied": False, "reason": "selected_topology_not_even_sided"}
    refined = refine_axis_aligned_by_boundary(poly, support["boundary_xz"])
    if refined is None or refined.shape != poly.shape:
        return scored, {"applied": False, "reason": "no_valid_axis_aligned_refit"}
    room_span = max(float(np.ptp(poly[:, 0])), float(np.ptp(poly[:, 1])), 1e-8)
    max_displacement = float(np.max(np.linalg.norm(refined - poly, axis=1)))
    area_before = float(abs(polygon_area(poly)))
    area_after = float(abs(polygon_area(refined)))
    area_ratio = area_after / max(area_before, 1e-8)
    if max_displacement > 0.10 * room_span or not 0.85 <= area_ratio <= 1.15:
        return scored, {
            "applied": False,
            "reason": "refit_changed_selected_topology_too_much",
            "max_displacement": max_displacement,
            "room_span": room_span,
            "area_ratio": area_ratio,
        }
    metrics = score_poly(
        refined,
        point_xz,
        support,
        args.min_corners,
        args.max_corners,
        args.target_corners,
        args.axis_weight,
        enforce_wall_edge_hard_min=False,
    )
    if metrics.get("invalid"):
        return scored, {"applied": False, "reason": "refit_failed_full_resolution_scoring"}
    selected_metrics = selected.get("metrics", {})
    original_axis = float(selected_metrics.get("axis_penalty", 1.0))
    refined_axis = float(metrics.get("axis_penalty", 1.0))
    original_mean_support = float(selected_metrics.get("wall_edge_mean_support", 0.0))
    refined_mean_support = float(metrics.get("wall_edge_mean_support", 0.0))
    if refined_axis >= original_axis - 1e-4:
        return scored, {"applied": False, "reason": "refit_did_not_improve_axis_alignment"}
    if refined_mean_support < max(0.10, 0.80 * original_mean_support):
        return scored, {
            "applied": False,
            "reason": "refit_lost_too_much_length_weighted_wall_support",
            "original_mean_support": original_mean_support,
            "refined_mean_support": refined_mean_support,
        }
    if float(metrics["score"]) >= float(selected["score"]) - 0.05:
        return scored, {
            "applied": False,
            "reason": "refit_did_not_improve_full_score",
            "original_score": float(selected["score"]),
            "refined_score": float(metrics["score"]),
        }
    stabilized = dict(selected)
    stabilized["source"] = f"{selected['source']}__topology_preserving_edge_refine"
    stabilized["score"] = float(metrics["score"])
    stabilized["metrics"] = metrics
    stabilized["polygon_xz"] = refined.tolist()
    reordered = [stabilized] + scored
    return reordered, {
        "applied": True,
        "method": "selected_topology_axis_refit_with_soft_sparse_edge_gate",
        "original_source": selected["source"],
        "selected_source": stabilized["source"],
        "corner_count": int(poly.shape[0]),
        "original_score": float(selected["score"]),
        "refined_score": float(metrics["score"]),
        "original_axis_penalty": original_axis,
        "refined_axis_penalty": refined_axis,
        "max_displacement": max_displacement,
        "room_span": room_span,
        "area_ratio": area_ratio,
        "original_mean_support": original_mean_support,
        "refined_mean_support": refined_mean_support,
    }


def make_structure(poly: np.ndarray, points_room: np.ndarray, world_from_room: np.ndarray, room_from_world: np.ndarray, source: str) -> dict:
    poly = start_top_left(ensure_ccw(poly))
    floor_y, ceil_y = np.percentile(points_room[:, 1], [2.0, 98.0])
    floor = np.column_stack([poly[:, 0], np.full(poly.shape[0], floor_y), poly[:, 1]])
    ceil = np.column_stack([poly[:, 0], np.full(poly.shape[0], ceil_y), poly[:, 1]])
    n = int(poly.shape[0])
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
        "method": "roomformer_da3_variable_corner_manhattan_search_v2",
        "selected_source": source,
        "coordinate_space": "da3_exported_glb",
        "vertices": vertices.tolist(),
        "room_vertices": room_vertices.tolist(),
        "faces": faces,
        "floor_corner_count": n,
        "ceiling_corner_count": n,
        "floor_polygon_xz": poly.tolist(),
        "vertical_bounds": {
            "floor_y": float(floor_y),
            "ceiling_y": float(ceil_y),
            "height": float(ceil_y - floor_y),
            "percentiles": [2.0, 98.0],
        },
        "world_from_room_matrix": world_from_room.tolist(),
        "room_from_world_matrix": room_from_world.tolist(),
    }


def structure_lines(structure: dict, room_space: bool = False) -> list[np.ndarray]:
    vertices = np.asarray(structure["room_vertices" if room_space else "vertices"], dtype=float)
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
        f.write("# RoomFormer + DA3 variable-corner Manhattan structure\n")
        for v in structure["vertices"]:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in structure["faces"]:
            f.write(f"o {face['name']}\n")
            f.write("f " + " ".join(str(int(i) + 1) for i in face["vertices"]) + "\n")


def save_density_images(out_dir: Path, images: dict[str, np.ndarray], occ: np.ndarray) -> None:
    for name, img in images.items():
        Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8)).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(out_dir / f"density_{name}.png")
    Image.fromarray(occ).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(out_dir / "occupancy_score_mask.png")


def plot_summary(
    path: Path,
    points_room: np.ndarray,
    density: np.ndarray,
    occ: np.ndarray,
    scored: list[dict],
    structure: dict,
    plot_points: int,
) -> None:
    sample = points_room
    if sample.shape[0] > plot_points:
        idx = np.linspace(0, sample.shape[0] - 1, plot_points).astype(np.int64)
        sample = sample[idx]
    best_poly = np.asarray(scored[0]["polygon_xz"], dtype=float)
    fig = plt.figure(figsize=(18, 10), dpi=170)
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.imshow(density, cmap="gray", origin="lower")
    ax1.set_title("RoomFormer input density")
    ax1.axis("off")

    ax2.imshow(occ, cmap="gray", origin="lower")
    ax2.set_title("DA3 occupancy used for scoring")
    ax2.axis("off")

    ax3.scatter(sample[:, 0], sample[:, 2], s=0.08, c="#666666", alpha=0.16)
    colors = ["#e53935", "#1976d2", "#ff9800", "#8e24aa", "#00897b", "#6d4c41"]
    for rank, item in enumerate(scored[: min(6, len(scored))]):
        poly = np.asarray(item["polygon_xz"], dtype=float)
        closed = np.vstack([poly, poly[:1]])
        ax3.plot(closed[:, 0], closed[:, 1], "-o", c=colors[rank % len(colors)], lw=1.1 if rank else 2.2, ms=2.6, alpha=0.78)
    for i, p in enumerate(best_poly):
        ax3.text(p[0], p[1], str(i), fontsize=8)
    ax3.set_title(f"Top candidates on leveled DA3; best score={scored[0]['score']:.3f}")
    ax3.set_aspect("equal", adjustable="box")

    ax4.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.08, c="#666666", alpha=0.13)
    for line in structure_lines(structure, room_space=True):
        ax4.plot(line[:, 0], line[:, 1], line[:, 2], c="#e53935", lw=1.6)
    ax4.set_title("Leveled room structure")
    ax4.view_init(elev=18, azim=-62)

    ax5.scatter(sample[:, 0], sample[:, 1], s=0.08, c="#666666", alpha=0.16)
    for line in structure_lines(structure, room_space=True):
        ax5.plot(line[:, 0], line[:, 1], c="#e53935", lw=1.5)
    ax5.set_title("Side x-y, leveled")
    ax5.set_aspect("equal", adjustable="box")

    ax6.scatter(sample[:, 2], sample[:, 1], s=0.08, c="#666666", alpha=0.16)
    for line in structure_lines(structure, room_space=True):
        ax6.plot(line[:, 2], line[:, 1], c="#e53935", lw=1.5)
    ax6.set_title("Side z-y, leveled")
    ax6.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_world_summary(path: Path, points_world: np.ndarray, structure: dict, plot_points: int) -> None:
    sample = points_world
    if sample.shape[0] > plot_points:
        idx = np.linspace(0, sample.shape[0] - 1, plot_points).astype(np.int64)
        sample = sample[idx]
    poly = np.asarray(structure["vertices"], dtype=float)
    fig = plt.figure(figsize=(16, 5.5), dpi=170)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=0.08, c="#666666", alpha=0.13)
    for line in structure_lines(structure, room_space=False):
        ax1.plot(line[:, 0], line[:, 1], line[:, 2], c="#e53935", lw=1.5)
    ax1.set_title("World DA3 GLB + structure")
    ax1.view_init(elev=18, azim=-62)
    ax2.scatter(sample[:, 0], sample[:, 2], s=0.08, c="#666666", alpha=0.16)
    floor_count = int(structure["floor_corner_count"])
    floor = poly[:floor_count][:, [0, 2]]
    floor_closed = np.vstack([floor, floor[:1]])
    ax2.plot(floor_closed[:, 0], floor_closed[:, 1], "-o", c="#e53935", lw=1.6, ms=2.8)
    ax2.set_title("World top x-z")
    ax2.set_aspect("equal", adjustable="box")
    ax3.scatter(sample[:, 2], sample[:, 1], s=0.08, c="#666666", alpha=0.16)
    for line in structure_lines(structure, room_space=False):
        ax3.plot(line[:, 2], line[:, 1], c="#e53935", lw=1.3)
    ax3.set_title("World side z-y")
    ax3.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_contact_sheet(scored: list[dict], out_path: Path, top_k: int) -> None:
    thumbs = []
    for rank, item in enumerate(scored[:top_k]):
        canvas = Image.new("RGB", (520, 300), "white")
        draw = ImageDraw.Draw(canvas)
        poly = np.asarray(item["polygon_xz"], dtype=np.float64)
        lo = np.min(poly, axis=0)
        hi = np.max(poly, axis=0)
        span = np.maximum(hi - lo, 1e-6)
        pts = (poly - lo[None, :]) / span[None, :]
        pts[:, 1] = 1.0 - pts[:, 1]
        pts[:, 0] = 30 + pts[:, 0] * 460
        pts[:, 1] = 54 + pts[:, 1] * 210
        draw.text((12, 10), f"{rank}: score={item['score']:.4f}", fill=(0, 0, 0))
        draw.text((12, 28), item["source"][:86], fill=(0, 0, 0))
        draw.line([tuple(p) for p in np.vstack([pts, pts[:1]])], fill=(220, 40, 40), width=3)
        for i, p in enumerate(pts):
            draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=(25, 118, 210))
            draw.text((p[0] + 5, p[1] - 8), str(i), fill=(0, 0, 0))
        thumbs.append(canvas)
    if not thumbs:
        return
    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 520, rows * 300), (245, 245, 245))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 520, (i // cols) * 300))
    sheet.save(out_path)


def main() -> int:
    args = parse_args()
    if args.min_corners < 4 or args.max_corners < args.min_corners:
        raise ValueError("Require 4 <= min-corners <= max-corners")
    if args.target_corners and not (args.min_corners <= args.target_corners <= args.max_corners):
        raise ValueError("target-corners must be 0 or lie inside [min-corners, max-corners]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points_world = load_point_cloud(args.scene_glb)
    base_world_from_room, _, basis_meta = estimate_room_basis(points_world, basis_args(args))
    points_base = world_to_room_points(points_world, np.linalg.inv(base_world_from_room))

    if args.disable_yaw_align:
        yaw = 0.0
        yaw_meta = {"method": "disabled", "yaw_degrees": 0.0, "line_count": 0}
    else:
        yaw, yaw_meta = estimate_manhattan_yaw(points_base, args)
    world_from_room, room_from_world = compose_yaw_basis(base_world_from_room, yaw)
    points_room = world_to_room_points(points_world, room_from_world)

    base_lo = np.percentile(points_room[:, [0, 2]], args.bounds_percentiles[0], axis=0).astype(np.float64)
    base_hi = np.percentile(points_room[:, [0, 2]], args.bounds_percentiles[1], axis=0).astype(np.float64)
    lo, hi, radial_bounds_meta = radial_evidence_bounds(points_room, base_lo, base_hi, args)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - 0.015 * span
    hi = hi + 0.015 * span

    rf_grid = build_density_grid(points_room, lo, hi, int(args.grid_size))
    rf_occ = occupancy_from_grid(
        rf_grid,
        float(args.occupancy_threshold_quantile),
        int(args.close_iterations),
        int(args.dilate_iterations),
    )
    rf_occ = largest_component(rf_occ)
    density_images = {mode: normalize_density(rf_grid, rf_occ, mode) for mode in args.density_modes}
    save_density_images(args.out_dir, density_images, rf_occ)

    support = build_scoring_support(points_room, lo, hi, args)
    if support.get("radial_mask") is not None:
        Image.fromarray(support["radial_mask"]).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(
            args.out_dir / "radial_free_space_score_mask.png"
        )
    Image.fromarray(support["wall_evidence_mask"]).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(
        args.out_dir / "vertical_wall_evidence_mask.png"
    )
    camera_room = world_to_room_points(np.zeros((1, 3), dtype=np.float64), room_from_world)[0]
    support["camera_xz"] = camera_room[[0, 2]].astype(np.float64)
    dominant_base_lo = base_lo.copy()
    dominant_base_hi = base_hi.copy()
    dominant_base_meta = {"enabled": False, "reason": "disabled"}
    if bool(args.dominant_base_from_wall_evidence):
        dominant_base_lo, dominant_base_hi, dominant_base_meta = dominant_enclosing_axis_bounds(
            support["wall_evidence_mask"],
            lo,
            hi,
            camera_room[[0, 2]],
            base_lo,
            base_hi,
        )
    point_xz = points_room[:, [0, 2]]
    valid = np.all((point_xz >= lo[None, :]) & (point_xz <= hi[None, :]), axis=1)
    point_xz = point_xz[valid]
    point_count_before_radial_filter = int(point_xz.shape[0])
    point_xz = filter_points_to_radial_mask(point_xz, support, int(args.radial_score_mask_dilate_px))
    point_count_after_radial_filter = int(point_xz.shape[0])
    if point_xz.shape[0] > args.point_sample:
        idx = np.linspace(0, point_xz.shape[0] - 1, int(args.point_sample)).astype(np.int64)
        point_xz = point_xz[idx]

    all_candidates: list[dict] = []
    radial_candidates: list[dict] = []
    radial_step_candidates_list: list[dict] = []
    if args.candidate_source_policy == "all":
        all_candidates.extend(
            contour_polygon_candidates(
                support["occupancy"],
                lo,
                hi,
                int(args.score_grid_size),
                args,
            )
        )
        if corner_count_allowed(4, args.min_corners, args.max_corners, args.target_corners):
            radial_lo = np.asarray(
                radial_bounds_meta.get("expanded_lo_xz", lo), dtype=np.float64
            )
            radial_hi = np.asarray(
                radial_bounds_meta.get("expanded_hi_xz", hi), dtype=np.float64
            )
            all_candidates.append(
                {
                    "source": "radial_bounds_axis_aligned_envelope_4",
                    "poly_xz": np.asarray(
                        [
                            [radial_lo[0], radial_hi[1]],
                            [radial_lo[0], radial_lo[1]],
                            [radial_hi[0], radial_lo[1]],
                            [radial_hi[0], radial_hi[1]],
                        ],
                        dtype=np.float64,
                    ),
                }
            )
            all_candidates.append(
                {
                    "source": "vertical_wall_dominant_envelope_4",
                    "poly_xz": np.asarray(
                        [
                            [dominant_base_lo[0], dominant_base_hi[1]],
                            [dominant_base_lo[0], dominant_base_lo[1]],
                            [dominant_base_hi[0], dominant_base_lo[1]],
                            [dominant_base_hi[0], dominant_base_hi[1]],
                        ],
                        dtype=np.float64,
                    ),
                }
            )
        radial_candidates = radial_polygon_candidates(
            support.get("radial_mask"), lo, hi, args
        )
        radial_step_candidates_list = radial_step_candidates(
            support.get("radial_mask"),
            dominant_base_lo,
            dominant_base_hi,
            lo,
            hi,
            args,
        )
        all_candidates.extend(radial_candidates)
        all_candidates.extend(radial_step_candidates_list)
    checkpoint_meta = {}
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    for checkpoint in args.checkpoints:
        model, load_meta = build_roomformer(args.roomformer_dir, checkpoint, device)
        checkpoint_meta[str(checkpoint)] = load_meta
        for mode, image in density_images.items():
            sample = torch.as_tensor(image[None, :, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                outputs = model([sample])
            all_candidates.extend(
                extract_roomformer_candidates(
                    outputs,
                    [float(x) for x in args.corner_thresholds],
                    lo,
                    hi,
                    checkpoint.stem,
                    mode,
                    int(args.min_corners),
                    int(args.max_corners),
                    int(args.target_corners),
                )
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    roomformer_candidate_count = sum(
        str(candidate.get("source", "")).startswith("roomformer_")
        for candidate in all_candidates
    )
    if args.candidate_source_policy == "roomformer_only":
        unexpected_sources = sorted(
            {
                str(candidate.get("source", ""))
                for candidate in all_candidates
                if not str(candidate.get("source", "")).startswith("roomformer_")
            }
        )
        if unexpected_sources:
            raise RuntimeError(
                "roomformer_only admitted non-RoomFormer candidates: "
                + ", ".join(unexpected_sources)
            )
        if not all_candidates:
            raise RuntimeError("RoomFormer produced no valid variable-corner proposals")

    scored = expand_and_score_candidates(all_candidates, point_xz, support, lo, hi, args)
    if not scored:
        raise RuntimeError(
            "No valid variable-corner footprint candidates were produced "
            f"under candidate-source-policy={args.candidate_source_policy}"
        )
    scored, supported_topology_policy = select_supported_topology_candidate(scored, args)
    scored, selection_policy = select_opening_aware_candidate(scored, args)
    scored, stabilization_policy = stabilize_selected_topology(scored, point_xz, support, args)
    if args.candidate_source_policy == "roomformer_only":
        unexpected_scored_sources = sorted(
            {
                str(candidate.get("source", ""))
                for candidate in scored
                if not str(candidate.get("source", "")).startswith("roomformer_")
            }
        )
        if unexpected_scored_sources:
            raise RuntimeError(
                "roomformer_only produced non-RoomFormer scored candidates after selection: "
                + ", ".join(unexpected_scored_sources)
            )
    selection_policy = dict(selection_policy)
    selection_policy["supported_nonrectangular_topology_preservation"] = supported_topology_policy
    selection_policy["topology_preserving_manhattan_stabilization"] = stabilization_policy

    best_poly = np.asarray(scored[0]["polygon_xz"], dtype=np.float64)
    structure = make_structure(best_poly, points_room, world_from_room, room_from_world, scored[0]["source"])
    if (
        args.candidate_source_policy == "roomformer_only"
        and not str(structure.get("selected_source", "")).startswith("roomformer_")
    ):
        raise RuntimeError("roomformer_only final structure lost RoomFormer provenance")
    structure["candidate_source_policy"] = str(args.candidate_source_policy)
    structure["source_scene_glb"] = str(args.scene_glb)
    structure["point_count"] = int(points_world.shape[0])
    structure["room_basis_fit"] = basis_meta
    structure["yaw_alignment"] = yaw_meta
    structure["fit_score"] = scored[0]["metrics"]

    structure_path = args.out_dir / "structure_roomformer_da3_polygon.json"
    structure_path.write_text(json.dumps(structure, indent=2), encoding="utf-8")
    write_obj(args.out_dir / "roomformer_da3_polygon.obj", structure)

    summary = {
        "scene_glb": str(args.scene_glb),
        "out_dir": str(args.out_dir),
        "candidate_source_policy": str(args.candidate_source_policy),
        "roomformer_candidate_count": int(roomformer_candidate_count),
        "non_roomformer_candidate_count": int(
            len(all_candidates) - roomformer_candidate_count
        ),
        "basis": basis_meta,
        "yaw_alignment": yaw_meta,
        "bounds_lo_xz": lo.tolist(),
        "bounds_hi_xz": hi.tolist(),
        "radial_bounds": radial_bounds_meta,
        "radial_evidence": support.get("radial_meta", {}),
        "vertical_wall_evidence": support.get("wall_evidence_meta", {}),
        "dominant_base": dominant_base_meta,
        "radial_candidate_count": int(len(radial_candidates)),
        "radial_step_candidate_count": int(len(radial_step_candidates_list)),
        "point_count_before_radial_filter": point_count_before_radial_filter,
        "point_count_after_radial_filter": point_count_after_radial_filter,
        "checkpoint_meta": checkpoint_meta,
        "corner_policy": {
            "min_corners": int(args.min_corners),
            "max_corners": int(args.max_corners),
            "target_corners": int(args.target_corners),
        },
        "candidate_count_before_expansion": int(len(all_candidates)),
        "candidate_count_scored": int(len(scored)),
        "selection_policy": selection_policy,
        "best": scored[0],
        # The candidate set is already bounded by --max-expanded-candidates.
        # Keep every fully scored alternative so topology-preserving variants
        # such as a Manhattan projection remain auditable after selection.
        "top_candidates": scored,
        "outputs": {
            "structure_json": str(structure_path),
            "structure_obj": str(args.out_dir / "roomformer_da3_polygon.obj"),
            "leveled_preview": str(args.out_dir / "corner_search_leveled_summary.png"),
            "world_preview": str(args.out_dir / "corner_search_world_summary.png"),
            "contact_sheet": str(args.out_dir / "corner_candidates_contact_sheet.png"),
            "radial_mask": str(args.out_dir / "radial_free_space_score_mask.png")
            if support.get("radial_mask") is not None
            else None,
            "vertical_wall_evidence_mask": str(args.out_dir / "vertical_wall_evidence_mask.png"),
        },
    }
    (args.out_dir / "corner_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    best_density = density_images[args.density_modes[0]]
    plot_summary(
        args.out_dir / "corner_search_leveled_summary.png",
        points_room,
        best_density,
        support["occupancy"],
        scored,
        structure,
        int(args.plot_points),
    )
    plot_world_summary(args.out_dir / "corner_search_world_summary.png", points_world, structure, int(args.plot_points))
    make_contact_sheet(scored, args.out_dir / "corner_candidates_contact_sheet.png", int(args.top_k_preview))

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "structure_json": str(structure_path),
                "best_source": scored[0]["source"],
                "best_score": scored[0]["score"],
                "candidate_count": len(scored),
                "yaw_alignment": yaw_meta,
                "previews": summary["outputs"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
