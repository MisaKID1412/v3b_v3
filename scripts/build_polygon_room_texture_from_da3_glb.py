import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pygltflib import GLTF2


COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
ACCESSOR_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a polygonal Manhattan room shell and per-surface textures from a DA3 "
            "colored point-cloud GLB. This is the generalized non-cuboid successor of "
            "the six-face room02 texture builder."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scene-name", default="polygon_room")
    parser.add_argument("--up-axis", choices=["auto", "x", "y", "z"], default="auto")
    parser.add_argument("--grid-resolution", type=float, default=0.035)
    parser.add_argument("--texture-ppm", type=float, default=420.0)
    parser.add_argument("--max-texture-size", type=int, default=4096)
    parser.add_argument("--min-texture-size", type=int, default=384)
    parser.add_argument(
        "--floorplan-source",
        choices=["all", "floor", "ceiling", "floor_ceiling", "structural"],
        default="all",
        help=(
            "Which point subset estimates the 2D room footprint. structural is the "
            "PixCuboid-like polygon mode: prefer floor/ceiling structural support, "
            "score candidates by wall support, then snap the footprint to a Manhattan "
            "polygon. all is kept for old diagnostics."
        ),
    )
    parser.add_argument(
        "--orthogonalize-floorplan",
        action="store_true",
        help="Snap the extracted footprint contour to an orthogonal Manhattan polygon and remove short contour burrs.",
    )
    parser.add_argument("--structural-min-points", type=int, default=8000)
    parser.add_argument("--wall-support-tol", type=float, default=0.12)
    parser.add_argument("--wall-support-min-height-ratio", type=float, default=0.12)
    parser.add_argument("--wall-support-max-height-ratio", type=float, default=0.90)
    parser.add_argument("--surface-tol", type=float, default=0.085)
    parser.add_argument("--wall-tol", type=float, default=0.11)
    parser.add_argument("--floor-ceiling-band", type=float, default=0.16)
    parser.add_argument("--occupancy-close-m", type=float, default=0.10)
    parser.add_argument("--occupancy-open-m", type=float, default=0.035)
    parser.add_argument("--simplify-m", type=float, default=0.16)
    parser.add_argument("--min-edge-m", type=float, default=0.28)
    parser.add_argument("--inpaint-radius", type=float, default=4.0)
    parser.add_argument("--low-confidence-blend", type=float, default=0.22)
    parser.add_argument("--debug-sample-points", type=int, default=800000)
    return parser.parse_args()


def accessor_array(gltf, accessor_index):
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    blob = gltf.binary_blob()
    dtype = COMPONENT_DTYPES[accessor.componentType]
    components = ACCESSOR_COMPONENTS[accessor.type]
    item_size = np.dtype(dtype).itemsize * components
    offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    stride = view.byteStride or item_size
    if stride == item_size:
        arr = np.frombuffer(blob, dtype=dtype, count=accessor.count * components, offset=offset)
        arr = arr.reshape(accessor.count, components)
    else:
        raw = np.frombuffer(blob, dtype=np.uint8, count=stride * accessor.count, offset=offset)
        raw = raw.reshape(accessor.count, stride)
        arr = np.empty((accessor.count, components), dtype=dtype)
        for c in range(components):
            s = c * np.dtype(dtype).itemsize
            e = s + np.dtype(dtype).itemsize
            arr[:, c] = np.frombuffer(raw[:, s:e].copy().tobytes(), dtype=dtype)
    if accessor.normalized and np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.min < 0:
            arr = np.maximum(arr.astype(np.float32) / float(info.max), -1.0)
        else:
            arr = arr.astype(np.float32) / float(info.max)
    return np.asarray(arr)


def load_point_cloud_glb(path):
    gltf = GLTF2().load_binary(str(path))
    if not gltf.meshes:
        raise ValueError(f"No mesh/point primitive found in {path}")
    primitive = gltf.meshes[0].primitives[0]
    attrs = primitive.attributes
    pos_idx = attrs.POSITION
    color_idx = attrs.COLOR_0
    if pos_idx is None:
        raise ValueError("GLB primitive has no POSITION accessor")
    points = accessor_array(gltf, pos_idx).astype(np.float32)
    if color_idx is None:
        colors = np.full((points.shape[0], 4), 255, dtype=np.uint8)
    else:
        colors = accessor_array(gltf, color_idx)
        if colors.dtype != np.uint8:
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        if colors.shape[1] == 3:
            alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
            colors = np.concatenate([colors, alpha], axis=1)
    return points, colors[:, :3]


def choose_up_axis(points, requested):
    if requested != "auto":
        return {"x": 0, "y": 1, "z": 2}[requested]
    ranges = np.percentile(points, 99.5, axis=0) - np.percentile(points, 0.5, axis=0)
    return int(np.argmin(ranges))


def edge_peak(values, side):
    lo, hi = np.percentile(values, [0.2, 99.8])
    hist, edges = np.histogram(values, bins=240, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if side == "low":
        span = max(8, len(hist) // 5)
        idx = int(np.argmax(hist[:span]))
    else:
        span = max(8, len(hist) // 5)
        idx = len(hist) - span + int(np.argmax(hist[-span:]))
    return float(centers[idx])


def rotate2(points, theta):
    c, s = math.cos(theta), math.sin(theta)
    r = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ r.T


def build_occupancy(coords, resolution, pad=8):
    mn = coords.min(axis=0) - pad * resolution
    mx = coords.max(axis=0) + pad * resolution
    size = np.maximum(np.ceil((mx - mn) / resolution).astype(int) + 1, 4)
    pix = np.floor((coords - mn[None, :]) / resolution).astype(np.int32)
    pix[:, 0] = np.clip(pix[:, 0], 0, size[0] - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, size[1] - 1)
    mask = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    mask[pix[:, 1], pix[:, 0]] = 255
    return mask, mn, resolution


def morph_radius(meters, resolution, minimum=1):
    return max(minimum, int(round(meters / max(resolution, 1e-6))))


def fill_binary_holes(mask):
    h, w = mask.shape
    flood = mask.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return ((labels == keep).astype(np.uint8) * 255)


def estimate_manhattan_angle(coords, resolution):
    mask, _, _ = build_occupancy(coords, resolution, pad=6)
    close = morph_radius(0.10, resolution)
    kernel = np.ones((2 * close + 1, 2 * close + 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(mask, 40, 120)
    min_len = max(12, int(0.45 / resolution))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=28, minLineLength=min_len, maxLineGap=max(4, min_len // 3))
    if lines is None or len(lines) < 4:
        centered = coords - coords.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, centered.shape[0])
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        return -float(math.atan2(axis[1], axis[0]))
    acc = 0.0 + 0.0j
    for item in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, item)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        theta = math.atan2(dy, dx)
        acc += length * complex(math.cos(4.0 * theta), math.sin(4.0 * theta))
    if abs(acc) < 1e-6:
        return 0.0
    dominant = math.atan2(acc.imag, acc.real) / 4.0
    return -dominant


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def remove_duplicate_vertices(poly, eps=1e-5):
    out = []
    for p in poly:
        if not out or np.linalg.norm(np.asarray(p) - np.asarray(out[-1])) > eps:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(np.asarray(out[0]) - np.asarray(out[-1])) <= eps:
        out.pop()
    return np.asarray(out, dtype=np.float32)


def remove_short_and_collinear(poly, min_edge):
    poly = remove_duplicate_vertices(poly)
    changed = True
    while changed and len(poly) > 4:
        changed = False
        keep = []
        n = len(poly)
        for i in range(n):
            prev = poly[(i - 1) % n]
            cur = poly[i]
            nxt = poly[(i + 1) % n]
            if np.linalg.norm(nxt - cur) < min_edge:
                changed = True
                continue
            v1 = cur - prev
            v2 = nxt - cur
            if abs(np.cross(v1, v2)) < 1e-4 and np.dot(v1, v2) >= 0:
                changed = True
                continue
            keep.append(cur)
        poly = np.asarray(keep, dtype=np.float32)
    return remove_duplicate_vertices(poly)


def orthogonalize_polygon(poly, min_edge):
    poly = remove_duplicate_vertices(poly)
    if len(poly) < 4:
        return poly
    if polygon_area(poly) < 0:
        poly = poly[::-1].copy()
    lines = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        d = q - p
        length = float(np.linalg.norm(d))
        if length < min_edge * 0.35:
            continue
        if abs(d[0]) >= abs(d[1]):
            lines.append(["h", float((p[1] + q[1]) * 0.5), length])
        else:
            lines.append(["v", float((p[0] + q[0]) * 0.5), length])
    if len(lines) < 4:
        return poly
    merged = []
    for kind, val, weight in lines:
        if merged and merged[-1][0] == kind:
            old = merged[-1]
            total = old[2] + weight
            old[1] = (old[1] * old[2] + val * weight) / max(total, 1e-6)
            old[2] = total
        else:
            merged.append([kind, val, weight])
    if len(merged) > 1 and merged[0][0] == merged[-1][0]:
        first, last = merged[0], merged[-1]
        total = first[2] + last[2]
        first[1] = (first[1] * first[2] + last[1] * last[2]) / max(total, 1e-6)
        first[2] = total
        merged.pop()
    if len(merged) < 4 or any(merged[i][0] == merged[(i + 1) % len(merged)][0] for i in range(len(merged))):
        return remove_short_and_collinear(poly, min_edge)
    verts = []
    for i in range(len(merged)):
        prev = merged[(i - 1) % len(merged)]
        cur = merged[i]
        if prev[0] == "v" and cur[0] == "h":
            verts.append([prev[1], cur[1]])
        elif prev[0] == "h" and cur[0] == "v":
            verts.append([cur[1], prev[1]])
    out = np.asarray(verts, dtype=np.float32)
    out = remove_short_and_collinear(out, min_edge)
    if len(out) >= 4 and abs(polygon_area(out)) > 1e-4:
        if polygon_area(out) < 0:
            out = out[::-1].copy()
        return out
    return poly


def points_in_polygon(points, poly):
    points = np.asarray(points, dtype=np.float32)
    poly = np.asarray(poly, dtype=np.float32)
    if points.size == 0 or len(poly) < 3:
        return np.zeros(points.shape[0], dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    xj, yj = poly[-1, 0], poly[-1, 1]
    for xi, yi in poly:
        denom = float(yj - yi)
        if abs(denom) < 1e-12:
            denom = 1e-12 if denom >= 0.0 else -1e-12
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / denom + xi)
        inside ^= crosses
        xj, yj = xi, yi
    return inside


def polygon_edge_lengths(poly):
    if len(poly) < 2:
        return np.zeros(0, dtype=np.float32)
    return np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1).astype(np.float32)


def wall_support_score(poly, horiz, heights, floor_y, ceiling_y, args):
    room_h = max(float(ceiling_y - floor_y), 1e-6)
    rel_h = (heights - float(floor_y)) / room_h
    wall_band = (
        (rel_h >= float(args.wall_support_min_height_ratio))
        & (rel_h <= float(args.wall_support_max_height_ratio))
    )
    coords = horiz[wall_band]
    if coords.shape[0] == 0 or len(poly) < 3:
        return 0.0, 0
    if coords.shape[0] > args.debug_sample_points:
        idx = np.linspace(0, coords.shape[0] - 1, args.debug_sample_points).astype(np.int64)
        coords = coords[idx]
    total = 0
    perimeter = 0.0
    tol = max(float(args.wall_support_tol), 1e-6)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-6:
            continue
        e = edge / length
        rel = coords - a[None, :]
        along = rel @ e
        perp = np.abs(rel[:, 0] * e[1] - rel[:, 1] * e[0])
        total += int(np.count_nonzero((along >= -tol) & (along <= length + tol) & (perp <= tol)))
        perimeter += length
    density = total / max(perimeter * 600.0, 1.0)
    return float(np.clip(density, 0.0, 1.0)), total


def candidate_polygon_score(poly, horiz, heights, floor_y, ceiling_y, fc_mask, args):
    if len(poly) < 4:
        return -1e9, {}
    area = abs(polygon_area(poly))
    lengths = polygon_edge_lengths(poly)
    bbox = np.maximum(poly.max(axis=0) - poly.min(axis=0), 1e-6)
    fill = float(np.clip(area / float(bbox[0] * bbox[1]), 0.0, 1.0))
    fc_coords = horiz[fc_mask]
    if fc_coords.shape[0] > args.debug_sample_points:
        idx = np.linspace(0, fc_coords.shape[0] - 1, args.debug_sample_points).astype(np.int64)
        fc_coords = fc_coords[idx]
    inside_fc = float(points_in_polygon(fc_coords, poly).mean()) if fc_coords.shape[0] else 0.0
    support, support_count = wall_support_score(poly, horiz, heights, floor_y, ceiling_y, args)
    short_penalty = float(np.mean(lengths < max(float(args.min_edge_m), 1e-6))) if lengths.size else 1.0
    vertex_penalty = max(0.0, (len(poly) - 14) / 18.0)
    score = 2.4 * inside_fc + 1.4 * support + 0.35 * fill - 0.55 * short_penalty - 0.20 * vertex_penalty
    info = {
        "vertices": int(len(poly)),
        "area": float(area),
        "fill_ratio": fill,
        "floor_ceiling_inside_ratio": inside_fc,
        "wall_support_score": support,
        "wall_support_points": int(support_count),
        "short_edge_fraction": short_penalty,
        "score": float(score),
    }
    return float(score), info


def extract_structural_floorplan(horiz, heights, floor_y, ceiling_y, floor_band, ceiling_band, args):
    all_mask = np.ones(horiz.shape[0], dtype=bool)
    candidate_masks = [
        ("floor_ceiling", floor_band | ceiling_band),
        ("ceiling", ceiling_band),
        ("floor", floor_band),
        ("all", all_mask),
    ]
    fc_mask = floor_band | ceiling_band
    candidates = []
    for label, mask in candidate_masks:
        count = int(np.count_nonzero(mask))
        if count < int(args.structural_min_points):
            continue
        coords = horiz[mask]
        if coords.shape[0] > args.debug_sample_points * 2:
            idx = np.linspace(0, coords.shape[0] - 1, args.debug_sample_points * 2).astype(np.int64)
            coords = coords[idx]
        try:
            poly, occ_mask, occ_origin = extract_floorplan_polygon(
                coords,
                args.grid_resolution,
                args.occupancy_close_m,
                args.occupancy_open_m,
                args.simplify_m,
                args.min_edge_m,
            )
        except Exception:
            continue
        poly = orthogonalize_polygon(poly, args.min_edge_m)
        score, info = candidate_polygon_score(poly, horiz, heights, floor_y, ceiling_y, fc_mask, args)
        info["source"] = label
        info["source_points"] = count
        candidates.append((score, label, poly, occ_mask, occ_origin, info))
    if not candidates:
        coords = horiz
        poly, occ_mask, occ_origin = extract_floorplan_polygon(
            coords,
            args.grid_resolution,
            args.occupancy_close_m,
            args.occupancy_open_m,
            args.simplify_m,
            args.min_edge_m,
        )
        poly = orthogonalize_polygon(poly, args.min_edge_m)
        return poly, occ_mask, occ_origin, "all_fallback", []
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, label, poly, occ_mask, occ_origin, info = candidates[0]
    return poly, occ_mask, occ_origin, label, [item[-1] for item in candidates]


def extract_floorplan_polygon(coords, resolution, close_m, open_m, simplify_m, min_edge_m):
    mask, origin, res = build_occupancy(coords, resolution, pad=10)
    if open_m > 0:
        r = morph_radius(open_m, res, minimum=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    r = morph_radius(close_m, res, minimum=1)
    mask = cv2.dilate(mask, np.ones((2 * r + 1, 2 * r + 1), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    mask = fill_binary_holes(mask)
    mask = largest_component(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        return np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]], dtype=np.float32), mask, origin
    contour = max(contours, key=cv2.contourArea)
    eps = max(2.0, simplify_m / max(res, 1e-6))
    approx = cv2.approxPolyDP(contour, eps, True)[:, 0, :].astype(np.float32)
    poly = np.empty((approx.shape[0], 2), dtype=np.float32)
    poly[:, 0] = origin[0] + approx[:, 0] * res
    poly[:, 1] = origin[1] + approx[:, 1] * res
    # The occupancy contour is already grid-aligned after Manhattan rotation. A
    # previous version merged consecutive same-orientation segments and could
    # collapse a large concavity into a tiny step. Here we preserve the contour
    # topology and only remove tiny duplicate/collinear artifacts.
    poly = remove_short_and_collinear(poly, min_edge_m)
    if len(poly) < 4 or abs(polygon_area(poly)) < 1e-4:
        ys, xs = np.where(mask > 0)
        mn = origin + np.array([xs.min(), ys.min()], dtype=np.float32) * res
        mx = origin + np.array([xs.max(), ys.max()], dtype=np.float32) * res
        poly = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]], dtype=np.float32)
    if polygon_area(poly) < 0:
        poly = poly[::-1].copy()
    return poly, mask, origin


def triangulate_earclip(poly):
    poly = np.asarray(poly, dtype=np.float64)
    if polygon_area(poly.astype(np.float32)) < 0:
        poly = poly[::-1].copy()
    indices = list(range(len(poly)))
    tris = []
    guard = 0
    while len(indices) > 3 and guard < len(poly) * len(poly):
        guard += 1
        ear_found = False
        for j in range(len(indices)):
            i0, i1, i2 = indices[(j - 1) % len(indices)], indices[j], indices[(j + 1) % len(indices)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if np.cross(b - a, c - b) <= 1e-9:
                continue
            tri = np.array([a, b, c])
            inside = False
            for k in indices:
                if k in (i0, i1, i2):
                    continue
                if point_in_triangle(poly[k], tri):
                    inside = True
                    break
            if inside:
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


def point_in_triangle(p, tri):
    a, b, c = tri
    v0, v1, v2 = c - a, b - a, p - a
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -1e-8 and v >= -1e-8 and (u + v) <= 1.0 + 1e-8


def texture_shape(span_u, span_v, ppm, min_size, max_size):
    w = max(min_size, int(math.ceil(span_u * ppm)))
    h = max(min_size, int(math.ceil(span_v * ppm)))
    scale = min(1.0, max_size / max(w, h))
    w = max(8, int(round(w * scale)))
    h = max(8, int(round(h * scale)))
    return w, h


def polygon_mask_for_texture(poly, bounds, shape):
    w, h = shape
    mn = bounds[0]
    span = np.maximum(bounds[1] - bounds[0], 1e-6)
    pts = np.empty_like(poly, dtype=np.int32)
    pts[:, 0] = np.clip(np.round((poly[:, 0] - mn[0]) / span[0] * (w - 1)), 0, w - 1).astype(np.int32)
    pts[:, 1] = np.clip(np.round((poly[:, 1] - mn[1]) / span[1] * (h - 1)), 0, h - 1).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def accumulate_points_to_texture(tex_x, tex_y, weights, rgb, w, h):
    valid = (
        (tex_x >= 0)
        & (tex_x < w)
        & (tex_y >= 0)
        & (tex_y < h)
        & np.isfinite(weights)
        & (weights > 0)
    )
    n = w * h
    if not np.any(valid):
        return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.int32)
    flat = tex_y[valid].astype(np.int64) * w + tex_x[valid].astype(np.int64)
    ww = weights[valid].astype(np.float64)
    weight_sum = np.bincount(flat, weights=ww, minlength=n).astype(np.float32)
    count = np.bincount(flat, minlength=n).astype(np.int32)
    sums = []
    for c in range(3):
        sums.append(np.bincount(flat, weights=ww * rgb[valid, c].astype(np.float64), minlength=n).astype(np.float32))
    sums = np.stack(sums, axis=1)
    out = np.zeros((n, 3), dtype=np.float32)
    nz = weight_sum > 1e-8
    out[nz] = sums[nz] / weight_sum[nz, None]
    out = np.clip(out.reshape(h, w, 3), 0, 255).astype(np.uint8)
    return out, weight_sum.reshape(h, w), count.reshape(h, w)


def complete_texture(raw, weight_sum, count, valid_mask, args):
    observed = (weight_sum > 1e-6) & (valid_mask > 0)
    out = raw.copy()
    if np.count_nonzero(observed) == 0:
        out[valid_mask > 0] = np.array([180, 180, 180], dtype=np.uint8)
        return out, np.zeros(valid_mask.shape, dtype=np.float32), observed
    med = np.median(raw[observed], axis=0).astype(np.uint8)
    init = raw.copy()
    init[(valid_mask > 0) & ~observed] = med
    hole = ((valid_mask > 0) & ~observed).astype(np.uint8) * 255
    if np.count_nonzero(hole) > 0:
        inpainted = cv2.inpaint(init, hole, args.inpaint_radius, cv2.INPAINT_TELEA)
    else:
        inpainted = init
    vals = weight_sum[observed]
    p95 = max(float(np.percentile(vals, 95)), 1e-6)
    conf = np.clip(weight_sum / p95, 0.0, 1.0).astype(np.float32)
    low_alpha = np.clip((0.48 - conf) / 0.48, 0.0, 1.0) * float(args.low_confidence_blend)
    low_alpha[(valid_mask == 0) | ~observed] = 0.0
    blended = (raw.astype(np.float32) * (1.0 - low_alpha[..., None]) + inpainted.astype(np.float32) * low_alpha[..., None])
    out = np.clip(blended, 0, 255).astype(np.uint8)
    out[(valid_mask > 0) & ~observed] = inpainted[(valid_mask > 0) & ~observed]
    out[valid_mask == 0] = 0
    return out, conf, observed


def save_debug_map(path, arr):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0 or finite.max() <= finite.min():
            img = np.zeros(arr.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(finite, [1, 99])
            img = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    else:
        img = arr
    Image.fromarray(img).save(path)


def save_texture(path, image):
    Image.fromarray(image).save(path)


def build_floor_or_ceiling(face, points_h, heights, colors, plane_y, poly, bounds, args):
    span = bounds[1] - bounds[0]
    w, h = texture_shape(span[0], span[1], args.texture_ppm, args.min_texture_size, args.max_texture_size)
    valid_mask = polygon_mask_for_texture(poly, bounds, (w, h))
    near = np.abs(heights - plane_y) <= args.surface_tol
    q = points_h[near]
    rgb = colors[near]
    dist = np.abs(heights[near] - plane_y)
    tex_x = np.floor((q[:, 0] - bounds[0, 0]) / max(span[0], 1e-6) * w).astype(np.int32)
    tex_y = np.floor((q[:, 1] - bounds[0, 1]) / max(span[1], 1e-6) * h).astype(np.int32)
    weights = np.exp(-((dist / max(args.surface_tol, 1e-6)) ** 2)).astype(np.float32)
    raw, weight_sum, count = accumulate_points_to_texture(tex_x, tex_y, weights, rgb, w, h)
    raw[valid_mask == 0] = 0
    completed, conf, observed = complete_texture(raw, weight_sum, count, valid_mask, args)
    meta = {
        "face": face,
        "type": "floor_ceiling_polygon",
        "texture_size": [w, h],
        "surface_point_count": int(np.count_nonzero(near)),
        "observed_texels": int(np.count_nonzero(observed)),
        "valid_texels": int(np.count_nonzero(valid_mask)),
    }
    return completed, raw, valid_mask, conf, count, weight_sum, meta


def build_wall(face, edge_a, edge_b, points_h, heights, colors, floor_y, ceiling_y, args):
    edge = edge_b - edge_a
    length = float(np.linalg.norm(edge))
    if length < 1e-6:
        raise ValueError(f"Degenerate wall edge for {face}")
    e = edge / length
    rel = points_h - edge_a[None, :]
    along = rel @ e
    perp = np.abs(rel[:, 0] * e[1] - rel[:, 1] * e[0])
    height = heights - floor_y
    room_h = max(float(ceiling_y - floor_y), 1e-6)
    near = (
        (along >= -args.wall_tol)
        & (along <= length + args.wall_tol)
        & (perp <= args.wall_tol)
        & (height >= -args.surface_tol)
        & (height <= room_h + args.surface_tol)
    )
    if np.count_nonzero(near) < 800:
        loose = args.wall_tol * 1.8
        near = (
            (along >= -loose)
            & (along <= length + loose)
            & (perp <= loose)
            & (height >= -args.surface_tol)
            & (height <= room_h + args.surface_tol)
        )
    w, h = texture_shape(length, room_h, args.texture_ppm, args.min_texture_size, args.max_texture_size)
    valid_mask = np.full((h, w), 255, dtype=np.uint8)
    tex_x = np.floor(np.clip(along[near], 0.0, length) / length * w).astype(np.int32)
    tex_y = np.floor((1.0 - np.clip(height[near], 0.0, room_h) / room_h) * h).astype(np.int32)
    weights = np.exp(-((perp[near] / max(args.wall_tol, 1e-6)) ** 2)).astype(np.float32)
    raw, weight_sum, count = accumulate_points_to_texture(tex_x, tex_y, weights, colors[near], w, h)
    completed, conf, observed = complete_texture(raw, weight_sum, count, valid_mask, args)
    meta = {
        "face": face,
        "type": "wall_edge",
        "edge_start": edge_a.tolist(),
        "edge_end": edge_b.tolist(),
        "length": length,
        "texture_size": [w, h],
        "surface_point_count": int(np.count_nonzero(near)),
        "observed_texels": int(np.count_nonzero(observed)),
        "valid_texels": int(np.count_nonzero(valid_mask)),
    }
    return completed, raw, valid_mask, conf, count, weight_sum, meta


def write_obj(out_dir, scene_name, poly, floor_y, ceiling_y, bounds, metas):
    obj_path = out_dir / f"{scene_name}.obj"
    mtl_path = out_dir / f"{scene_name}.mtl"
    room_h = float(ceiling_y - floor_y)
    mn = bounds[0]
    verts = []
    uvs = []
    faces = []

    def add_vertex(pos, uv):
        verts.append(pos)
        uvs.append(uv)
        return len(verts)

    def add_face(indices, mat):
        faces.append((mat, indices))

    floor_ids = []
    ceil_ids = []
    span = np.maximum(bounds[1] - bounds[0], 1e-6)
    for p in poly:
        x = float(p[0] - mn[0])
        z = float(p[1] - mn[1])
        u = float((p[0] - mn[0]) / span[0])
        v_img = float((p[1] - mn[1]) / span[1])
        # Floor/ceiling atlases are written in image row coordinates, where
        # v=0 is the top row. OBJ/Unity texture sampling uses v=0 at the
        # bottom, so convert only these polygon-surface UVs here. Wall
        # atlases already write rows as 1-height/room_h and should not flip.
        uv = (u, 1.0 - v_img)
        floor_ids.append(add_vertex((x, 0.0, z), uv))
        ceil_ids.append(add_vertex((x, room_h, z), uv))
    for tri in triangulate_earclip(poly):
        add_face([floor_ids[i] for i in tri], "floor")
        add_face([floor_ids[i] for i in tri[::-1]], "floor")
        add_face([ceil_ids[i] for i in tri[::-1]], "ceiling")
        add_face([ceil_ids[i] for i in tri], "ceiling")

    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        face = f"wall_{i:02d}"
        ax, az = float(a[0] - mn[0]), float(a[1] - mn[1])
        bx, bz = float(b[0] - mn[0]), float(b[1] - mn[1])
        ids = [
            add_vertex((ax, 0.0, az), (0.0, 0.0)),
            add_vertex((bx, 0.0, bz), (1.0, 0.0)),
            add_vertex((bx, room_h, bz), (1.0, 1.0)),
            add_vertex((ax, room_h, az), (0.0, 1.0)),
        ]
        add_face([ids[0], ids[1], ids[2], ids[3]], face)
        add_face([ids[3], ids[2], ids[1], ids[0]], face)

    with open(mtl_path, "w", encoding="utf-8") as f:
        for meta in metas:
            face = meta["face"]
            f.write(f"newmtl {face}\n")
            f.write("Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\n")
            f.write(f"map_Kd textures/{face}.png\n\n")
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        current = None
        for mat, inds in faces:
            if mat != current:
                f.write(f"usemtl {mat}\n")
                current = mat
            items = " ".join(f"{idx}/{idx}" for idx in inds)
            f.write(f"f {items}\n")
    return obj_path, mtl_path


def draw_floorplan_debug(path, mask, origin, resolution, poly):
    img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    pts = np.empty((len(poly), 2), dtype=np.int32)
    pts[:, 0] = np.round((poly[:, 0] - origin[0]) / resolution).astype(np.int32)
    pts[:, 1] = np.round((poly[:, 1] - origin[1]) / resolution).astype(np.int32)
    cv2.polylines(img, [pts], True, (255, 40, 40), 2, cv2.LINE_AA)
    Image.fromarray(img).save(path)


def build_contact_sheet(out_dir, metas):
    thumbs = []
    for meta in metas:
        face = meta["face"]
        im = Image.open(out_dir / "textures" / f"{face}.png").convert("RGB")
        im.thumbnail((420, 260))
        tile = Image.new("RGB", (440, 300), (24, 24, 24))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 8), face, fill=(245, 245, 245))
        tile.paste(im, ((440 - im.width) // 2, 34))
        thumbs.append(tile)
    if not thumbs:
        return
    cols = 2 if len(thumbs) <= 8 else 3
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 440, rows * 300), (18, 18, 18))
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i % cols) * 440, (i // cols) * 300))
    sheet.save(out_dir / "structured_texture_preview.jpg", quality=92)


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    glb = dataset_dir / "scene.glb"
    if not glb.exists():
        raise FileNotFoundError(glb)
    out_dir = args.out_dir
    tex_dir = out_dir / "textures"
    dbg_dir = out_dir / "debug"
    tex_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)

    points, colors = load_point_cloud_glb(glb)
    up_axis = choose_up_axis(points, args.up_axis)
    h_axes = [i for i in range(3) if i != up_axis]
    heights0 = points[:, up_axis].astype(np.float32)
    floor_y = edge_peak(heights0, "low")
    ceiling_y = edge_peak(heights0, "high")
    if ceiling_y < floor_y:
        floor_y, ceiling_y = ceiling_y, floor_y

    horiz0 = points[:, h_axes].astype(np.float32)
    floor_band = np.abs(heights0 - floor_y) <= args.floor_ceiling_band
    ceiling_band = np.abs(heights0 - ceiling_y) <= args.floor_ceiling_band
    fc_for_angle = floor_band | ceiling_band
    if np.count_nonzero(fc_for_angle) < 10000:
        fc_for_angle = np.ones(points.shape[0], dtype=bool)
    if np.count_nonzero(fc_for_angle) > args.debug_sample_points:
        idx = np.linspace(0, np.count_nonzero(fc_for_angle) - 1, args.debug_sample_points).astype(np.int64)
        fc_coords = horiz0[fc_for_angle][idx]
    else:
        fc_coords = horiz0[fc_for_angle]
    theta = estimate_manhattan_angle(fc_coords, args.grid_resolution)
    horiz = rotate2(horiz0, theta).astype(np.float32)
    layout_candidates = []
    selected_floorplan_source = args.floorplan_source
    if args.floorplan_source == "structural":
        poly, occ_mask, occ_origin, selected_floorplan_source, layout_candidates = extract_structural_floorplan(
            horiz,
            heights0,
            floor_y,
            ceiling_y,
            floor_band,
            ceiling_band,
            args,
        )
    else:
        if args.floorplan_source == "floor":
            footprint_sel = floor_band
        elif args.floorplan_source == "ceiling":
            footprint_sel = ceiling_band
        elif args.floorplan_source == "floor_ceiling":
            footprint_sel = floor_band | ceiling_band
        else:
            footprint_sel = np.ones(points.shape[0], dtype=bool)
        if np.count_nonzero(footprint_sel) > args.debug_sample_points * 2:
            fp_idx = np.linspace(0, np.count_nonzero(footprint_sel) - 1, args.debug_sample_points * 2).astype(np.int64)
            footprint_coords = horiz[footprint_sel][fp_idx]
        else:
            footprint_coords = horiz[footprint_sel]

        poly, occ_mask, occ_origin = extract_floorplan_polygon(
            footprint_coords,
            args.grid_resolution,
            args.occupancy_close_m,
            args.occupancy_open_m,
            args.simplify_m,
            args.min_edge_m,
        )
        if args.orthogonalize_floorplan:
            poly = orthogonalize_polygon(poly, args.min_edge_m)
    bounds = np.stack([poly.min(axis=0), poly.max(axis=0)], axis=0).astype(np.float32)
    draw_floorplan_debug(dbg_dir / "floorplan_polygon_debug.png", occ_mask, occ_origin, args.grid_resolution, poly)
    if layout_candidates:
        with open(dbg_dir / "floorplan_structural_candidates.json", "w", encoding="utf-8") as f:
            json.dump(layout_candidates, f, indent=2, ensure_ascii=False)

    metas = []
    for face, plane_y in [("floor", floor_y), ("ceiling", ceiling_y)]:
        completed, raw, valid_mask, conf, count, weight_sum, meta = build_floor_or_ceiling(
            face, horiz, heights0, colors, plane_y, poly, bounds, args
        )
        save_texture(tex_dir / f"{face}.png", completed)
        save_texture(dbg_dir / f"{face}_raw_projected.png", raw)
        save_debug_map(dbg_dir / f"{face}_valid_mask.png", valid_mask)
        save_debug_map(dbg_dir / f"{face}_confidence.png", conf)
        np.save(dbg_dir / f"{face}_valid_count.npy", count)
        np.save(dbg_dir / f"{face}_weight_sum.npy", weight_sum)
        metas.append(meta)

    for i in range(len(poly)):
        face = f"wall_{i:02d}"
        completed, raw, valid_mask, conf, count, weight_sum, meta = build_wall(
            face, poly[i], poly[(i + 1) % len(poly)], horiz, heights0, colors, floor_y, ceiling_y, args
        )
        save_texture(tex_dir / f"{face}.png", completed)
        save_texture(dbg_dir / f"{face}_raw_projected.png", raw)
        save_debug_map(dbg_dir / f"{face}_confidence.png", conf)
        np.save(dbg_dir / f"{face}_valid_count.npy", count)
        np.save(dbg_dir / f"{face}_weight_sum.npy", weight_sum)
        metas.append(meta)

    obj_path, mtl_path = write_obj(out_dir, args.scene_name, poly, floor_y, ceiling_y, bounds, metas)
    build_contact_sheet(out_dir, metas)
    manifest = {
        "method": "polygonal_manhattan_room_from_da3_glb_point_cloud_v1",
        "dataset_dir": str(dataset_dir),
        "scene_glb": str(glb),
        "point_count": int(points.shape[0]),
        "color_source": "glb_COLOR_0",
        "up_axis": int(up_axis),
        "horizontal_axes": h_axes,
        "manhattan_rotation_theta_rad": float(theta),
        "floor_y": float(floor_y),
        "ceiling_y": float(ceiling_y),
        "height": float(ceiling_y - floor_y),
        "floorplan_polygon_uv": poly.tolist(),
        "floorplan_source": args.floorplan_source,
        "selected_floorplan_source": selected_floorplan_source,
        "orthogonalized_floorplan": bool(args.orthogonalize_floorplan or args.floorplan_source == "structural"),
        "floorplan_structural_candidates": layout_candidates,
        "bounds_uv": bounds.tolist(),
        "faces": metas,
        "outputs": {
            "obj": str(obj_path),
            "mtl": str(mtl_path),
            "preview": str(out_dir / "structured_texture_preview.jpg"),
            "floorplan_debug": str(dbg_dir / "floorplan_polygon_debug.png"),
        },
        "limitations": [
            "This dataset export contains a fused DA3 colored point cloud but no per-frame camera intrinsics/extrinsics/depth arrays.",
            "The first pass therefore uses point-cloud-to-surface observation instead of full image-space v53 projection.",
            "COLMAP/pycolmap poses can be attached later without changing the polygonal surface abstraction.",
        ],
    }
    for manifest_name in ("manifest.json", "metadata.json"):
        with open(out_dir / manifest_name, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
