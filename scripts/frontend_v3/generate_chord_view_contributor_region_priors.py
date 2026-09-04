#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parents[1]
PIX_DIR = PROJECT_DIR / "pipeline" / "pixcuboid" / "PixCuboid-main"
if str(PIX_DIR) not in sys.path:
    sys.path.insert(0, str(PIX_DIR))
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import build_polygon_photo_source_from_colmap as proj  # noqa: E402
import generate_material_priors as gmp  # noqa: E402
import generate_multi_material_priors as mm  # noqa: E402


PBR_ALIASES = {
    "basecolor": ["basecolor", "base_color", "albedo", "diffuse", "color"],
    "normal": ["normal", "normal_map", "bump"],
    "roughness": ["roughness", "rough"],
    "metallic": ["metallic", "metalness", "metal"],
    "irradiance": ["irradiance", "approxIrr", "approx_irr"],
    "rou_met": ["rou_met", "roughness_metallic", "roughnessMetallic"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select strict high-weight atlas regions, trace their actual source-view "
            "contributors, create masked original-image Chord inputs, then score Chord "
            "outputs against the atlas region before composing material territories."
        )
    )
    parser.add_argument("--stage", choices=["prepare", "compose"], required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--polygon-source-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--colmap-model-dir", type=Path, required=True)
    parser.add_argument("--da3-dir", type=Path, required=True)
    parser.add_argument("--object-mask-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--faces", nargs="*", default=None)

    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-stride", type=int, default=64)
    parser.add_argument("--floor-min-source-fraction", type=float, default=0.34)
    parser.add_argument("--ceiling-min-source-fraction", type=float, default=0.42)
    parser.add_argument("--wall-min-source-fraction", type=float, default=0.42)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--max-priors-per-face", type=int, default=8)
    parser.add_argument("--min-priors-per-face", type=int, default=1)
    parser.add_argument("--candidate-nms-iou", type=float, default=0.24)
    parser.add_argument("--candidate-min-center-frac", type=float, default=0.14)
    parser.add_argument("--cluster-lab-delta", type=float, default=0.55)
    parser.add_argument("--cluster-texture-delta", type=float, default=0.16)
    parser.add_argument(
        "--material-cluster-discovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Discover weighted material regions before selecting Chord inputs.",
    )
    parser.add_argument("--material-cluster-components", type=int, default=12)
    parser.add_argument("--material-cluster-max-samples", type=int, default=100000)
    parser.add_argument("--material-cluster-min-fraction", type=float, default=0.05)
    parser.add_argument("--single-material-faces", nargs="*", default=None, help="faces forced to one material")
    parser.add_argument("--face-max-materials", nargs="*", default=None, help="per-face material-count caps, e.g. wall_00=2")
    parser.add_argument("--material-cluster-purity", type=float, default=0.86)
    parser.add_argument("--material-cluster-min-region-size", type=int, default=96)
    parser.add_argument(
        "--material-cluster-exemplars",
        type=int,
        default=3,
        help=(
            "Keep multiple spatial observation regions for each discovered material. "
            "They share one material identity and compete only after Chord inference."
        ),
    )
    parser.add_argument(
        "--discover-persistent-wall-bands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Detect narrow horizontally persistent wall materials before contributor tracing. "
            "The detected atlas interval is only a support prior; CHORD still receives a "
            "rectified crop traced to an original camera image."
        ),
    )
    parser.add_argument("--wall-band-min-height-frac", type=float, default=0.018)
    parser.add_argument("--wall-band-max-height-frac", type=float, default=0.12)
    parser.add_argument("--wall-band-context-frac", type=float, default=0.075)
    parser.add_argument("--wall-band-min-center-frac", type=float, default=0.20)
    parser.add_argument("--wall-band-max-center-frac", type=float, default=0.80)
    parser.add_argument("--wall-band-min-score", type=float, default=1.10)
    parser.add_argument("--wall-band-min-tangent-coverage", type=float, default=0.48)
    parser.add_argument("--wall-band-max-count", type=int, default=2)
    parser.add_argument("--wall-band-edge-merge-frac", type=float, default=0.016)
    parser.add_argument("--wall-band-min-traceable-views", type=int, default=2)
    parser.add_argument(
        "--wall-band-reserve-material-slot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reserve room under per-face material caps for a validated wall-band candidate.",
    )
    parser.add_argument("--blend-distance-frac", type=float, default=0.34)
    parser.add_argument("--blend-smooth-frac", type=float, default=0.018)
    parser.add_argument("--raw-compat-sigma", type=float, default=0.92)
    parser.add_argument("--raw-compat-strength", type=float, default=0.78)
    parser.add_argument("--support-dilate-frac", type=float, default=0.018)
    parser.add_argument("--strict-ma-support", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-support-min-reliability", type=float, default=0.04)
    parser.add_argument("--strict-support-clean-thresh", type=float, default=0.58)
    parser.add_argument("--strict-support-object-risk-thresh", type=float, default=0.05)
    parser.add_argument("--strict-support-boundary-trust-thresh", type=float, default=0.55)

    parser.add_argument("--keep-valid-views", type=int, default=3)
    parser.add_argument("--candidate-valid-views", type=int, default=2)
    parser.add_argument("--min-valid-views", type=int, default=2)
    parser.add_argument("--keep-clean-thresh", type=float, default=0.62)
    parser.add_argument("--source-clean-thresh", type=float, default=0.58)
    parser.add_argument("--keep-object-risk-thresh", type=float, default=0.05)
    parser.add_argument("--source-object-risk-thresh", type=float, default=0.05)
    parser.add_argument("--mask-boundary-keep-thresh", type=float, default=0.55)
    parser.add_argument("--footprint-keep-min", type=float, default=0.12)

    parser.add_argument("--views-per-face", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=180000)
    parser.add_argument("--min-view-cos", type=float, default=0.08)
    parser.add_argument("--depth-abs-tol", type=float, default=0.045)
    parser.add_argument("--depth-rel-tol", type=float, default=0.035)
    parser.add_argument("--distance-weight-scale", type=float, default=1.15)
    parser.add_argument("--distance-weight-power", type=float, default=1.15)
    parser.add_argument("--surface-distance-tol", type=float, default=0.055)
    parser.add_argument("--surface-distance-power", type=float, default=1.0)
    parser.add_argument("--surface-distance-hard-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--object-mask-dilate-px", type=int, default=8)
    parser.add_argument("--object-risk-dilate-px", type=int, default=14)
    parser.add_argument("--object-risk-blur-px", type=int, default=9)
    parser.add_argument("--object-risk-hard-thresh", type=float, default=0.05)
    parser.add_argument("--mask-boundary-safe-px", type=float, default=18.0)
    parser.add_argument("--mask-boundary-power", type=float, default=1.15)
    parser.add_argument("--min-mask-boundary-trust", type=float, default=0.55)
    parser.add_argument("--footprint-min-area", type=float, default=0.32)
    parser.add_argument("--footprint-power", type=float, default=0.85)
    parser.add_argument("--zbuffer-stride", type=int, default=5)
    parser.add_argument("--color-std-clean-tol", type=float, default=0.18)
    parser.add_argument("--valid-ratio-penalty", type=float, default=0.45)

    parser.add_argument("--max-region-sample-texels", type=int, default=22000)
    parser.add_argument("--max-view-candidates", type=int, default=3)
    parser.add_argument("--max-source-views-eval", type=int, default=14)
    parser.add_argument("--min-view-weight-frac", type=float, default=0.035)
    parser.add_argument("--min-view-mask-pixels", type=int, default=1800)
    parser.add_argument("--view-crop-margin-px", type=int, default=34)
    parser.add_argument("--min-pure-crop-size", type=int, default=96)
    parser.add_argument("--min-pure-crop-mask-frac", type=float, default=0.96)
    parser.add_argument("--pure-crop-stride-frac", type=float, default=0.16)
    parser.add_argument(
        "--chord-input-mode",
        choices=["view_crop", "atlas_rectified"],
        default="atlas_rectified",
        help=(
            "atlas_rectified samples the original contributing view back into "
            "the atlas/face UV tile before Chord, so floor and wall patterns are "
            "axis-aligned in material space. view_crop keeps the older image crop path."
        ),
    )
    parser.add_argument("--min-rectified-valid-frac", type=float, default=0.42)
    parser.add_argument(
        "--rectified-inner-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Crop each rectified view to a clean interior square before Chord. "
            "The crop is selected only from valid source pixels and avoids mask boundaries."
        ),
    )
    parser.add_argument("--rectified-inner-min-size", type=int, default=128)
    parser.add_argument("--rectified-inner-max-side-frac", type=float, default=0.68)
    parser.add_argument("--rectified-inner-min-valid-frac", type=float, default=0.995)
    parser.add_argument("--rectified-inner-safe-border-px", type=int, default=10)
    parser.add_argument("--rectified-inner-stride-frac", type=float, default=0.08)
    parser.add_argument("--chord-input-size", type=int, default=512)
    parser.add_argument("--include-atlas-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chord-output-dir", type=Path, default=None)
    parser.add_argument("--basecolor-key", default="basecolor")
    parser.add_argument("--pbr-keys", default="basecolor,normal,roughness,metallic")
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_gray(path: Path, shape: tuple[int, int] | None = None, default: float = 0.0) -> np.ndarray:
    if not path.exists():
        if shape is None:
            raise FileNotFoundError(path)
        return np.full(shape, default, dtype=np.float32)
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if shape is not None and arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return arr.astype(np.float32)


def load_npy(path: Path, shape: tuple[int, int], default: float) -> np.ndarray:
    if not path.exists():
        return np.full(shape, default, dtype=np.float32)
    arr = np.load(path).astype(np.float32)
    if arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    arr[~np.isfinite(arr)] = default
    return arr


def pbr_keys(args: argparse.Namespace) -> list[str]:
    keys = [item.strip() for item in args.pbr_keys.split(",") if item.strip()]
    if "basecolor" not in [key.lower() for key in keys]:
        keys.insert(0, "basecolor")
    return keys


def channel_candidates(key: str) -> list[str]:
    names = [key]
    names.extend(PBR_ALIASES.get(key.lower(), []))
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def load_chord_map(chord_output_dir: Path, stem: str, key: str, shape_hw: tuple[int, int] | None = None) -> np.ndarray | None:
    candidates: list[Path] = []
    for name in channel_candidates(key):
        candidates.extend(
            [
                chord_output_dir / stem / f"{name}.png",
                chord_output_dir / stem / f"{name}.jpg",
                chord_output_dir / f"{stem}_{name}.png",
                chord_output_dir / f"{stem}_{name}.jpg",
            ]
        )
    for path in candidates:
        if path.exists():
            image = load_rgb(path)
            if shape_hw is not None and image.shape[:2] != shape_hw:
                image = cv2.resize(image, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_CUBIC)
            return image.astype(np.float32)
    return None


def default_channel_tile(key: str, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    key = key.lower()
    if key == "normal":
        color = np.array([0.5, 0.5, 1.0], dtype=np.float32)
    elif key == "roughness":
        color = np.array([0.65, 0.65, 0.65], dtype=np.float32)
    elif key == "metallic":
        color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    return np.ones((h, w, 3), dtype=np.float32) * color.reshape(1, 1, 3)


def manifest_and_faces(args: argparse.Namespace) -> tuple[dict, dict, list[str]]:
    manifest_path = args.polygon_source_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = args.polygon_source_dir / "metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metas = proj.face_meta_map(manifest)
    if args.faces:
        faces = args.faces
    else:
        faces = proj.face_names(manifest, None)
    return manifest, metas, faces


def source_image_path(source_dir: Path, face: str) -> Path:
    raw = source_dir / "debug" / f"{face}_raw_projected.png"
    if raw.exists():
        return raw
    return source_dir / "textures" / f"{face}.png"


def face_min_source_fraction(args: argparse.Namespace, face: str) -> float:
    if face == "floor":
        return args.floor_min_source_fraction
    if face == "ceiling":
        return args.ceiling_min_source_fraction
    return args.wall_min_source_fraction


def load_strict_masks(args: argparse.Namespace, face: str, shape: tuple[int, int]) -> dict:
    debug = args.source_dir / "debug"
    final_keep = load_gray(debug / f"{face}_final_keep_mask.png", shape, 0.0) > 0.5
    valid_count = load_npy(debug / f"{face}_valid_count.npy", shape, 0.0)
    candidate_count = load_npy(debug / f"{face}_candidate_count.npy", shape, 0.0)
    clean_score = load_npy(debug / f"{face}_clean_score.npy", shape, 0.0)
    object_risk = load_npy(debug / f"{face}_object_risk.npy", shape, 1.0)
    weight_sum = load_npy(debug / f"{face}_weight_sum.npy", shape, 0.0)
    mask_boundary_trust = load_npy(debug / f"{face}_mask_boundary_trust.npy", shape, 0.0)
    footprint_area = load_npy(debug / f"{face}_footprint_area.npy", shape, 0.0)

    positive = weight_sum[final_keep & (weight_sum > 1e-8)]
    weight_scale = float(np.percentile(positive, 90.0)) if positive.size else 1.0
    count_rel = np.clip(valid_count / max(1, args.keep_valid_views), 0.0, 1.0)
    weight_rel = np.clip(weight_sum / max(weight_scale, 1e-8), 0.0, 1.0)
    footprint_rel = np.clip(footprint_area / max(args.footprint_keep_min, 1e-6), 0.0, 1.0)
    reliability = (
        np.sqrt(count_rel * weight_rel)
        * clean_score
        * (1.0 - 0.62 * object_risk)
        * np.sqrt(np.clip(mask_boundary_trust, 0.0, 1.0))
        * np.sqrt(footprint_rel)
    ).astype(np.float32)
    reliability[~final_keep] = 0.0

    source = (
        final_keep
        & (valid_count >= args.candidate_valid_views)
        & (clean_score >= args.source_clean_thresh)
        & (object_risk <= args.source_object_risk_thresh)
        & (mask_boundary_trust >= args.mask_boundary_keep_thresh)
    )
    high = (
        final_keep
        & (valid_count >= args.keep_valid_views)
        & (clean_score >= args.keep_clean_thresh)
        & (object_risk <= args.keep_object_risk_thresh)
        & (mask_boundary_trust >= args.mask_boundary_keep_thresh)
        & (footprint_area >= args.footprint_keep_min)
    )
    if np.count_nonzero(source) < 0.006 * source.size:
        source = final_keep & (valid_count >= args.min_valid_views)
    return {
        "valid_count": valid_count,
        "candidate_count": candidate_count,
        "clean_score": clean_score,
        "object_risk": object_risk,
        "weight_sum": weight_sum,
        "mask_boundary_trust": mask_boundary_trust,
        "footprint_area": footprint_area,
        "contaminant": ~final_keep,
        "observed": final_keep,
        "source": source,
        "high": high,
        "fill": ~final_keep,
        "mid": final_keep & ~high,
        "reliability": reliability,
    }


def representative_support(shape: tuple[int, int], masks: dict, cluster: dict) -> np.ndarray:
    y, x, size = [int(v) for v in cluster["representative"]["box"]]
    support = np.zeros(shape, dtype=bool)
    sl = np.s_[y : y + size, x : x + size]
    support[sl] = masks["source"][sl] | masks["high"][sl]
    if np.count_nonzero(support) < 128:
        support[sl] = masks["observed"][sl]
    return support


def target_tile_from_region(face: str, image: np.ndarray, masks: dict, box: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    y, x, size = [int(v) for v in box]
    sl = np.s_[y : y + size, x : x + size]
    tile = image[sl].copy()
    tile_mask = (masks["source"][sl] | masks["high"][sl]).astype(bool)
    if np.count_nonzero(tile_mask) < 64:
        tile_mask = masks["observed"][sl].astype(bool)
    if np.count_nonzero(tile_mask) < 64:
        tile_mask = np.ones(tile.shape[:2], dtype=bool)
    fixed = gmp.inpaint_tile_holes(tile, tile_mask)
    if face in {"floor", "ceiling"}:
        fixed = mm.normalize_tile_lighting(fixed)
    return np.clip(fixed, 0.0, 1.0), tile_mask


def target_tile_from_material_cluster(
    face: str,
    image: np.ndarray,
    masks: dict,
    box: tuple[int, int, int],
    material_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y, x, size = [int(value) for value in box]
    sl = np.s_[y : y + size, x : x + size]
    tile = image[sl].copy()
    tile_mask = material_mask[sl] & (masks["source"][sl] | masks["high"][sl])
    if np.count_nonzero(tile_mask) < 64:
        tile_mask = material_mask[sl] & masks["observed"][sl]
    fixed = gmp.inpaint_tile_holes(tile, tile_mask)
    if face in {"floor", "ceiling"}:
        fixed = mm.normalize_tile_lighting(fixed)
    return np.clip(fixed, 0.0, 1.0), tile_mask.astype(bool)


def select_regions(args: argparse.Namespace, face: str, image: np.ndarray, masks: dict) -> tuple[list[dict], list[dict]]:
    candidates = mm.enumerate_candidates(
        args,
        face,
        image,
        masks,
        args.tile_size,
        face_min_source_fraction(args, face),
    )
    spread = mm.select_spread_candidates(args, candidates, image.shape[:2])
    clusters = mm.cluster_candidates(args, spread)
    return candidates, clusters


def material_feature_scales(face: str) -> np.ndarray:
    if face.startswith("wall_"):
        return np.array([45.0, 8.0, 8.0], dtype=np.float32)
    return np.array([32.0, 10.0, 10.0], dtype=np.float32)


def box_iou(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    ay, ax, side_a = first
    by, bx, side_b = second
    y0 = max(ay, by)
    x0 = max(ax, bx)
    y1 = min(ay + side_a, by + side_b)
    x1 = min(ax + side_a, bx + side_b)
    intersection = max(0, y1 - y0) * max(0, x1 - x0)
    union = side_a * side_a + side_b * side_b - intersection
    return float(intersection / max(union, 1))


def best_material_region_box(
    material_mask: np.ndarray,
    observed: np.ndarray,
    reliability: np.ndarray,
    purity_threshold: float,
    min_size: int,
) -> tuple[int, int, int] | None:
    h, w = material_mask.shape
    max_side = min(512, h, w)
    sizes = []
    side = max_side
    while side >= min_size:
        sizes.append(side)
        next_side = int(round(side * 0.78))
        side = next_side if next_side < side else side - 1
    if min_size not in sizes:
        sizes.append(min_size)
    material_integral = cv2.integral(material_mask.astype(np.float32))
    observed_integral = cv2.integral(observed.astype(np.float32))
    reliability_integral = cv2.integral((reliability * material_mask).astype(np.float32))
    best = None
    best_score = -np.inf
    for side in sizes:
        stride = max(8, side // 8)
        y_positions = list(range(0, h - side + 1, stride))
        x_positions = list(range(0, w - side + 1, stride))
        if y_positions[-1] != h - side:
            y_positions.append(h - side)
        if x_positions[-1] != w - side:
            x_positions.append(w - side)
        for y in y_positions:
            for x in x_positions:
                material_count = integral_rect_sum(material_integral, y, x, side)
                observed_count = integral_rect_sum(observed_integral, y, x, side)
                if observed_count < 0.28 * side * side:
                    continue
                purity = material_count / max(observed_count, 1.0)
                if purity < purity_threshold:
                    continue
                reliable = integral_rect_sum(reliability_integral, y, x, side) / max(material_count, 1.0)
                score = side * side * purity * (0.55 + 0.45 * reliable)
                if score > best_score:
                    best_score = score
                    best = (int(y), int(x), int(side))
    return best


def discover_weighted_material_regions(
    args: argparse.Namespace,
    face: str,
    image: np.ndarray,
    masks: dict,
) -> tuple[list[dict], list[dict]]:
    observed = (masks["source"] | masks["high"]).astype(bool)
    if np.count_nonzero(observed) < 512:
        observed = masks["observed"].astype(bool)
    rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    values = lab[observed]
    ys, xs = np.nonzero(observed)
    if values.shape[0] < 256:
        return select_regions(args, face, image, masks)

    scales = material_feature_scales(face)
    sample_count = min(values.shape[0], int(args.material_cluster_max_samples))
    reliability_values = np.maximum(masks["reliability"][observed], 1e-4)
    probability = reliability_values / np.sum(reliability_values)
    rng = np.random.default_rng(args.seed + sum(ord(char) for char in face))
    sample_indices = rng.choice(values.shape[0], sample_count, replace=False, p=probability)
    sample = values[sample_indices] / scales
    component_count = min(int(args.material_cluster_components), max(1, sample_count // 128))
    _, sample_labels, centers_scaled = cv2.kmeans(
        sample.astype(np.float32),
        component_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.02),
        8,
        cv2.KMEANS_PP_CENTERS,
    )
    centers = centers_scaled * scales
    sample_counts = np.bincount(sample_labels.reshape(-1), minlength=component_count).astype(np.float64)

    parent = list(range(component_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first = find(first)
        second = find(second)
        if first != second:
            parent[second] = first

    for first in range(component_count):
        for second in range(first):
            chroma = float(np.linalg.norm((centers[first, 1:] - centers[second, 1:]) / 8.0))
            lightness = abs(float(centers[first, 0] - centers[second, 0]))
            if chroma < 0.55 and lightness < 90.0:
                union(first, second)

    component_groups: dict[int, list[int]] = {}
    for index in range(component_count):
        component_groups.setdefault(find(index), []).append(index)
    group_centers = []
    for members in component_groups.values():
        weights = np.maximum(sample_counts[members], 1.0)
        group_centers.append(np.average(centers[members], axis=0, weights=weights))
    group_centers = np.asarray(group_centers, dtype=np.float32)

    normalized_values = values / scales
    normalized_centers = group_centers / scales
    labels_observed = np.empty(values.shape[0], dtype=np.int16)
    for start in range(0, values.shape[0], 250000):
        block = normalized_values[start : start + 250000]
        labels_observed[start : start + len(block)] = np.argmin(
            np.sum((block[:, None, :] - normalized_centers[None, :, :]) ** 2, axis=2),
            axis=1,
        )

    counts = np.bincount(labels_observed, minlength=len(group_centers)).astype(np.float64)
    fractions = counts / max(float(np.sum(counts)), 1.0)
    y_medians = np.array(
        [
            float(np.median(ys[labels_observed == index]))
            if np.any(labels_observed == index)
            else 0.5 * image.shape[0]
            for index in range(len(group_centers))
        ],
        dtype=np.float32,
    )
    major = [index for index, fraction in enumerate(fractions) if fraction >= args.material_cluster_min_fraction]
    if not major:
        major = [int(np.argmax(fractions))]
    # v3b_realrooms: optional per-face override to force a single material (faces with almost no
    # observation, e.g. a wall fully covered by a door/cabinets, otherwise yield a second cluster with
    # no usable CHORD candidate and break the atlas compose with a missing_chord_fallback stem).
    if face in set(getattr(args, "single_material_faces", None) or []):
        major = [int(np.argmax(fractions))]
    face_caps = {}
    for spec in getattr(args, "face_max_materials", None) or []:
        name, _, cap = spec.partition("=")
        if cap.isdigit():
            face_caps[name] = int(cap)
    cap = face_caps.get(face)
    if cap is not None and len(major) > cap:
        # Keep the most mutually distinct clusters instead of the largest ones: lighting casts split a
        # single paint into several large clusters, and keeping "top by fraction" can drop the real second
        # material (e.g. a wood wainscot) while keeping two shades of the same white.
        import itertools as _it
        best_subset = None
        best_score = -1.0
        for subset in _it.combinations(major, cap):
            score = 0.0
            for first, second in _it.combinations(subset, 2):
                chroma = float(np.linalg.norm((group_centers[first, 1:] - group_centers[second, 1:]) / 8.0))
                lightness = abs(float(group_centers[first, 0] - group_centers[second, 0])) / 50.0
                weight = float(np.sqrt(fractions[first] * fractions[second]))
                score += weight * (chroma + 0.35 * lightness)
            if score > best_score:
                best_score = score
                best_subset = subset
        major = list(best_subset)
    for index in range(len(group_centers)):
        if index in major:
            continue
        costs = []
        for candidate in major:
            chroma = float(np.linalg.norm((group_centers[index, 1:] - group_centers[candidate, 1:]) / 8.0))
            y_delta = abs(float(y_medians[index] - y_medians[candidate])) / max(image.shape[0], 1)
            lightness = abs(float(group_centers[index, 0] - group_centers[candidate, 0])) / 50.0
            costs.append(chroma + 0.8 * y_delta + 0.2 * lightness)
        labels_observed[labels_observed == index] = major[int(np.argmin(costs))]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clusters = []
    candidates = []
    for discovery_index, label in enumerate(sorted(int(value) for value in np.unique(labels_observed))):
        material_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        selected = labels_observed == label
        material_mask[ys[selected], xs[selected]] = 1
        material_mask = cv2.morphologyEx(material_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        material_mask = cv2.morphologyEx(material_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        material_mask = (material_mask > 0) & observed
        fraction = float(np.count_nonzero(material_mask) / max(np.count_nonzero(observed), 1))
        if fraction < args.material_cluster_min_fraction:
            continue
        box = best_material_region_box(
            material_mask,
            observed,
            masks["reliability"],
            args.material_cluster_purity,
            args.material_cluster_min_region_size,
        )
        if box is None:
            continue
        material_values = lab[material_mask]
        score = float(np.sum(masks["reliability"][material_mask]))
        representative = {
            "box": box,
            "score": score,
            "mean_lab": np.median(material_values, axis=0).astype(np.float32),
            "edge_mean": float(np.mean(edge[material_mask])),
            "sat_mean": float(np.mean(hsv[..., 1][material_mask]) / 255.0),
            "material_purity": float(
                np.count_nonzero(material_mask[box[0] : box[0] + box[2], box[1] : box[1] + box[2]])
                / max(
                    np.count_nonzero(observed[box[0] : box[0] + box[2], box[1] : box[1] + box[2]]),
                    1,
                )
            ),
        }
        clusters.append(
            {
                "items": [representative],
                "score": score,
                "mean_lab": representative["mean_lab"],
                "edge_mean": representative["edge_mean"],
                "sat_mean": representative["sat_mean"],
                "representative": representative,
                "material_mask": material_mask,
                "material_fraction": fraction,
                "discovery_index": discovery_index,
            }
        )
        candidates.append(representative)

    # Material identity and exemplar quality are separate decisions. The older
    # fixed-window selector often finds a sharper spatial observation of an
    # already discovered material, so retain those boxes as extra exemplars
    # instead of discarding them during color-cluster merging.
    _, classic_clusters = select_regions(args, face, image, masks)
    for classic in classic_clusters:
        classic_rep = dict(classic["representative"])
        y, x, side = [int(value) for value in classic_rep["box"]]
        sl = np.s_[y : y + side, x : x + side]
        observed_count = max(int(np.count_nonzero(observed[sl])), 1)
        purities = [float(np.count_nonzero(cluster["material_mask"][sl]) / observed_count) for cluster in clusters]
        if not purities:
            continue
        owner = int(np.argmax(purities))
        purity = purities[owner]
        if purity < 0.18:
            continue
        classic_rep["material_purity"] = purity
        classic_rep["source"] = "classic_high_weight_region"
        owner_items = clusters[owner]["items"]
        if any(box_iou(tuple(classic_rep["box"]), tuple(item["box"])) >= 0.72 for item in owner_items):
            continue
        owner_items.append(classic_rep)
        candidates.append(classic_rep)

    max_exemplars = max(1, int(args.material_cluster_exemplars))
    for cluster in clusters:
        cluster["items"].sort(
            key=lambda item: float(item.get("score", 0.0))
            * (0.35 + 0.65 * float(item.get("material_purity", 0.0))),
            reverse=True,
        )
        cluster["items"] = cluster["items"][:max_exemplars]
        if all(item is not cluster["representative"] for item in cluster["items"]):
            cluster["items"][-1] = cluster["representative"]
    clusters.sort(key=lambda item: item["score"], reverse=True)
    keep_count = max(args.min_priors_per_face, min(args.max_priors_per_face, len(clusters)))
    return candidates, clusters[:keep_count]


def discover_persistent_wall_bands(
    args: argparse.Namespace,
    face: str,
    image: np.ndarray,
    masks: dict,
    clusters: list[dict],
) -> tuple[list[dict], dict]:
    """Propose coherent horizontal materials that Lab clustering tends to split apart."""
    audit = {"enabled": bool(args.discover_persistent_wall_bands), "bands": []}
    if not args.discover_persistent_wall_bands or not face.startswith("wall_"):
        return clusters, audit
    if face in set(getattr(args, "single_material_faces", None) or []):
        audit["reason"] = "face_forced_single_material"
        return clusters, audit

    observed = (masks["source"] | masks["high"]).astype(bool)
    if np.count_nonzero(observed) < 512:
        observed = masks["observed"].astype(bool)
    h, w = observed.shape
    x_margin = max(1, int(round(0.04 * w)))
    x0, x1 = x_margin, max(x_margin + 1, w - x_margin)
    core_observed = observed[:, x0:x1]
    core_image = np.clip(image[:, x0:x1] * 255.0, 0, 255).astype(np.float32)

    descriptor = np.zeros((h, 11), dtype=np.float32)
    row_coverage = np.mean(core_observed, axis=1).astype(np.float32)
    for y in range(h):
        values = core_image[y][core_observed[y]]
        if values.shape[0] < 16:
            descriptor[y] = np.nan
            continue
        descriptor[y, :9] = np.percentile(values, [20.0, 50.0, 80.0], axis=0).reshape(-1)
        gray_values = np.mean(values, axis=1)
        descriptor[y, 9] = float(np.std(gray_values))
        descriptor[y, 10] = (
            float(np.mean(np.abs(np.diff(gray_values)))) if gray_values.size > 1 else 0.0
        )

    valid_rows = np.flatnonzero(np.isfinite(descriptor[:, 0]) & (row_coverage >= 0.20))
    if valid_rows.size < max(32, int(round(0.25 * h))):
        audit["reason"] = "insufficient_observed_rows"
        return clusters, audit
    row_ids = np.arange(h)
    for channel in range(descriptor.shape[1]):
        descriptor[:, channel] = np.interp(row_ids, valid_rows, descriptor[valid_rows, channel])
    descriptor /= np.asarray([35.0, 15.0, 15.0] * 3 + [25.0, 25.0], dtype=np.float32)
    sigma = max(1.0, 0.004 * h)
    smooth = cv2.GaussianBlur(
        descriptor.reshape(h, 1, -1), (0, 0), sigmaX=0.0, sigmaY=sigma
    ).reshape(h, -1)
    transition = np.linalg.norm(np.diff(smooth, axis=0), axis=1)
    interior = transition[max(8, int(0.02 * h)) : max(9, int(0.98 * h))]
    if interior.size < 16:
        audit["reason"] = "small_transition_profile"
        return clusters, audit
    peak_threshold = float(np.percentile(interior, 90.0))
    nms_radius = max(4, int(round(0.006 * h)))
    peaks = []
    for y in range(nms_radius, h - nms_radius - 1):
        if transition[y] < peak_threshold:
            continue
        if transition[y] >= float(np.max(transition[y - nms_radius : y + nms_radius + 1])):
            peaks.append(int(y))

    min_height = max(
        int(args.material_cluster_min_region_size),
        int(round(args.wall_band_min_height_frac * h)),
    )
    max_height = max(min_height, int(round(args.wall_band_max_height_frac * h)))
    base_context = max(24, int(round(args.wall_band_context_frac * h)))
    proposals = []
    for first_i, top in enumerate(peaks):
        for bottom in peaks[first_i + 1 :]:
            band_height = int(bottom - top)
            if band_height < min_height:
                continue
            if band_height > max_height:
                break
            center_fraction = float(0.5 * (top + bottom) / max(h, 1))
            if center_fraction < float(args.wall_band_min_center_frac):
                continue
            if center_fraction > float(args.wall_band_max_center_frac):
                continue
            context = max(min_height, min(base_context, band_height))
            if top - context < 4 or bottom + context >= h - 4:
                continue
            band_value = np.mean(smooth[top + 3 : max(top + 4, bottom - 2)], axis=0)
            above_value = np.mean(smooth[top - context : top - 3], axis=0)
            below_value = np.mean(smooth[bottom + 3 : bottom + context], axis=0)
            above_delta = float(np.linalg.norm(band_value - above_value))
            below_delta = float(np.linalg.norm(band_value - below_value))
            context_delta = float(np.linalg.norm(above_value - below_value))
            score = (
                min(above_delta, below_delta)
                + 0.20 * float(transition[top] + transition[bottom])
                - 0.15 * context_delta
            )
            coverage = float(np.median(row_coverage[top : bottom + 1]))
            if score < float(args.wall_band_min_score):
                continue
            if coverage < float(args.wall_band_min_tangent_coverage):
                continue
            confidence = float(
                np.clip(
                    0.50 * coverage
                    + 0.50 * score / max(2.0 * float(args.wall_band_min_score), 1e-6),
                    0.0,
                    1.0,
                )
            )
            proposals.append(
                {
                    "top": int(top),
                    "bottom": int(bottom),
                    "score": score,
                    "coverage": coverage,
                    "confidence": confidence,
                    "center_fraction": center_fraction,
                    "above_delta": above_delta,
                    "below_delta": below_delta,
                    "context_delta": context_delta,
                }
            )

    proposals.sort(key=lambda item: float(item["score"]), reverse=True)
    selected = []
    for proposal in proposals:
        top, bottom = int(proposal["top"]), int(proposal["bottom"])
        if any(
            max(top, int(other["top"])) < min(bottom, int(other["bottom"]))
            for other in selected
        ):
            continue
        selected.append(proposal)
        if len(selected) >= max(0, int(args.wall_band_max_count)):
            break

    edge_merge = max(1, int(round(args.wall_band_edge_merge_frac * h)))
    rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy) / 255.0
    band_clusters = []
    for proposal in sorted(selected, key=lambda item: int(item["top"])):
        top, bottom = int(proposal["top"]), int(proposal["bottom"])
        nearby_top = [peak for peak in peaks if 0 < top - peak <= edge_merge]
        nearby_bottom = [peak for peak in peaks if 0 < peak - bottom <= edge_merge]
        if nearby_top:
            top = int(min(nearby_top))
        if nearby_bottom:
            bottom = int(max(nearby_bottom))
        if bottom - top > max_height:
            continue
        material_mask = np.zeros((h, w), dtype=bool)
        material_mask[max(0, top) : min(h, bottom + 1)] = observed[
            max(0, top) : min(h, bottom + 1)
        ]
        fraction = float(np.count_nonzero(material_mask) / max(np.count_nonzero(observed), 1))
        box = best_material_region_box(
            material_mask,
            observed,
            masks["reliability"],
            min(float(args.material_cluster_purity), 0.84),
            max(24, min(int(args.material_cluster_min_region_size), max(24, bottom - top))),
        )
        if box is None:
            continue
        values = lab[material_mask]
        reliability_score = float(np.sum(masks["reliability"][material_mask]))
        representative = {
            "box": box,
            "score": reliability_score,
            "mean_lab": np.median(values, axis=0).astype(np.float32),
            "edge_mean": float(np.mean(edge[material_mask])),
            "sat_mean": float(np.mean(hsv[..., 1][material_mask]) / 255.0),
            "material_purity": float(
                np.count_nonzero(
                    material_mask[box[0] : box[0] + box[2], box[1] : box[1] + box[2]]
                )
                / max(
                    np.count_nonzero(
                        observed[box[0] : box[0] + box[2], box[1] : box[1] + box[2]]
                    ),
                    1,
                )
            ),
            "source": "persistent_wall_band",
        }
        discovery_index = f"wall_band_{len(band_clusters):02d}"
        band_clusters.append(
            {
                "items": [representative],
                "score": reliability_score,
                "mean_lab": representative["mean_lab"],
                "edge_mean": representative["edge_mean"],
                "sat_mean": representative["sat_mean"],
                "representative": representative,
                "material_mask": material_mask,
                "material_fraction": fraction,
                "discovery_index": discovery_index,
            }
        )
        audit["bands"].append(
            {
                **proposal,
                "discovery_index": discovery_index,
                "expanded_top": int(top),
                "expanded_bottom": int(bottom),
                "height_fraction": float((bottom - top + 1) / max(h, 1)),
                "material_fraction": fraction,
                "box_yx_size": [int(value) for value in box],
            }
        )

    if not band_clusters:
        audit["reason"] = "no_valid_persistent_band"
        return clusters, audit

    union_band = np.logical_or.reduce([cluster["material_mask"] for cluster in band_clusters])
    cleaned = []
    for cluster in clusters:
        updated = dict(cluster)
        updated_mask = np.asarray(cluster["material_mask"], dtype=bool) & ~union_band
        if np.count_nonzero(updated_mask) < 128:
            continue
        updated["material_mask"] = updated_mask
        updated["material_fraction"] = float(
            np.count_nonzero(updated_mask) / max(np.count_nonzero(observed), 1)
        )
        cleaned.append(updated)
    return cleaned + band_clusters, audit


def apply_integrated_material_cap(
    args: argparse.Namespace,
    face: str,
    clusters: list[dict],
) -> tuple[list[dict], dict]:
    cap = int(args.max_priors_per_face)
    for spec in getattr(args, "face_max_materials", None) or []:
        name, _, value = spec.partition("=")
        if name == face and value.isdigit():
            cap = min(cap, int(value))
    cap = max(int(args.min_priors_per_face), cap)
    if len(clusters) <= cap:
        return clusters, {"cap": cap, "applied": False}
    bands = [
        cluster
        for cluster in clusters
        if str(cluster.get("discovery_index", "")).startswith("wall_band_")
    ]
    generic = [cluster for cluster in clusters if cluster not in bands]
    if bands and bool(args.wall_band_reserve_material_slot) and cap >= 2:
        band_slots = min(len(bands), max(1, cap - 1))
        generic_slots = max(1, cap - band_slots)
        kept = generic[:generic_slots] + bands[:band_slots]
    else:
        kept = sorted(clusters, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:cap]
    return kept, {
        "cap": cap,
        "applied": True,
        "input_count": len(clusters),
        "output_count": len(kept),
        "reserved_band_slots": sum(
            str(cluster.get("discovery_index", "")).startswith("wall_band_") for cluster in kept
        ),
    }


def depth_surface_for_pose(
    args: argparse.Namespace,
    pose: proj.ImagePose,
    pts_da3: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    face: str,
    manifest: dict,
    metas: dict,
    sim: proj.Similarity,
    da3_view: proj.Da3View | None,
    da3_depth_calib: proj.Da3DepthCalibration | None,
    raw_to_room_matrix: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sampled_conf = np.ones(u.shape, dtype=np.float32)
    if da3_view is not None and da3_depth_calib is not None:
        du, dv = proj.da3_uv_from_colmap_pixels(u, v, pose, da3_view)
        sampled_depth_raw = proj.sample_float_map(da3_view.depth, du, dv, border_value=np.nan)
        sampled_depth = proj.apply_depth_calibration(sampled_depth_raw, da3_depth_calib)
        if da3_view.conf is not None:
            sampled_conf = proj.sample_float_map(da3_view.conf, du, dv, border_value=0.0)
        has_depth = np.isfinite(sampled_depth) & (sampled_depth > 1e-6)
        camera_depth = sampled_depth
        projected_depth = z
        if args.surface_distance_tol > 0.0:
            sampled_world = proj.backproject_camera_z_to_world(u, v, camera_depth, pose)
            if sampled_world is not None:
                sampled_world = sim.colmap_to_da3(sampled_world)
            else:
                sampled_world = proj.backproject_da3_raw_depth_to_room(
                    du,
                    dv,
                    sampled_depth_raw,
                    da3_view,
                    raw_to_room_matrix=raw_to_room_matrix,
                )
            if sampled_world is None:
                surface_distance = np.full(u.shape, np.inf, dtype=np.float32)
            else:
                surface_distance = proj.face_surface_distance(sampled_world, face, manifest, metas)
        else:
            surface_distance = np.zeros(u.shape, dtype=np.float32)
    else:
        has_depth = np.zeros(u.shape, dtype=bool)
        camera_depth = np.full(u.shape, np.inf, dtype=np.float32)
        projected_depth = z
        surface_distance = np.full(u.shape, np.inf, dtype=np.float32)
    return has_depth, camera_depth, projected_depth, surface_distance, sampled_conf


def footprint_weight_for_samples(
    args: argparse.Namespace,
    face: str,
    rows_sel: np.ndarray,
    cols_sel: np.ndarray,
    size: tuple[int, int],
    u_sel: np.ndarray,
    v_sel: np.ndarray,
    pose: proj.ImagePose,
    sim: proj.Similarity,
    manifest: dict,
    metas: dict,
) -> tuple[np.ndarray, np.ndarray]:
    w, h = size
    cols_x = np.where(cols_sel < w - 1, cols_sel + 1, np.maximum(cols_sel - 1, 0)).astype(np.int32)
    rows_y = np.where(rows_sel < h - 1, rows_sel + 1, np.maximum(rows_sel - 1, 0)).astype(np.int32)
    dx_sign = np.where(cols_sel < w - 1, 1.0, -1.0).astype(np.float32)
    dy_sign = np.where(rows_sel < h - 1, 1.0, -1.0).astype(np.float32)
    pts_x = proj.face_points_for_indices(face, rows_sel, cols_x, size, manifest, metas)
    pts_y = proj.face_points_for_indices(face, rows_y, cols_sel, size, manifest, metas)
    u_x, v_x, _ = proj.project_points(sim.da3_to_colmap(pts_x), pose)
    u_y, v_y, _ = proj.project_points(sim.da3_to_colmap(pts_y), pose)
    du_dx = (u_x - u_sel) / dx_sign
    dv_dx = (v_x - v_sel) / dx_sign
    du_dy = (u_y - u_sel) / dy_sign
    dv_dy = (v_y - v_sel) / dy_sign
    footprint_area = np.abs(du_dx * dv_dy - du_dy * dv_dx).astype(np.float32)
    footprint_area = np.where(np.isfinite(footprint_area), footprint_area, 0.0)
    if args.footprint_min_area > 0.0:
        weight = np.clip(footprint_area / max(float(args.footprint_min_area), 1e-6), 0.0, 1.0)
        if args.footprint_power > 0.0 and args.footprint_power != 1.0:
            weight = np.power(weight, args.footprint_power)
    else:
        weight = np.ones(rows_sel.shape, dtype=np.float32)
    return weight.astype(np.float32), footprint_area.astype(np.float32)


def trace_region_contributors(
    args: argparse.Namespace,
    face: str,
    region_i: int,
    support: np.ndarray,
    poses: list[proj.ImagePose],
    sim: proj.Similarity,
    manifest: dict,
    metas: dict,
    all_faces: list[str],
    da3_views: dict[int, proj.Da3View],
    raw_to_room_matrix: np.ndarray | None,
    caches: dict,
) -> list[dict]:
    size = proj.face_texture_size(args.polygon_source_dir, metas[face])
    w, h = size
    ys, xs = np.nonzero(support)
    if ys.size == 0:
        return []
    if ys.size > args.max_region_sample_texels:
        pick = np.linspace(0, ys.size - 1, args.max_region_sample_texels).astype(np.int64)
        ys = ys[pick]
        xs = xs[pick]
    rows = ys.astype(np.int32)
    cols = xs.astype(np.int32)
    pts_da3 = proj.face_points_for_indices(face, rows, cols, size, manifest, metas)
    pts_col = sim.da3_to_colmap(pts_da3)
    n = proj.face_normal(face, manifest, metas)
    face_idx = all_faces.index(face)

    contributors = []
    for pose_i, pose in enumerate(poses):
        if pose.image_id not in caches["face_id"] or pose.image_id not in caches["zbuffer"]:
            face_id_map, zbuf = proj.build_face_id_and_zbuffer(
                pose,
                all_faces,
                args.polygon_source_dir,
                sim,
                manifest,
                metas,
                args.zbuffer_stride,
            )
            caches["face_id"][pose.image_id] = face_id_map
            caches["zbuffer"][pose.image_id] = zbuf
        if pose.image_id not in caches["reject"]:
            caches["reject"][pose.image_id] = proj.load_view_reject_maps(pose, args)
        if pose.image_id not in caches["depth_calib"] and pose.image_id in da3_views:
            caches["depth_calib"][pose.image_id] = proj.calibrate_da3_depth_to_colmap_zbuffer(
                da3_views[pose.image_id],
                pose,
                caches["zbuffer"][pose.image_id],
            )

        u, v, z = proj.project_points(pts_col, pose)
        view = pose.center_da3[None, :] - pts_da3
        dist = np.linalg.norm(view, axis=1)
        cos = np.abs((view / np.maximum(dist[:, None], 1e-8)) @ n)
        in_frame = (
            (z > 1e-6)
            & (u >= 0.0)
            & (v >= 0.0)
            & (u <= pose.width - 1.0)
            & (v <= pose.height - 1.0)
            & (cos >= args.min_view_cos)
        )
        if not np.any(in_frame):
            continue
        idx0 = np.flatnonzero(in_frame)
        uu = u[idx0]
        vv = v[idx0]
        zz = z[idx0]
        face_id_map = caches["face_id"][pose.image_id]
        object_mask, object_risk, boundary_trust = caches["reject"][pose.image_id]
        da3_view = da3_views.get(pose.image_id)
        da3_depth_calib = caches["depth_calib"].get(pose.image_id)
        has_depth, camera_depth, projected_depth, surface_distance, sampled_conf = depth_surface_for_pose(
            args,
            pose,
            pts_da3[idx0],
            uu,
            vv,
            zz,
            face,
            manifest,
            metas,
            sim,
            da3_view,
            da3_depth_calib,
            raw_to_room_matrix,
        )
        depth_tol = args.depth_abs_tol + args.depth_rel_tol * np.maximum(camera_depth, 0.0)
        depth_diff = np.full(idx0.shape, np.inf, dtype=np.float32)
        depth_diff[has_depth] = np.abs(projected_depth[has_depth] - camera_depth[has_depth])
        sampled_face = proj.sample_nearest_map(face_id_map, uu, vv, border_value=255)
        sampled_object = proj.sample_nearest_map(object_mask, uu, vv, border_value=0)
        sampled_object_risk = proj.sample_float_map(object_risk, uu, vv, border_value=0.0)
        sampled_boundary = proj.sample_float_map(boundary_trust, uu, vv, border_value=1.0)
        surface_ok = np.ones(idx0.shape, dtype=bool)
        if args.surface_distance_tol > 0.0 and args.surface_distance_hard_gate:
            surface_ok = np.isfinite(surface_distance) & (surface_distance <= args.surface_distance_tol)
        valid = (
            has_depth
            & np.isfinite(depth_diff)
            & (sampled_conf >= args.min_conf)
            & (depth_diff <= depth_tol)
            & (sampled_face == face_idx)
            & (sampled_object == 0)
            & (sampled_object_risk <= args.object_risk_hard_thresh)
            & (sampled_boundary >= args.min_mask_boundary_trust)
            & surface_ok
        )
        if not np.any(valid):
            continue
        idx = idx0[valid]
        uu = u[idx]
        vv = v[idx]
        depth_diff_v = depth_diff[valid]
        depth_tol_v = depth_tol[valid]
        surface_distance_v = surface_distance[valid]
        sampled_conf_v = sampled_conf[valid]
        sampled_boundary_v = sampled_boundary[valid]
        rows_v = rows[idx]
        cols_v = cols[idx]
        footprint_w, footprint_area = footprint_weight_for_samples(
            args,
            face,
            rows_v,
            cols_v,
            size,
            uu,
            vv,
            pose,
            sim,
            manifest,
            metas,
        )
        angle_w = np.clip(cos[idx], 0.0, 1.0) ** 2
        if args.distance_weight_power > 0.0:
            distance_w = 1.0 / (
                1.0 + dist[idx] / max(args.distance_weight_scale, 1e-6)
            ) ** args.distance_weight_power
        else:
            distance_w = np.ones(idx.shape, dtype=np.float32)
        depth_w = 1.0 / (1.0 + depth_diff_v / np.maximum(depth_tol_v, 1e-6))
        if args.surface_distance_tol > 0.0:
            surface_w = 1.0 / (1.0 + surface_distance_v / max(float(args.surface_distance_tol), 1e-6))
            if args.surface_distance_power > 0.0 and args.surface_distance_power != 1.0:
                surface_w = np.power(surface_w, float(args.surface_distance_power))
        else:
            surface_w = np.ones(idx.shape, dtype=np.float32)
        weights = (
            sampled_conf_v
            * angle_w
            * distance_w
            * depth_w
            * surface_w
            * np.clip(sampled_boundary_v, 0.0, 1.0)
            * footprint_w
        ).astype(np.float32)
        weights[~np.isfinite(weights)] = 0.0
        keep = weights > 1e-8
        if not np.any(keep):
            continue
        contributors.append(
            {
                "pose": pose,
                "pose_order": pose_i,
                "weight": float(np.sum(weights[keep])),
                "count": int(np.count_nonzero(keep)),
                "u": uu[keep].astype(np.float32),
                "v": vv[keep].astype(np.float32),
                "rows": rows_v[keep].astype(np.int32),
                "cols": cols_v[keep].astype(np.int32),
                "mean_depth_residual": float(np.mean(depth_diff_v[keep] / np.maximum(depth_tol_v[keep], 1e-6))),
                "mean_surface_distance": float(np.mean(surface_distance_v[keep])),
                "mean_footprint_area": float(np.mean(footprint_area[keep])),
            }
        )
    total = sum(item["weight"] for item in contributors)
    contributors.sort(key=lambda item: item["weight"], reverse=True)
    if total > 0:
        for item in contributors:
            item["weight_frac"] = float(item["weight"] / total)
    return contributors


def square_bbox_from_mask(mask: np.ndarray, margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0 = int(max(0, ys.min() - margin))
    y1 = int(min(shape[0], ys.max() + margin + 1))
    x0 = int(max(0, xs.min() - margin))
    x1 = int(min(shape[1], xs.max() + margin + 1))
    side = max(y1 - y0, x1 - x0, 32)
    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    y0 = int(round(cy - 0.5 * side))
    x0 = int(round(cx - 0.5 * side))
    y0 = max(0, min(y0, shape[0] - side))
    x0 = max(0, min(x0, shape[1] - side))
    y1 = int(min(shape[0], y0 + side))
    x1 = int(min(shape[1], x0 + side))
    return y0, y1, x0, x1


def integral_rect_sum(integral: np.ndarray, y: int, x: int, side: int) -> float:
    y1 = y + side
    x1 = x + side
    return float(integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x])


def best_pure_crop_box(
    mask: np.ndarray,
    min_size: int,
    min_frac: float,
    stride_frac: float,
) -> tuple[int, int, int, int, float] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    max_side = min(h, w, max(y_max - y_min, x_max - x_min))
    min_size = int(max(16, min(min_size, max_side)))
    if max_side < min_size:
        return None
    sizes = []
    side = int(max_side)
    while side >= min_size:
        sizes.append(side)
        side = int(round(side * 0.82))
        if sizes and side == sizes[-1]:
            side -= 1
    if min_size not in sizes:
        sizes.append(min_size)
    integral = cv2.integral(mask.astype(np.float32))
    best: tuple[int, int, int, int, float] | None = None
    best_score = -1.0
    for side in sizes:
        stride = max(4, int(round(side * max(0.04, stride_frac))))
        y0 = max(0, y_min - side + 1)
        y1 = min(y_max, h - side) + 1
        x0 = max(0, x_min - side + 1)
        x1 = min(x_max, w - side) + 1
        if y1 <= y0 or x1 <= x0:
            continue
        ys_grid = list(range(y0, y1, stride))
        xs_grid = list(range(x0, x1, stride))
        if ys_grid[-1] != y1 - 1:
            ys_grid.append(y1 - 1)
        if xs_grid[-1] != x1 - 1:
            xs_grid.append(x1 - 1)
        for yy in ys_grid:
            for xx in xs_grid:
                frac = integral_rect_sum(integral, yy, xx, side) / float(side * side)
                if frac < min_frac:
                    continue
                score = side * side * (0.6 + 0.4 * frac)
                if score > best_score:
                    best_score = score
                    best = (yy, yy + side, xx, xx + side, float(frac))
        if best is not None:
            break
    return best


def make_pure_chord_input(
    image: np.ndarray,
    full_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    out_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    y0, y1, x0, x1 = crop_box
    crop = image[y0:y1, x0:x1].copy()
    mask = full_mask[y0:y1, x0:x1].astype(bool)
    if out_size > 0 and (crop.shape[0] != out_size or crop.shape[1] != out_size):
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask.astype(np.uint8), (out_size, out_size), interpolation=cv2.INTER_NEAREST) > 0
    return np.clip(crop, 0.0, 1.0), mask.astype(bool)


def best_rectified_inner_crop(
    image: np.ndarray,
    valid_mask: np.ndarray,
    min_size: int,
    max_side_frac: float,
    min_valid_frac: float,
    safe_border_px: int,
    stride_frac: float,
) -> tuple[tuple[int, int, int, int], dict] | None:
    """Find a sharp, central square surrounded by valid rectified source pixels."""
    h, w = valid_mask.shape
    max_side = int(round(min(h, w) * np.clip(max_side_frac, 0.2, 1.0)))
    min_size = int(max(24, min(min_size, max_side)))
    if max_side < min_size or np.count_nonzero(valid_mask) < min_size * min_size:
        return None

    safe_mask = valid_mask.astype(np.uint8)
    border = int(max(0, safe_border_px))
    if border > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
        safe_mask = cv2.erode(safe_mask, kernel, iterations=1)
    if np.count_nonzero(safe_mask) < min_size * min_size * min_valid_frac:
        safe_mask = valid_mask.astype(np.uint8)
        border = 0

    safe_integral = cv2.integral(safe_mask.astype(np.float32))
    valid_integral = cv2.integral(valid_mask.astype(np.float32))
    rgb8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    sharp = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)) / 255.0
    sharp_integral = cv2.integral(sharp.astype(np.float32))
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
    coarse_lab = cv2.GaussianBlur(lab, (0, 0), sigmaX=12.0, sigmaY=12.0)
    coarse_integrals = [cv2.integral(coarse_lab[..., c]) for c in range(3)]
    coarse_sq_integrals = [cv2.integral(coarse_lab[..., c] ** 2) for c in range(3)]
    highlight_integral = cv2.integral((gray >= 248).astype(np.float32))
    cy_image = 0.5 * h
    cx_image = 0.5 * w

    sizes: list[int] = []
    side = max_side
    while side >= min_size:
        sizes.append(side)
        next_side = int(round(side * 0.88))
        side = next_side if next_side < side else side - 1
    if min_size not in sizes:
        sizes.append(min_size)

    best = None
    best_score = -np.inf
    for side in sizes:
        stride = max(3, int(round(side * max(0.03, stride_frac))))
        y_positions = list(range(0, h - side + 1, stride))
        x_positions = list(range(0, w - side + 1, stride))
        if y_positions[-1] != h - side:
            y_positions.append(h - side)
        if x_positions[-1] != w - side:
            x_positions.append(w - side)
        for y0 in y_positions:
            for x0 in x_positions:
                area = float(side * side)
                safe_frac = integral_rect_sum(safe_integral, y0, x0, side) / area
                valid_frac = integral_rect_sum(valid_integral, y0, x0, side) / area
                if valid_frac < min_valid_frac or safe_frac < min_valid_frac:
                    continue
                sharpness = integral_rect_sum(sharp_integral, y0, x0, side) / area
                coarse_stds = []
                for sum_integral, sq_integral in zip(coarse_integrals, coarse_sq_integrals):
                    mean = integral_rect_sum(sum_integral, y0, x0, side) / area
                    mean_sq = integral_rect_sum(sq_integral, y0, x0, side) / area
                    coarse_stds.append(math.sqrt(max(0.0, mean_sq - mean * mean)))
                lowfreq_std = float(np.linalg.norm(coarse_stds))
                highlight_frac = integral_rect_sum(highlight_integral, y0, x0, side) / area
                cy = y0 + 0.5 * side
                cx = x0 + 0.5 * side
                center_dist = math.hypot((cy - cy_image) / max(h, 1), (cx - cx_image) / max(w, 1))
                size_frac = side / max(max_side, 1)
                score = (
                    2.15 * size_frac
                    + 0.10 * math.log1p(30.0 * sharpness)
                    - 0.075 * lowfreq_std
                    - 1.10 * max(0.0, highlight_frac - 0.08)
                    - 0.45 * center_dist
                )
                if score > best_score:
                    best_score = score
                    best = (
                        (int(y0), int(y0 + side), int(x0), int(x0 + side)),
                        {
                            "inner_crop_side": int(side),
                            "inner_crop_valid_frac": float(valid_frac),
                            "inner_crop_safe_frac": float(safe_frac),
                            "inner_crop_sharpness": float(sharpness),
                            "inner_crop_lowfreq_std": float(lowfreq_std),
                            "inner_crop_highlight_frac": float(highlight_frac),
                            "inner_crop_center_dist": float(center_dist),
                            "inner_crop_safe_border_px": int(border),
                            "inner_crop_score": float(score),
                        },
                    )
    return best


def crop_rectified_chord_input(
    args: argparse.Namespace,
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    crop_info = best_rectified_inner_crop(
        image,
        valid_mask,
        args.rectified_inner_min_size,
        args.rectified_inner_max_side_frac,
        args.rectified_inner_min_valid_frac,
        args.rectified_inner_safe_border_px,
        args.rectified_inner_stride_frac,
    )
    if crop_info is None:
        return None
    crop_box, info = crop_info
    chord_input, mask_crop = make_pure_chord_input(
        image,
        valid_mask,
        crop_box,
        args.chord_input_size,
    )
    info["inner_crop_box_y0_y1_x0_x1"] = [int(v) for v in crop_box]
    info["inner_crop_output_valid_frac"] = float(np.mean(mask_crop))
    return chord_input, mask_crop, info


def rectified_tile_from_view(
    args: argparse.Namespace,
    face: str,
    box: tuple[int, int, int],
    target_mask: np.ndarray,
    pose: proj.ImagePose,
    image: np.ndarray,
    sim: proj.Similarity,
    manifest: dict,
    metas: dict,
    all_faces: list[str],
    da3_views: dict[int, proj.Da3View],
    raw_to_room_matrix: np.ndarray | None,
    caches: dict,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    y0, x0, size_box = [int(v) for v in box]
    tile_h, tile_w = target_mask.shape[:2]
    if tile_h <= 0 or tile_w <= 0:
        return None
    yy, xx = np.mgrid[0:tile_h, 0:tile_w]
    rows = np.clip(y0 + yy.reshape(-1), 0, y0 + size_box - 1).astype(np.int32)
    cols = np.clip(x0 + xx.reshape(-1), 0, x0 + size_box - 1).astype(np.int32)
    size = proj.face_texture_size(args.polygon_source_dir, metas[face])
    pts_da3 = proj.face_points_for_indices(face, rows, cols, size, manifest, metas)
    pts_col = sim.da3_to_colmap(pts_da3)
    u, v, z = proj.project_points(pts_col, pose)
    n = proj.face_normal(face, manifest, metas)
    view = pose.center_da3[None, :] - pts_da3
    dist = np.linalg.norm(view, axis=1)
    cos = np.abs((view / np.maximum(dist[:, None], 1e-8)) @ n)
    in_frame = (
        (z > 1e-6)
        & (u >= 0.0)
        & (v >= 0.0)
        & (u <= pose.width - 1.0)
        & (v <= pose.height - 1.0)
        & (cos >= args.min_view_cos)
    )

    if pose.image_id not in caches["face_id"] or pose.image_id not in caches["zbuffer"]:
        face_id_map, zbuf = proj.build_face_id_and_zbuffer(
            pose,
            all_faces,
            args.polygon_source_dir,
            sim,
            manifest,
            metas,
            args.zbuffer_stride,
        )
        caches["face_id"][pose.image_id] = face_id_map
        caches["zbuffer"][pose.image_id] = zbuf
    if pose.image_id not in caches["reject"]:
        caches["reject"][pose.image_id] = proj.load_view_reject_maps(pose, args)
    if pose.image_id not in caches["depth_calib"] and pose.image_id in da3_views:
        caches["depth_calib"][pose.image_id] = proj.calibrate_da3_depth_to_colmap_zbuffer(
            da3_views[pose.image_id],
            pose,
            caches["zbuffer"][pose.image_id],
        )

    da3_view = da3_views.get(pose.image_id)
    da3_depth_calib = caches["depth_calib"].get(pose.image_id)
    has_depth, camera_depth, projected_depth, surface_distance, sampled_conf = depth_surface_for_pose(
        args,
        pose,
        pts_da3,
        u,
        v,
        z,
        face,
        manifest,
        metas,
        sim,
        da3_view,
        da3_depth_calib,
        raw_to_room_matrix,
    )
    depth_tol = args.depth_abs_tol + args.depth_rel_tol * np.maximum(camera_depth, 0.0)
    depth_diff = np.full(rows.shape, np.inf, dtype=np.float32)
    depth_diff[has_depth] = np.abs(projected_depth[has_depth] - camera_depth[has_depth])
    face_idx = all_faces.index(face)
    face_id_map = caches["face_id"][pose.image_id]
    object_mask, object_risk, boundary_trust = caches["reject"][pose.image_id]
    sampled_face = proj.sample_nearest_map(face_id_map, u, v, border_value=255)
    sampled_object = proj.sample_nearest_map(object_mask, u, v, border_value=0)
    sampled_object_risk = proj.sample_float_map(object_risk, u, v, border_value=0.0)
    sampled_boundary = proj.sample_float_map(boundary_trust, u, v, border_value=1.0)
    if args.surface_distance_tol > 0.0 and args.surface_distance_hard_gate:
        surface_ok = np.isfinite(surface_distance) & (surface_distance <= args.surface_distance_tol)
    else:
        surface_ok = np.ones(rows.shape, dtype=bool)
    valid = (
        target_mask.reshape(-1).astype(bool)
        & in_frame
        & has_depth
        & np.isfinite(depth_diff)
        & (sampled_conf >= args.min_conf)
        & (depth_diff <= depth_tol)
        & (sampled_face == face_idx)
        & (sampled_object == 0)
        & (sampled_object_risk <= args.object_risk_hard_thresh)
        & (sampled_boundary >= args.min_mask_boundary_trust)
        & surface_ok
    )
    valid_mask = valid.reshape(tile_h, tile_w)
    target_count = max(1, int(np.count_nonzero(target_mask)))
    valid_frac = float(np.count_nonzero(valid_mask) / target_count)
    if valid_frac < args.min_rectified_valid_frac:
        return None

    colors = proj.bilinear_sample(image, u.astype(np.float32), v.astype(np.float32)).reshape(tile_h, tile_w, 3)
    if args.chord_input_size > 0 and not args.rectified_inner_crop and (
        tile_h != args.chord_input_size or tile_w != args.chord_input_size
    ):
        colors = cv2.resize(colors, (args.chord_input_size, args.chord_input_size), interpolation=cv2.INTER_CUBIC)
        valid_mask_out = cv2.resize(
            valid_mask.astype(np.uint8),
            (args.chord_input_size, args.chord_input_size),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    elif not args.rectified_inner_crop:
        valid_mask_out = valid_mask
    else:
        valid_mask_out = valid_mask
    info = {
        "rectified_valid_frac": valid_frac,
        "mean_view_cos": float(np.mean(cos[valid])) if np.any(valid) else 0.0,
        "mean_depth_residual": float(np.mean(depth_diff[valid] / np.maximum(depth_tol[valid], 1e-6))) if np.any(valid) else None,
        "mean_surface_distance": float(np.mean(surface_distance[valid])) if np.any(valid) else None,
    }
    return np.clip(colors, 0.0, 1.0), valid_mask_out, info


def write_region_view_inputs(
    args: argparse.Namespace,
    face: str,
    region_i: int,
    box: tuple[int, int, int],
    contributors: list[dict],
    target_tile: np.ndarray,
    target_mask: np.ndarray,
    sim: proj.Similarity,
    manifest: dict,
    metas: dict,
    all_faces: list[str],
    da3_views: dict[int, proj.Da3View],
    raw_to_room_matrix: np.ndarray | None,
    caches: dict,
    out_dirs: dict,
) -> list[dict]:
    total_weight = sum(float(item.get("weight", 0.0)) for item in contributors)
    selected = []
    if args.chord_input_mode == "atlas_rectified":
        evaluated = []
        for rank, item in enumerate(contributors[: max(args.max_source_views_eval, args.max_view_candidates)]):
            if total_weight > 0 and item.get("weight", 0.0) / total_weight < args.min_view_weight_frac * 0.45:
                continue
            pose: proj.ImagePose = item["pose"]
            image = load_rgb(pose.image_path)
            rectified = rectified_tile_from_view(
                args,
                face,
                box,
                target_mask,
                pose,
                image,
                sim,
                manifest,
                metas,
                all_faces,
                da3_views,
                raw_to_room_matrix,
                caches,
            )
            if rectified is None:
                continue
            rectified_full, rectified_mask, extra_info = rectified
            if args.rectified_inner_crop:
                cropped = crop_rectified_chord_input(args, rectified_full, rectified_mask)
                if cropped is None:
                    continue
                chord_input, mask_crop, crop_info = cropped
                extra_info.update(crop_info)
            else:
                chord_input, mask_crop = rectified_full, rectified_mask
            q = (
                2.10 * float(extra_info.get("rectified_valid_frac", 0.0))
                + 0.80 * float(extra_info.get("mean_view_cos", 0.0))
                + 0.55 * float(item.get("weight_frac", 0.0))
                + 0.20 * float(extra_info.get("inner_crop_score", 0.0))
                - 0.35 * float(extra_info.get("mean_depth_residual") or 0.0)
                - 1.20 * max(0.0, args.min_rectified_valid_frac - float(extra_info.get("rectified_valid_frac", 0.0)))
            )
            evaluated.append((q, rank, item, pose, chord_input, mask_crop, extra_info))
        evaluated.sort(key=lambda x: x[0], reverse=True)
        for q, rank, item, pose, chord_input, mask_crop, extra_info in evaluated[: args.max_view_candidates]:
            stem = f"{face}_r{region_i:02d}_v{len(selected):02d}_{Path(pose.name).stem}"
            save_rgb(out_dirs["chord_inputs"] / f"{stem}.png", chord_input)
            save_mask(out_dirs["candidate_masks"] / f"{stem}_mask.png", mask_crop)
            save_rgb(out_dirs["candidate_crops"] / f"{stem}_original_crop.png", chord_input)
            overlay = chord_input.copy()
            overlay[mask_crop] = 0.58 * overlay[mask_crop] + 0.42 * np.array([1.0, 0.08, 0.04], dtype=np.float32)
            save_rgb(out_dirs["candidate_overlays"] / f"{stem}_overlay.png", overlay)
            selected.append(
                {
                    "stem": stem,
                    "type": "view_contributor_rectified",
                    "view_name": pose.name,
                    "image_id": int(pose.image_id),
                    "input_mode": args.chord_input_mode,
                    "selection_score": float(q),
                    "source_rank_by_weight": int(rank),
                    "weight": float(item["weight"]),
                    "weight_frac": float(item.get("weight_frac", 0.0)),
                    "valid_sample_count": int(item["count"]),
                    "crop_box_y0_y1_x0_x1": extra_info.get("inner_crop_box_y0_y1_x0_x1"),
                    "mask_pixels": int(np.count_nonzero(mask_crop)),
                    "mask_fraction_in_chord_input": float(np.mean(mask_crop)),
                    "pure_crop_source_mask_fraction": float(np.mean(mask_crop)),
                    **extra_info,
                    "chord_input": str(out_dirs["chord_inputs"] / f"{stem}.png"),
                    "candidate_mask": str(out_dirs["candidate_masks"] / f"{stem}_mask.png"),
                    "candidate_overlay": str(out_dirs["candidate_overlays"] / f"{stem}_overlay.png"),
                    "mean_depth_residual": float(item["mean_depth_residual"]),
                    "mean_surface_distance": float(item["mean_surface_distance"]),
                }
            )
        if selected or not args.include_atlas_fallback:
            return selected

    for rank, item in enumerate(contributors):
        if len(selected) >= args.max_view_candidates:
            break
        if total_weight > 0 and item.get("weight", 0.0) / total_weight < args.min_view_weight_frac:
            continue
        pose: proj.ImagePose = item["pose"]
        image = load_rgb(pose.image_path)
        view_mask = None
        crop_box = None
        mask_frac = 1.0
        extra_info = {}
        if args.chord_input_mode == "atlas_rectified":
            rectified = rectified_tile_from_view(
                args,
                face,
                box,
                target_mask,
                pose,
                image,
                sim,
                manifest,
                metas,
                all_faces,
                da3_views,
                raw_to_room_matrix,
                caches,
            )
            if rectified is None:
                continue
            rectified_full, rectified_mask, extra_info = rectified
            if args.rectified_inner_crop:
                cropped = crop_rectified_chord_input(args, rectified_full, rectified_mask)
                if cropped is None:
                    continue
                chord_input, mask_crop, crop_info = cropped
                extra_info.update(crop_info)
                crop_box = tuple(extra_info["inner_crop_box_y0_y1_x0_x1"])
            else:
                chord_input, mask_crop = rectified_full, rectified_mask
            original_crop = chord_input.copy()
            mask_frac = float(np.mean(mask_crop))
        else:
            view_mask = np.zeros((pose.height, pose.width), dtype=np.uint8)
            px = np.clip(np.round(item["u"]).astype(np.int32), 0, pose.width - 1)
            py = np.clip(np.round(item["v"]).astype(np.int32), 0, pose.height - 1)
            view_mask[py, px] = 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            view_mask = cv2.dilate(view_mask, k, iterations=1)
            view_mask = cv2.morphologyEx(view_mask, cv2.MORPH_CLOSE, k, iterations=1)
            if int(np.count_nonzero(view_mask)) < args.min_view_mask_pixels:
                continue
            pure_box = best_pure_crop_box(
                view_mask > 0,
                args.min_pure_crop_size,
                args.min_pure_crop_mask_frac,
                args.pure_crop_stride_frac,
            )
            if pure_box is None:
                continue
            y0, y1, x0, x1, mask_frac = pure_box
            crop_box = (y0, y1, x0, x1)
            chord_input, mask_crop = make_pure_chord_input(
                image,
                view_mask > 0,
                crop_box,
                args.chord_input_size,
            )
            if float(np.mean(mask_crop)) < args.min_pure_crop_mask_frac * 0.92:
                continue
            original_crop = chord_input.copy()
        stem = f"{face}_r{region_i:02d}_v{len(selected):02d}_{Path(pose.name).stem}"
        save_rgb(out_dirs["chord_inputs"] / f"{stem}.png", chord_input)
        save_mask(out_dirs["candidate_masks"] / f"{stem}_mask.png", mask_crop)
        save_rgb(out_dirs["candidate_crops"] / f"{stem}_original_crop.png", original_crop)
        overlay = original_crop.copy()
        overlay[mask_crop] = 0.58 * overlay[mask_crop] + 0.42 * np.array([1.0, 0.08, 0.04], dtype=np.float32)
        save_rgb(out_dirs["candidate_overlays"] / f"{stem}_overlay.png", overlay)
        selected.append(
            {
                "stem": stem,
                "type": (
                    "view_contributor_rectified_inner"
                    if args.chord_input_mode == "atlas_rectified" and args.rectified_inner_crop
                    else "view_contributor"
                ),
                "view_name": pose.name,
                "image_id": int(pose.image_id),
                "input_mode": args.chord_input_mode,
                "weight": float(item["weight"]),
                "weight_frac": float(item.get("weight_frac", 0.0)),
                "valid_sample_count": int(item["count"]),
                "crop_box_y0_y1_x0_x1": [int(v) for v in crop_box] if crop_box is not None else None,
                "mask_pixels": int(np.count_nonzero(mask_crop)),
                "mask_fraction_in_chord_input": float(np.mean(mask_crop)),
                "pure_crop_source_mask_fraction": float(mask_frac),
                **extra_info,
                "chord_input": str(out_dirs["chord_inputs"] / f"{stem}.png"),
                "candidate_mask": str(out_dirs["candidate_masks"] / f"{stem}_mask.png"),
                "candidate_overlay": str(out_dirs["candidate_overlays"] / f"{stem}_overlay.png"),
                "mean_depth_residual": float(item["mean_depth_residual"]),
                "mean_surface_distance": float(item["mean_surface_distance"]),
            }
        )
    if selected or not args.include_atlas_fallback:
        return selected

    stem = f"{face}_r{region_i:02d}_atlas_fallback"
    save_rgb(out_dirs["chord_inputs"] / f"{stem}.png", target_tile)
    save_mask(out_dirs["candidate_masks"] / f"{stem}_mask.png", target_mask)
    selected.append(
        {
            "stem": stem,
            "type": "atlas_fallback",
            "view_name": None,
            "image_id": None,
            "weight": 0.0,
            "weight_frac": 0.0,
            "valid_sample_count": int(np.count_nonzero(target_mask)),
            "crop_box_y0_y1_x0_x1": None,
            "mask_pixels": int(np.count_nonzero(target_mask)),
            "chord_input": str(out_dirs["chord_inputs"] / f"{stem}.png"),
            "candidate_mask": str(out_dirs["candidate_masks"] / f"{stem}_mask.png"),
            "candidate_overlay": None,
            "mean_depth_residual": None,
            "mean_surface_distance": None,
        }
    )
    return selected


def prepare_stage(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = {
        "chord_inputs": args.out_dir / "chord_inputs",
        "candidate_masks": args.out_dir / "candidate_masks",
        "candidate_crops": args.out_dir / "candidate_crops",
        "candidate_overlays": args.out_dir / "candidate_overlays",
        "atlas_targets": args.out_dir / "atlas_targets",
        "debug": args.out_dir / "debug",
    }
    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    manifest, metas, faces = manifest_and_faces(args)
    poses, sim, hf_alignment = proj.load_da3_hfalign_poses(args.dataset_dir, args.da3_dir)
    da3_views = proj.load_da3_views(args.da3_dir, poses)
    all_faces = proj.face_names(manifest, None)
    caches = {"zbuffer": {}, "face_id": {}, "reject": {}, "depth_calib": {}}
    stats = []
    for face in faces:
        image = load_rgb(source_image_path(args.source_dir, face))
        masks = load_strict_masks(args, face, image.shape[:2])
        wall_band_audit = {"enabled": False, "bands": []}
        material_cap_audit = {"applied": False}
        if args.material_cluster_discovery:
            candidates, clusters = discover_weighted_material_regions(args, face, image, masks)
            clusters, wall_band_audit = discover_persistent_wall_bands(
                args,
                face,
                image,
                masks,
                clusters,
            )
            clusters, material_cap_audit = apply_integrated_material_cap(args, face, clusters)
            for band in wall_band_audit.get("bands", []):
                print(
                    f"[prepare-wall-band] {face}: rows="
                    f"{band['expanded_top']}:{band['expanded_bottom']} "
                    f"score={band['score']:.3f} coverage={band['coverage']:.3f} "
                    f"confidence={band['confidence']:.3f}",
                    flush=True,
                )
        else:
            candidates, clusters = select_regions(args, face, image, masks)
        face_regions = []
        region_specs = []
        for material_i, cluster in enumerate(clusters):
            exemplars = cluster.get("items", [cluster["representative"]])
            for exemplar_i, exemplar in enumerate(exemplars):
                exemplar_cluster = dict(cluster)
                exemplar_cluster["representative"] = exemplar
                region_specs.append((material_i, exemplar_i, exemplar_cluster))
        for region_i, (material_i, exemplar_i, cluster) in enumerate(region_specs):
            box = tuple(int(v) for v in cluster["representative"]["box"])
            support = representative_support(image.shape[:2], masks, cluster)
            if "material_mask" in cluster:
                support &= cluster["material_mask"]
                if np.count_nonzero(support) < 128:
                    support = cluster["material_mask"] & masks["observed"]
                target_tile, target_mask = target_tile_from_material_cluster(
                    face,
                    image,
                    masks,
                    box,
                    cluster["material_mask"],
                )
                save_mask(out_dirs["debug"] / f"{face}_region_{region_i:02d}_material_mask.png", cluster["material_mask"])
            else:
                target_tile, target_mask = target_tile_from_region(face, image, masks, box)
            target_stem = f"{face}_region_{region_i:02d}"
            save_rgb(out_dirs["atlas_targets"] / f"{target_stem}_target.png", target_tile)
            save_mask(out_dirs["atlas_targets"] / f"{target_stem}_target_mask.png", target_mask)
            save_mask(out_dirs["debug"] / f"{face}_region_{region_i:02d}_support.png", support)
            contributors = trace_region_contributors(
                args,
                face,
                region_i,
                support,
                poses,
                sim,
                manifest,
                metas,
                all_faces,
                da3_views,
                hf_alignment,
                caches,
            )
            view_candidates = write_region_view_inputs(
                args,
                face,
                region_i,
                box,
                contributors,
                target_tile,
                target_mask,
                sim,
                manifest,
                metas,
                all_faces,
                da3_views,
                hf_alignment,
                caches,
                out_dirs,
            )
            is_wall_band = str(cluster.get("discovery_index", "")).startswith("wall_band_")
            if is_wall_band and len(view_candidates) < int(args.wall_band_min_traceable_views):
                print(
                    f"[prepare-wall-band-drop] {face} r{region_i:02d}: "
                    f"traceable_views={len(view_candidates)} "
                    f"required={int(args.wall_band_min_traceable_views)}",
                    flush=True,
                )
                continue
            face_regions.append(
                {
                    "region": int(region_i),
                    "material_id": int(material_i),
                    "exemplar_index": int(exemplar_i),
                    "material_box_purity": float(cluster["representative"].get("material_purity", 1.0)),
                    "box_yx_size": [int(v) for v in box],
                    "cluster_score": float(cluster["score"]),
                    "cluster_items": int(len(cluster["items"])),
                    "mean_lab": [float(x) for x in cluster["mean_lab"]],
                    "edge_mean": float(cluster["edge_mean"]),
                    "sat_mean": float(cluster["sat_mean"]),
                    "material_fraction": float(cluster.get("material_fraction", 1.0)),
                    "discovery_index": cluster.get("discovery_index"),
                    "source": cluster["representative"].get("source", "material_cluster"),
                    "support_texels": int(np.count_nonzero(support)),
                    "target_tile": str(out_dirs["atlas_targets"] / f"{target_stem}_target.png"),
                    "target_mask": str(out_dirs["atlas_targets"] / f"{target_stem}_target_mask.png"),
                    "contributors_total": int(len(contributors)),
                    "contributors_considered": [
                        {
                            "view_name": item["pose"].name,
                            "image_id": int(item["pose"].image_id),
                            "weight": float(item["weight"]),
                            "weight_frac": float(item.get("weight_frac", 0.0)),
                            "count": int(item["count"]),
                        }
                        for item in contributors[:12]
                    ],
                    "view_candidates": view_candidates,
                }
            )
            print(
                f"[prepare] {face} m{material_i:02d} e{exemplar_i:02d} r{region_i:02d}: "
                f"contributors={len(contributors)} chord_inputs={len(view_candidates)} box={box}",
                flush=True,
            )
        stats.append(
            {
                "face": face,
                "shape_hw": [int(image.shape[0]), int(image.shape[1])],
                "candidate_count": int(len(candidates)),
                "material_count": int(len(clusters)),
                "region_count": int(len(face_regions)),
                "persistent_wall_band_discovery": wall_band_audit,
                "integrated_material_cap": material_cap_audit,
                "regions": face_regions,
            }
        )
        print(f"[prepare] {face}: regions={len(face_regions)} candidates={len(candidates)}", flush=True)

    metadata = {
        "method": (
            "chord_view_contributor_atlas_rectified_inner_crop_v1"
            if args.chord_input_mode == "atlas_rectified" and args.rectified_inner_crop
            else "chord_view_contributor_region_inputs_v1"
        ),
        "source_dir": str(args.source_dir),
        "polygon_source_dir": str(args.polygon_source_dir),
        "dataset_dir": str(args.dataset_dir),
        "da3_dir": str(args.da3_dir),
        "object_mask_dir": str(args.object_mask_dir),
        "faces": faces,
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "stats": stats,
    }
    (args.out_dir / "metadata_view_contributor_chord_inputs.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_prepare_contact_sheet(args.out_dir, stats)
    print("[done] prepared Chord inputs:", args.out_dir / "chord_inputs")
    return 0


def masked_stats(image: np.ndarray, mask: np.ndarray) -> dict:
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if np.count_nonzero(mask) < 16:
        mask = np.ones(image.shape[:2], dtype=bool)
    rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.abs(gx) / 255.0
    gradient = np.sqrt(gx * gx + gy * gy) / 255.0
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)) / 255.0
    return {
        "lab_med": np.median(lab[mask], axis=0).astype(np.float32),
        "lab_std": np.std(lab[mask], axis=0).astype(np.float32),
        "rgb_std": np.std(image[mask], axis=0).astype(np.float32),
        "edge": float(np.mean(edge[mask])),
        "gradient": float(np.mean(gradient[mask])),
        "laplacian": float(np.mean(laplacian[mask])),
        "gray_std": float(np.std(gray[mask]) / 255.0),
        "sat": float(np.mean(hsv[..., 1][mask]) / 255.0),
    }


def candidate_score(target_stats: dict, cand_stats: dict, input_stats: dict | None = None, face: str = "") -> float:
    d = target_stats["lab_med"] - cand_stats["lab_med"]
    lab_delta = math.sqrt((float(d[0]) / 30.0) ** 2 + (float(d[1]) / 11.0) ** 2 + (float(d[2]) / 11.0) ** 2)
    std_delta = float(np.linalg.norm(target_stats["lab_std"] - cand_stats["lab_std"]) / 42.0)
    rgb_std_delta = float(np.linalg.norm(target_stats["rgb_std"] - cand_stats["rgb_std"]) / 0.45)
    edge_delta = abs(float(target_stats["edge"]) - float(cand_stats["edge"])) / 0.18
    sat_delta = abs(float(target_stats["sat"]) - float(cand_stats["sat"])) / 0.28
    score = float(lab_delta + 0.22 * std_delta + 0.15 * rgb_std_delta + 0.16 * edge_delta + 0.12 * sat_delta)
    if input_stats is None:
        return score

    input_d = input_stats["lab_med"] - cand_stats["lab_med"]
    input_lab_delta = math.sqrt(
        (float(input_d[0]) / 30.0) ** 2
        + (float(input_d[1]) / 11.0) ** 2
        + (float(input_d[2]) / 11.0) ** 2
    )
    input_sat_delta = abs(float(input_stats["sat"]) - float(cand_stats["sat"])) / 0.28
    score += 0.48 * input_lab_delta + 0.08 * input_sat_delta

    min_retention = 0.55 if face == "floor" else 0.28
    retention_weight = 2.4 if face == "floor" else 0.55
    retentions = []
    for key in ("gradient", "laplacian", "gray_std"):
        source_value = max(float(input_stats[key]), 1e-4)
        retentions.append(float(cand_stats[key]) / source_value)
    collapse = float(np.mean([max(0.0, min_retention - value) / min_retention for value in retentions]))
    score += retention_weight * collapse
    return float(score)


def robust_materialize_tile(face: str, tile: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Keep Chord intrinsic basecolor intact after filling any rejected pixels."""
    tile = np.clip(tile.astype(np.float32), 0.0, 1.0)
    if mask is None:
        mask = np.ones(tile.shape[:2], dtype=bool)
    elif mask.shape != tile.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if np.count_nonzero(mask) < 64:
        mask = np.ones(tile.shape[:2], dtype=bool)

    return np.clip(gmp.inpaint_tile_holes(tile, mask), 0.0, 1.0)


def tile_from_chord_candidate(face: str, chord_tile: np.ndarray, mask: np.ndarray | None, target_shape: tuple[int, int]) -> np.ndarray:
    tile = chord_tile.astype(np.float32)
    if mask is not None:
        if mask.shape != tile.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if np.count_nonzero(mask) >= 64:
            tile = gmp.inpaint_tile_holes(tile, mask)
    if tile.shape[:2] != target_shape:
        tile = cv2.resize(tile, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    tile = robust_materialize_tile(face, tile, mask)
    return gmp.make_seamless_tile(np.clip(tile, 0.0, 1.0))


def tile_from_chord_pbr_channel(chord_tile: np.ndarray, mask: np.ndarray | None, target_shape: tuple[int, int]) -> np.ndarray:
    tile = np.clip(chord_tile.astype(np.float32), 0.0, 1.0)
    if mask is not None:
        if mask.shape != tile.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if np.count_nonzero(mask) >= 64:
            tile = gmp.inpaint_tile_holes(tile, mask)
    if tile.shape[:2] != target_shape:
        tile = cv2.resize(tile, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
    return gmp.make_seamless_tile(np.clip(tile, 0.0, 1.0))


def build_face_masks_for_compose(args: argparse.Namespace, face: str, image: np.ndarray) -> dict:
    masks = load_strict_masks(args, face, image.shape[:2])
    masks["material_ref"] = masks["source"] | masks["high"]
    return masks


def compose_stage(args: argparse.Namespace) -> int:
    # Fix of a latent NameError in the frozen v3b copy: compose_stage used
    # `chord_output_dir` without binding it (the forStructure fork binds it the same way).
    if args.chord_output_dir is None:
        raise ValueError("--chord-output-dir is required for --stage compose")
    chord_output_dir = args.chord_output_dir
    keys = pbr_keys(args)  # same latent-unbound fix; only used in the compose manifest
    meta_path = args.out_dir / "metadata_view_contributor_chord_inputs.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    prior_dir = args.out_dir / "priors"
    region_dir = args.out_dir / "region_priors"
    territory_dir = args.out_dir / "territory"
    debug_dir = args.out_dir / "debug"
    for path in (prior_dir, region_dir, territory_dir, debug_dir):
        path.mkdir(parents=True, exist_ok=True)

    stats_out = []
    for face_info in metadata["stats"]:
        face = face_info["face"]
        image = load_rgb(source_image_path(args.source_dir, face))
        masks = build_face_masks_for_compose(args, face, image)
        shape = image.shape[:2]
        face_region_priors = []
        face_supports = []
        face_clusters = []
        region_stats = []
        (region_dir / face).mkdir(parents=True, exist_ok=True)
        for region in face_info["regions"]:
            region_i = int(region["region"])
            box = tuple(int(v) for v in region["box_yx_size"])
            y, x, size = box
            support = load_gray(debug_dir / f"{face}_region_{region_i:02d}_support.png", shape, 0.0) > 0.5
            target_tile = load_rgb(Path(region["target_tile"]))
            target_mask = load_gray(Path(region["target_mask"]), target_tile.shape[:2], 1.0) > 0.5
            target_stats = masked_stats(target_tile, target_mask)
            scored = []
            for candidate in region["view_candidates"]:
                stem = candidate["stem"]
                chord_tile = load_chord_map(chord_output_dir, stem, args.basecolor_key)
                if chord_tile is None:
                    continue
                mask = load_gray(Path(candidate["candidate_mask"]), chord_tile.shape[:2], 1.0) > 0.5
                scored_tile = tile_from_chord_candidate(face, chord_tile, mask, target_tile.shape[:2])
                stats = masked_stats(scored_tile, np.ones(scored_tile.shape[:2], dtype=bool))
                input_tile = load_rgb(Path(candidate["chord_input"]))
                input_mask = load_gray(Path(candidate["candidate_mask"]), input_tile.shape[:2], 1.0) > 0.5
                input_stats = masked_stats(input_tile, input_mask)
                score = candidate_score(target_stats, stats, input_stats, face)
                if candidate.get("type") == "atlas_fallback":
                    score += 0.15
                scored.append(
                    {
                        "candidate": candidate,
                        "score": float(score),
                        "tile": scored_tile,
                        "raw_tile": chord_tile,
                        "mask": mask,
                        "stats": {
                            "lab_med": [float(v) for v in stats["lab_med"]],
                            "lab_std": [float(v) for v in stats["lab_std"]],
                            "edge": float(stats["edge"]),
                            "gradient": float(stats["gradient"]),
                            "laplacian": float(stats["laplacian"]),
                            "gray_std": float(stats["gray_std"]),
                            "sat": float(stats["sat"]),
                        },
                        "input_stats": {
                            "gradient": float(input_stats["gradient"]),
                            "laplacian": float(input_stats["laplacian"]),
                            "gray_std": float(input_stats["gray_std"]),
                        },
                    }
                )
            if face == "floor" and scored:
                detail_values = [
                    item["stats"]["gradient"]
                    + 0.65 * item["stats"]["laplacian"]
                    + 0.20 * item["stats"]["gray_std"]
                    for item in scored
                ]
                max_detail = max(max(detail_values), 1e-6)
                for item, detail in zip(scored, detail_values):
                    relative_detail = detail / max_detail
                    item["score"] += 0.90 * max(0.0, 0.45 - relative_detail) / 0.45
                    item["relative_detail"] = float(relative_detail)
            if not scored:
                stem = f"{face}_r{region_i:02d}_missing_chord_fallback"
                chord_tile = target_tile
                mask = target_mask
                chosen = {
                    "candidate": {"stem": stem, "type": "target_tile_fallback", "candidate_mask": region["target_mask"]},
                    "score": 999.0,
                    "tile": chord_tile,
                    "mask": mask,
                    "stats": {},
                }
                scored = [chosen]
            scored.sort(key=lambda item: item["score"])
            chosen = scored[0]
            material_tile = np.clip(chosen["tile"], 0.0, 1.0)
            cluster_masks = dict(masks)
            cluster_masks["material_ref"] = support
            prior = gmp.build_full_prior(face, image, cluster_masks, material_tile, box)
            face_region_priors.append(prior)
            face_supports.append(support)
            cluster = {
                "score": float(region["cluster_score"]),
                "mean_lab": np.asarray(region["mean_lab"], dtype=np.float32),
                "representative": {"box": box, "score": float(region["cluster_score"])},
                "items": [{"box": box, "score": float(region["cluster_score"])}],
            }
            face_clusters.append(cluster)
            save_rgb(region_dir / face / f"region_{region_i:02d}.png", prior)
            save_rgb(region_dir / face / f"region_{region_i:02d}_chosen_tile.png", material_tile)
            save_rgb(region_dir / face / f"region_{region_i:02d}_target_tile.png", target_tile)
            save_mask(debug_dir / f"{face}_region_{region_i:02d}_support.png", support)
            for scored_i, item in enumerate(scored):
                save_rgb(region_dir / face / f"region_{region_i:02d}_candidate_{scored_i:02d}_{item['candidate']['stem']}_materialized.png", item["tile"])
                if "raw_tile" in item:
                    save_rgb(region_dir / face / f"region_{region_i:02d}_candidate_{scored_i:02d}_{item['candidate']['stem']}_basecolor_raw.png", item["raw_tile"])
            region_stats.append(
                {
                    "region": region_i,
                    "box_yx_size": [int(v) for v in box],
                    "support_texels": int(np.count_nonzero(support)),
                    "chosen_stem": chosen["candidate"]["stem"],
                    "chosen_type": chosen["candidate"].get("type"),
                    "chosen_score": float(chosen["score"]),
                    "candidate_scores": [
                        {
                            "stem": item["candidate"]["stem"],
                            "type": item["candidate"].get("type"),
                            "score": float(item["score"]),
                            "view_name": item["candidate"].get("view_name"),
                            "weight_frac": item["candidate"].get("weight_frac"),
                            "relative_detail": item.get("relative_detail"),
                            "gradient_retention": (
                                float(item["stats"]["gradient"] / max(item["input_stats"]["gradient"], 1e-6))
                                if "input_stats" in item
                                else None
                            ),
                            "laplacian_retention": (
                                float(item["stats"]["laplacian"] / max(item["input_stats"]["laplacian"], 1e-6))
                                if "input_stats" in item
                                else None
                            ),
                        }
                        for item in scored
                    ],
                }
            )
            print(
                f"[compose] {face} r{region_i:02d}: chosen={chosen['candidate']['stem']} "
                f"score={chosen['score']:.3f}",
                flush=True,
            )

        if not face_region_priors:
            face_region_priors = [image]
            face_supports = [masks["source"] | masks["high"]]
            face_clusters = [{"score": 0.0, "mean_lab": np.array([128, 128, 128], dtype=np.float32)}]
        composite, weights = mm.blend_region_priors(
            args,
            face,
            image,
            masks,
            face_clusters,
            face_region_priors,
            face_supports,
            region_valids=None,
        )
        save_rgb(prior_dir / f"{face}.png", composite)
        save_rgb(debug_dir / f"{face}_view_chord_composite.png", composite)
        save_rgb(debug_dir / f"{face}_raw_projected.png", image)
        labels = hard_labels(weights, masks["observed"])
        npy_dir = territory_dir / "assignment_labels_npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / f"{face}_labels.npy", labels.astype(np.int16))
        save_rgb(territory_dir / f"{face}_territory_overlay.png", label_overlay(image, labels, masks["observed"]))
        hard = hard_composite(face_region_priors, labels)
        save_rgb(territory_dir / "textures" / f"{face}.png", hard)
        for idx, weight in enumerate(weights):
            save_mask(debug_dir / f"{face}_region_{idx:02d}_blend_weight.png", weight > 0.5 * np.max(weight))
        stats_out.append(
            {
                "face": face,
                "shape_hw": [int(shape[0]), int(shape[1])],
                "prior_count": int(len(region_stats)),
                "regions": region_stats,
            }
        )

    out_meta = {
        "method": "chord_view_contributor_scored_material_territories_v1",
        "summary": (
            "Each strict high-weight atlas region is traced back to original contributing images. "
            "Chord runs on masked source pixels rectified into face/atlas coordinates, not raw "
            "perspective crops. For each atlas region, all Chord candidates are scored against "
            "the strict projected atlas target; the best candidate becomes that region material, "
            "and regions expand/compete over each face."
        ),
        "source_dir": str(args.source_dir),
        "chord_output_dir": str(chord_output_dir),
        "pbr_keys": keys,
        "stats": stats_out,
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.out_dir / "metadata_view_contributor_chord_materials.json").write_text(
        json.dumps(out_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_compose_contact_sheets(args.out_dir, out_meta)
    print("[done] composed view-contributor Chord materials:", args.out_dir)
    return 0


def hard_labels(weights: list[np.ndarray], valid: np.ndarray) -> np.ndarray:
    if not weights:
        return np.full(valid.shape, -1, dtype=np.int16)
    stack = np.stack(weights, axis=0)
    labels = np.argmax(stack, axis=0).astype(np.int16)
    labels[~valid] = -1
    return labels


def label_color(idx: int) -> np.ndarray:
    palette = np.asarray(
        [
            [235, 95, 87],
            [75, 166, 232],
            [112, 193, 112],
            [245, 183, 76],
            [168, 124, 220],
            [86, 205, 185],
            [230, 118, 180],
            [180, 180, 76],
        ],
        dtype=np.float32,
    ) / 255.0
    return palette[idx % len(palette)]


def label_overlay(image: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = image.copy()
    color = np.zeros_like(image)
    for idx in sorted(int(x) for x in np.unique(labels) if x >= 0):
        color[labels == idx] = label_color(idx)
    out[valid] = 0.55 * out[valid] + 0.45 * color[valid]
    out[~valid] *= 0.28
    return np.clip(out, 0.0, 1.0)


def hard_composite(priors: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(priors[0])
    for idx, prior in enumerate(priors):
        mask = labels == idx
        out[mask] = prior[mask]
    return np.clip(out, 0.0, 1.0)


def text_tile(label: str, image: np.ndarray, size: tuple[int, int] = (230, 270)) -> Image.Image:
    im = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).convert("RGB")
    im.thumbnail((size[0] - 14, size[1] - 38))
    tile = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 8), label[:46], fill=(20, 20, 20))
    tile.paste(im, ((size[0] - im.width) // 2, 32))
    return tile


def write_prepare_contact_sheet(out_dir: Path, stats: list[dict]) -> None:
    tiles = []
    for face in stats:
        for region in face["regions"]:
            target = load_rgb(Path(region["target_tile"]))
            tiles.append(text_tile(f"{face['face']} r{region['region']} target", target))
            for cand in region["view_candidates"][:3]:
                path = Path(cand["chord_input"])
                if path.exists():
                    tiles.append(text_tile(f"{face['face']} r{region['region']} {cand['stem']}", load_rgb(path)))
    if not tiles:
        return
    cols = 4
    rows = int(math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 230, rows * 270), (235, 235, 235))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 230, (i // cols) * 270))
    sheet.save(out_dir / "prepared_chord_inputs_preview.jpg", quality=92)


def write_compose_contact_sheets(out_dir: Path, metadata: dict) -> None:
    tiles = []
    for face in metadata["stats"]:
        for region in face["regions"]:
            face_name = face["face"]
            rid = int(region["region"])
            chosen = out_dir / "region_priors" / face_name / f"region_{rid:02d}_chosen_tile.png"
            target = out_dir / "region_priors" / face_name / f"region_{rid:02d}_target_tile.png"
            if target.exists():
                tiles.append(text_tile(f"{face_name} r{rid} target", load_rgb(target)))
            if chosen.exists():
                tiles.append(text_tile(f"{face_name} r{rid} chosen {region['chosen_score']:.2f}", load_rgb(chosen)))
    if tiles:
        cols = 4
        rows = int(math.ceil(len(tiles) / cols))
        sheet = Image.new("RGB", (cols * 230, rows * 270), (235, 235, 235))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * 230, (i // cols) * 270))
        sheet.save(out_dir / "chosen_material_tiles_preview.jpg", quality=92)

    face_tiles = []
    for face in metadata["stats"]:
        face_name = face["face"]
        prior = out_dir / "priors" / f"{face_name}.png"
        territory = out_dir / "territory" / f"{face_name}_territory_overlay.png"
        hard = out_dir / "territory" / "textures" / f"{face_name}.png"
        if prior.exists():
            face_tiles.append(text_tile(f"{face_name} soft material", load_rgb(prior), (360, 250)))
        if territory.exists():
            face_tiles.append(text_tile(f"{face_name} territory labels", load_rgb(territory), (360, 250)))
        if hard.exists():
            face_tiles.append(text_tile(f"{face_name} hard territory", load_rgb(hard), (360, 250)))
    if face_tiles:
        cols = 3
        rows = int(math.ceil(len(face_tiles) / cols))
        sheet = Image.new("RGB", (cols * 360, rows * 250), (235, 235, 235))
        for i, tile in enumerate(face_tiles):
            sheet.paste(tile, ((i % cols) * 360, (i // cols) * 250))
        sheet.save(out_dir / "material_territory_preview.jpg", quality=92)


def main() -> int:
    args = parse_args()
    if args.stage == "prepare":
        return prepare_stage(args)
    return compose_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
