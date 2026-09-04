import argparse
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


FACE_NAMES = ["floor", "ceiling", "wall_00", "wall_01", "wall_02", "wall_03"]
WALL_FACE_NAMES = ["wall_00", "wall_01", "wall_02", "wall_03"]
MODEL_FACE_NAMES = ["floor", "ceiling", "wall_00", "wall_01", "wall_02", "wall_03"]

FACE_PROMPTS = {
    "floor": "top down clean empty room light wood floor material, realistic indoor floor, no furniture, no object, no shadow",
    "ceiling": "plain off white indoor ceiling material, clean empty room ceiling, no lamp, no object, no shadow",
    "wall_00": "plain painted indoor wall material, empty room wall, no furniture, no frame, no objects, no shadows",
    "wall_01": "plain painted indoor wall material, empty room wall, no furniture, no frame, no objects, no shadows",
    "wall_02": "plain painted indoor wall material, empty room wall, no furniture, no frame, no objects, no shadows",
    "wall_03": "plain painted indoor wall material, empty room wall, no furniture, no frame, no objects, no shadows",
}
NEGATIVE_PROMPT = (
    "furniture, chair, table, cabinet, bookshelf, bed, curtain, window, door, "
    "person, object, clutter, poster, painting, frame, lamp, cable, shadow, "
    "text, logo, watermark, distorted perspective, blurry, high contrast object"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Full structured completion for empty-room atlases: reliable texels are locked, "
            "medium-confidence abnormal texels are lightly repaired with a material prior, "
            "and true holes are completed per face with seam-aware blending."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed-texture", choices=["raw", "texture"], default="raw")
    parser.add_argument(
        "--faces",
        default=None,
        help="Comma-separated face list. Defaults to textures in --input-dir, then the historical six faces.",
    )
    parser.add_argument("--surface-contaminant-mask-dir", type=Path, default=None)
    parser.add_argument("--surface-contaminant-dilate-px", type=int, default=8)
    parser.add_argument("--material-prior-dir", type=Path, default=None)
    parser.add_argument("--prepare-material-tiles-only", action="store_true")

    parser.add_argument("--keep-valid-views", type=int, default=3)
    parser.add_argument("--candidate-valid-views", type=int, default=2)
    parser.add_argument("--min-valid-views", type=int, default=1)
    parser.add_argument("--keep-clean-thresh", type=float, default=0.58)
    parser.add_argument("--source-clean-thresh", type=float, default=0.43)
    parser.add_argument("--fill-clean-thresh", type=float, default=0.16)
    parser.add_argument("--keep-object-risk-thresh", type=float, default=0.48)
    parser.add_argument("--source-object-risk-thresh", type=float, default=0.66)
    parser.add_argument("--fill-object-risk-thresh", type=float, default=0.82)
    parser.add_argument("--mask-boundary-keep-thresh", type=float, default=0.46)
    parser.add_argument("--footprint-keep-min", type=float, default=0.12)

    parser.add_argument("--floor-tile-size", type=int, default=384)
    parser.add_argument("--ceiling-tile-size", type=int, default=320)
    parser.add_argument("--wall-tile-size", type=int, default=256)
    parser.add_argument("--tile-stride", type=int, default=24)
    parser.add_argument("--floor-min-source-fraction", type=float, default=0.84)
    parser.add_argument("--ceiling-min-source-fraction", type=float, default=0.70)
    parser.add_argument("--wall-min-source-fraction", type=float, default=0.62)
    parser.add_argument("--floor-anomaly-percentile", type=float, default=91.0)
    parser.add_argument("--ceiling-anomaly-percentile", type=float, default=89.0)
    parser.add_argument("--wall-anomaly-percentile", type=float, default=92.0)
    parser.add_argument("--medium-max-alpha", type=float, default=0.56)
    parser.add_argument("--hole-feather-px", type=int, default=34)
    parser.add_argument("--seam-band-px", type=int, default=14)
    parser.add_argument("--seam-smooth-px", type=int, default=9)

    parser.add_argument("--wall-ring-px", type=int, default=0)
    parser.add_argument("--wall-mask-dilate-px", type=int, default=2)
    parser.add_argument("--wall-telea-radius", type=int, default=4)
    parser.add_argument("--wall-hole-feather-px", type=int, default=28)
    parser.add_argument("--wall-prior-blur-sigma", type=float, default=2.2)
    parser.add_argument("--wall-prior-max-std-ratio", type=float, default=1.35)
    parser.add_argument("--wall-material-anomaly-thresh", type=float, default=2.35)
    parser.add_argument("--wall-material-anomaly-alpha", type=float, default=0.72)
    parser.add_argument("--max-model-mask-ratio", type=float, default=1.00)
    parser.add_argument(
        "--max-model-component-ratio",
        type=float,
        default=0.035,
        help=(
            "Connected hole components at or below this face-area ratio are marked as small; "
            "larger components are still sent to the model, but receive stronger material-prior guidance."
        ),
    )
    parser.add_argument("--small-brushnet-beta", type=float, default=0.88)
    parser.add_argument("--large-brushnet-beta", type=float, default=0.72)
    parser.add_argument("--small-brushnet-min-beta", type=float, default=0.45)
    parser.add_argument("--large-brushnet-min-beta", type=float, default=0.30)
    parser.add_argument("--material-loss-anomaly-scale", type=float, default=0.42)
    parser.add_argument("--material-loss-prior-scale", type=float, default=0.30)
    parser.add_argument("--material-loss-prior-start", type=float, default=0.95)
    parser.add_argument("--material-delta-gate-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--material-delta-gate-thresh",
        type=float,
        default=0.72,
        help="LAB-normalized BrushNet-vs-material-prior delta where the material prior starts to dominate.",
    )
    parser.add_argument(
        "--material-delta-gate-soft-width",
        type=float,
        default=0.18,
        help="Soft transition width above the material delta gate threshold.",
    )
    parser.add_argument(
        "--lock-all-high",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preserve all high-weight texels exactly after completion.",
    )
    parser.add_argument("--face-material-outlier-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--face-material-source-reliability", type=float, default=0.44)
    parser.add_argument("--face-material-inlier-percentile", type=float, default=90.0)
    parser.add_argument("--face-material-outlier-dilate-px", type=int, default=2)
    parser.add_argument("--face-material-outlier-max-ratio", type=float, default=0.34)

    parser.add_argument(
        "--wall-model-backend",
        choices=["none", "opencv", "diffusers", "brushnet"],
        default="diffusers",
    )
    parser.add_argument(
        "--model-id",
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Diffusers inpainting model for the wall-hole candidate.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=3.2)
    parser.add_argument("--strength", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--allow-model-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--brushnet-path",
        type=Path,
        default=Path("models/brushnet/segmentation_mask_brushnet_ckpt"),
    )
    parser.add_argument(
        "--brushnet-base-model",
        type=Path,
        default=Path("models/brushnet/sd15"),
    )
    parser.add_argument("--brushnet-conditioning-scale", type=float, default=1.0)
    return parser.parse_args()


def parse_faces(value, input_dir=None):
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    if input_dir is not None:
        texture_dir = Path(input_dir) / "textures"
        if texture_dir.exists():
            faces = sorted(path.stem for path in texture_dir.glob("*.png"))
            ordered = [face for face in ("floor", "ceiling") if face in faces]
            ordered.extend(sorted(face for face in faces if face.startswith("wall_")))
            seen = set(ordered)
            ordered.extend(sorted(face for face in faces if face not in seen))
            if ordered:
                return ordered
    return list(FACE_NAMES)


def source_image_path(input_dir, face, mode):
    if mode == "raw":
        raw = input_dir / "debug" / f"{face}_raw_projected.png"
        if raw.exists():
            return raw
    return input_dir / "textures" / f"{face}.png"


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def load_map(debug_dir, face, name, shape, default):
    path = debug_dir / f"{face}_{name}.npy"
    if path.exists():
        data = np.load(path).astype(np.float32)
        if data.shape != shape:
            data = cv2.resize(data, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return data
    return np.full(shape, default, dtype=np.float32)


def dilate_bool(mask, radius):
    if radius <= 0 or not np.any(mask):
        return mask.astype(bool)
    k = 2 * radius + 1
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1) > 0


def erode_bool(mask, radius):
    if radius <= 0 or not np.any(mask):
        return mask.astype(bool)
    k = 2 * radius + 1
    return cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1) > 0


def load_surface_contaminant(face, args, shape):
    if args.surface_contaminant_mask_dir is None:
        return np.zeros(shape, dtype=bool)
    candidates = [
        args.surface_contaminant_mask_dir / f"{face}_contaminant_mask.png",
        args.surface_contaminant_mask_dir / f"{face}.png",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return np.zeros(shape, dtype=bool)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read contaminant mask: {path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return dilate_bool(mask > 0, args.surface_contaminant_dilate_px)


def compute_quality_masks(args, face, shape):
    debug_dir = args.input_dir / "debug"
    valid_count = load_map(debug_dir, face, "valid_count", shape, 0.0)
    candidate_count = load_map(debug_dir, face, "candidate_count", shape, 0.0)
    clean_score = load_map(debug_dir, face, "clean_score", shape, 1.0)
    object_risk = load_map(debug_dir, face, "object_risk", shape, 0.0)
    weight_sum = load_map(debug_dir, face, "weight_sum", shape, 0.0)
    mask_boundary_trust = load_map(debug_dir, face, "mask_boundary_trust", shape, 1.0)
    footprint_area = load_map(debug_dir, face, "footprint_area", shape, 1.0)
    contaminant = load_surface_contaminant(face, args, shape)

    observed = valid_count >= args.min_valid_views
    high = (
        (valid_count >= args.keep_valid_views)
        & (clean_score >= args.keep_clean_thresh)
        & (object_risk <= args.keep_object_risk_thresh)
        & (mask_boundary_trust >= args.mask_boundary_keep_thresh)
        & (footprint_area >= args.footprint_keep_min)
        & (~contaminant)
    )
    source = (
        (valid_count >= args.candidate_valid_views)
        & (clean_score >= args.source_clean_thresh)
        & (object_risk <= args.source_object_risk_thresh)
        & (mask_boundary_trust >= max(0.20, args.mask_boundary_keep_thresh - 0.18))
        & (~dilate_bool(contaminant, max(1, args.surface_contaminant_dilate_px // 2)))
    )
    if np.count_nonzero(source) < 0.010 * source.size:
        source = (
            (valid_count >= args.min_valid_views)
            & (clean_score >= max(0.24, args.source_clean_thresh - 0.16))
            & (object_risk <= 0.78)
            & (~contaminant)
        )
    source = erode_bool(source, 1)
    fill = (
        (valid_count < args.min_valid_views)
        | (object_risk >= args.fill_object_risk_thresh)
        | ((clean_score < args.fill_clean_thresh) & (candidate_count > 0))
        | contaminant
    )
    mid = observed & (~high) & (~fill)

    positive_weight = weight_sum[weight_sum > 1e-8]
    weight_scale = float(np.percentile(positive_weight, 90.0)) if positive_weight.size else 1.0
    count_rel = np.clip(valid_count / max(1, args.keep_valid_views), 0.0, 1.0)
    weight_rel = np.clip(weight_sum / max(weight_scale, 1e-8), 0.0, 1.0)
    footprint_rel = np.clip(footprint_area / max(args.footprint_keep_min, 1e-6), 0.0, 1.0)
    reliability = (
        np.sqrt(count_rel * weight_rel)
        * clean_score
        * (1.0 - 0.62 * object_risk)
        * np.sqrt(np.clip(mask_boundary_trust, 0.0, 1.0))
        * np.sqrt(footprint_rel)
    )
    reliability = np.clip(reliability, 0.0, 1.0).astype(np.float32)
    reliability[~observed] = 0.0
    return {
        "valid_count": valid_count,
        "candidate_count": candidate_count,
        "clean_score": clean_score,
        "object_risk": object_risk,
        "weight_sum": weight_sum,
        "mask_boundary_trust": mask_boundary_trust,
        "footprint_area": footprint_area,
        "contaminant": contaminant,
        "observed": observed,
        "high": high,
        "source": source,
        "fill": fill,
        "mid": mid,
        "reliability": reliability,
    }


def integral_sum(mask):
    return cv2.integral(mask.astype(np.float32))[1:, 1:]


def rect_sum(integral, y0, x0, size):
    y1 = y0 + size - 1
    x1 = x0 + size - 1
    total = integral[y1, x1]
    if y0 > 0:
        total -= integral[y0 - 1, x1]
    if x0 > 0:
        total -= integral[y1, x0 - 1]
    if y0 > 0 and x0 > 0:
        total += integral[y0 - 1, x0 - 1]
    return float(total)


def edge_energy(image):
    gray = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy) / 255.0


def color_anomaly_score(image, reference_mask, percentile):
    if np.count_nonzero(reference_mask) < 128:
        return np.zeros(image.shape[:2], dtype=np.float32), np.zeros(image.shape[:2], dtype=bool)
    lab = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    src = lab[reference_mask]
    median = np.median(src, axis=0)
    mad = np.median(np.abs(src - median[None, :]), axis=0) + np.array([5.0, 3.5, 3.5], dtype=np.float32)
    dist = np.sqrt(np.sum(((lab - median[None, None, :]) / mad[None, None, :]) ** 2, axis=2))
    cap = max(float(np.percentile(dist[reference_mask], percentile)), 2.8)
    luma = lab[..., 0]
    low_l = float(np.percentile(src[:, 0], 1.5))
    high_l = float(np.percentile(src[:, 0], 98.5))
    score = np.maximum(dist / cap - 1.0, 0.0)
    score = np.maximum(score, np.maximum((low_l - luma - 6.0) / 18.0, 0.0))
    score = np.maximum(score, np.maximum((luma - high_l - 10.0) / 22.0, 0.0))
    return score.astype(np.float32), score > 0.0


def apply_face_material_identity(args, face, image, masks):
    if not getattr(args, "face_material_outlier_enable", True):
        masks["face_material_outlier"] = np.zeros(image.shape[:2], dtype=bool)
        masks["material_ref"] = masks["source"] | masks["high"]
        return masks

    h, w = image.shape[:2]
    edge = cv2.GaussianBlur(edge_energy(image), (0, 0), 1.0)
    hsv = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    ref = (
        masks["source"]
        & (masks["reliability"] >= getattr(args, "face_material_source_reliability", 0.44))
        & (masks["object_risk"] <= (0.34 if face == "floor" else 0.24))
        & (masks["mask_boundary_trust"] >= (0.44 if face == "floor" else 0.55))
        & (~masks["contaminant"])
    )
    if face.startswith("wall") or face == "ceiling":
        ref_edge = edge[ref]
        ref_sat = sat[ref]
        edge_cap = float(np.percentile(ref_edge, 68.0)) if ref_edge.size else 0.07
        sat_cap = float(np.percentile(ref_sat, 72.0)) if ref_sat.size else 0.22
        ref &= (edge <= max(0.055, edge_cap)) & (sat <= max(0.16, sat_cap + 0.04))
    if np.count_nonzero(ref) < max(512, int(0.002 * h * w)):
        ref = masks["source"] & (~masks["contaminant"])
    if np.count_nonzero(ref) < 256:
        masks["face_material_outlier"] = np.zeros((h, w), dtype=bool)
        masks["material_ref"] = masks["source"] | masks["high"]
        return masks

    lab = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    src = lab[ref]
    if src.shape[0] > 250000:
        step = max(1, src.shape[0] // 250000)
        src = src[::step]
    median = np.median(src, axis=0)
    mad = np.median(np.abs(src - median[None, :]), axis=0) + np.array([8.0, 4.0, 4.0], dtype=np.float32)
    dist = np.sqrt(np.sum(((lab - median[None, None, :]) / mad[None, None, :]) ** 2, axis=2))
    ref_dist = dist[ref]
    if ref_dist.size:
        floor_cap = 4.6 if face == "floor" else 3.2
        cap = max(floor_cap, float(np.percentile(ref_dist, getattr(args, "face_material_inlier_percentile", 90.0))))
    else:
        cap = 4.6 if face == "floor" else 3.2

    ref_sat = sat[ref]
    sat_cap = float(np.percentile(ref_sat, 92.0)) + (0.10 if face == "floor" else 0.055) if ref_sat.size else 0.35
    ref_l = lab[ref, 0]
    low_l = float(np.percentile(ref_l, 1.5)) if ref_l.size else 0.0
    high_l = float(np.percentile(ref_l, 98.5)) if ref_l.size else 255.0
    chroma_outlier = (dist > cap) | ((sat > sat_cap) & (dist > 0.58 * cap))
    luma_outlier = (lab[..., 0] < low_l - 9.0) | (lab[..., 0] > high_l + 12.0)
    suspicious_support = (
        (masks["reliability"] < (0.90 if face == "floor" else 0.96))
        | (masks["object_risk"] > (0.20 if face == "floor" else 0.12))
        | (masks["mask_boundary_trust"] < (0.70 if face == "floor" else 0.78))
        | (edge > (0.11 if face == "floor" else 0.065))
        | (sat > sat_cap)
    )
    outlier = masks["observed"] & (~masks["contaminant"]) & (chroma_outlier | luma_outlier) & suspicious_support
    outlier &= ~erode_bool(ref, 1)
    max_count = int(max(0.0, getattr(args, "face_material_outlier_max_ratio", 0.34)) * h * w)
    if max_count > 0 and np.count_nonzero(outlier) > max_count:
        score = dist + 0.75 * np.maximum(sat - sat_cap, 0.0)
        values = score[outlier]
        thresh = float(np.partition(values, max(0, values.size - max_count))[max(0, values.size - max_count)])
        outlier = outlier & (score >= thresh)
    outlier = dilate_bool(outlier, getattr(args, "face_material_outlier_dilate_px", 2))

    masks["face_material_outlier"] = outlier
    masks["source"] = masks["source"] & (~outlier)
    masks["high"] = masks["high"] & (~outlier)
    masks["fill"] = masks["fill"] | outlier
    masks["mid"] = masks["observed"] & (~masks["high"]) & (~masks["fill"])
    masks["reliability"] = masks["reliability"].copy()
    masks["reliability"][outlier] = 0.0
    masks["material_ref"] = (masks["source"] | masks["high"]) & (~outlier)
    return masks


def material_prior_delta(image, prior):
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    prior_u8 = np.clip(prior * 255.0, 0, 255).astype(np.uint8)
    image_lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    prior_lab = cv2.cvtColor(prior_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = image_lab - prior_lab
    return np.sqrt((diff[..., 0] / 30.0) ** 2 + (diff[..., 1] / 11.0) ** 2 + (diff[..., 2] / 11.0) ** 2)


def normalize_tile_lighting(tile):
    size = max(16, min(tile.shape[:2]))
    sigma = max(10.0, size / 6.5)
    low = cv2.GaussianBlur(tile, (0, 0), sigma)
    mean = np.mean(low.reshape(-1, 3), axis=0).reshape(1, 1, 3)
    return np.clip(tile / np.maximum(low, 1e-3) * mean, 0.0, 1.0).astype(np.float32)


def make_seamless_tile(tile, seam_px=None):
    h, w = tile.shape[:2]
    if h < 64 or w < 64:
        return tile
    if seam_px is None:
        seam_px = max(8, min(h, w) // 16)
    seam_px = int(min(seam_px, h // 4, w // 4))
    if seam_px <= 0:
        return tile
    out = tile.copy()
    original = tile.copy()
    for x in range(seam_px):
        alpha = (x + 1.0) / (seam_px + 1.0)
        out[:, x] = alpha * original[:, x] + (1.0 - alpha) * original[:, w - seam_px + x]
        out[:, w - seam_px + x] = alpha * original[:, w - seam_px + x] + (1.0 - alpha) * original[:, x]
    original = out.copy()
    for y in range(seam_px):
        alpha = (y + 1.0) / (seam_px + 1.0)
        out[y, :] = alpha * original[y, :] + (1.0 - alpha) * original[h - seam_px + y, :]
        out[h - seam_px + y, :] = alpha * original[h - seam_px + y, :] + (1.0 - alpha) * original[y, :]
    return np.clip(out, 0.0, 1.0)


def tile_image(tile, shape, offset_y=0, offset_x=0):
    h, w = shape
    th, tw = tile.shape[:2]
    yy = (np.arange(h) + offset_y) % th
    xx = (np.arange(w) + offset_x) % tw
    return tile[yy[:, None], xx[None, :]]


def match_lab_statistics(prior, target, target_mask, max_std_ratio=None):
    if np.count_nonzero(target_mask) < 64:
        return prior
    prior_u8 = np.clip(prior * 255.0, 0, 255).astype(np.uint8)
    target_u8 = np.clip(target * 255.0, 0, 255).astype(np.uint8)
    prior_lab = cv2.cvtColor(prior_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    src = prior_lab.reshape(-1, 3)
    dst = target_lab[target_mask]
    src_mean = np.mean(src, axis=0)
    src_std = np.std(src, axis=0) + 1e-4
    dst_mean = np.mean(dst, axis=0)
    dst_std = np.std(dst, axis=0) + 1e-4
    std_ratio = dst_std / src_std
    if max_std_ratio is not None and max_std_ratio > 0.0:
        std_ratio = np.clip(std_ratio, 1.0 / max_std_ratio, max_std_ratio)
    matched = (prior_lab - src_mean[None, None, :]) * std_ratio[None, None, :] + dst_mean[None, None, :]
    matched[..., 0] = np.clip(matched[..., 0], 0, 255)
    matched[..., 1:] = np.clip(matched[..., 1:], 0, 255)
    rgb = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return rgb


def find_best_tile(args, face, image, masks, tile_size, min_source_fraction):
    h, w = image.shape[:2]
    tile_size = int(min(tile_size, h, w))
    if tile_size < 32:
        return image.copy(), (0, 0, tile_size)
    source = masks["source"]
    high = masks["high"]
    reliability = masks["reliability"]
    object_risk = masks["object_risk"]
    clean_score = masks["clean_score"]
    edge = edge_energy(image)
    hsv = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    bad = (~source) | masks["contaminant"]

    source_int = integral_sum(source)
    high_int = integral_sum(high)
    bad_int = integral_sum(bad)
    min_source = min_source_fraction * tile_size * tile_size
    ys = list(range(0, h - tile_size + 1, args.tile_stride))
    xs = list(range(0, w - tile_size + 1, args.tile_stride))
    if not ys or ys[-1] != h - tile_size:
        ys.append(h - tile_size)
    if not xs or xs[-1] != w - tile_size:
        xs.append(w - tile_size)

    best = None
    best_score = -1e9
    for y in ys:
        for x in xs:
            source_count = rect_sum(source_int, y, x, tile_size)
            if source_count < min_source:
                continue
            bad_frac = rect_sum(bad_int, y, x, tile_size) / (tile_size * tile_size)
            if bad_frac > max(0.18, 1.0 - min_source_fraction + 0.06):
                continue
            high_count = rect_sum(high_int, y, x, tile_size)
            sl = np.s_[y : y + tile_size, x : x + tile_size]
            score = (
                1.55 * source_count / (tile_size * tile_size)
                + 0.70 * high_count / (tile_size * tile_size)
                + 1.05 * float(np.mean(reliability[sl]))
                + 0.58 * float(np.mean(clean_score[sl]))
                - 0.82 * float(np.mean(object_risk[sl]))
                - 0.30 * float(np.mean(edge[sl]))
                - 2.15 * bad_frac
            )
            if face == "ceiling":
                score -= 1.05 * float(np.mean(edge[sl]))
                score -= 0.85 * float(np.std(image[sl]))
                score -= 0.70 * float(np.mean(sat[sl]))
            elif face.startswith("wall"):
                score -= 1.55 * float(np.mean(edge[sl]))
                score -= 1.45 * float(np.std(image[sl]))
                score -= 0.95 * float(np.mean(sat[sl]))
            if score > best_score:
                best_score = score
                best = (y, x, tile_size)

    if best is None and tile_size > 64:
        smaller = max(64, tile_size // 2)
        return find_best_tile(args, face, image, masks, smaller, max(0.34, min_source_fraction - 0.18))
    if best is None:
        return image[:tile_size, :tile_size].copy(), (0, 0, tile_size)
    y, x, size = best
    return image[y : y + size, x : x + size].copy(), best


def load_external_material_prior(face, shape, args):
    if args.material_prior_dir is None:
        return None
    roots = [Path(args.material_prior_dir)]
    candidates = []
    for root in roots:
        candidates.extend(
            [
                root / face / "basecolor.png",
                root / face / "base_color.png",
                root / face / "albedo.png",
                root / face / "input.png",
                root / f"{face}.png",
                root / f"{face}_basecolor.png",
            ]
        )
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    tile = load_rgb(path)
    if tile.shape[:2] == shape:
        return tile
    return tile_image(tile, shape)


def build_material_prior(args, face, image, masks, tile_size, min_source_fraction, normalize=True):
    tile, tile_box = find_best_tile(args, face, image, masks, tile_size, min_source_fraction)
    material_tile = normalize_tile_lighting(tile) if normalize else tile
    if face == "floor":
        material_tile = make_seamless_tile(material_tile, max(12, min(material_tile.shape[:2]) // 18))
    elif face == "ceiling":
        color = np.median(material_tile.reshape(-1, 3), axis=0).reshape(1, 1, 3)
        material_tile = np.ones_like(material_tile) * color
    elif face.startswith("wall"):
        detail = cv2.bilateralFilter(
            np.clip(material_tile * 255.0, 0, 255).astype(np.uint8),
            d=0,
            sigmaColor=18,
            sigmaSpace=max(4, min(material_tile.shape[:2]) // 32),
        ).astype(np.float32) / 255.0
        smooth = cv2.GaussianBlur(detail, (0, 0), max(1.0, min(material_tile.shape[:2]) / 18.0))
        color = np.median(detail.reshape(-1, 3), axis=0).reshape(1, 1, 3)
        material_tile = np.clip(0.12 * smooth + 0.88 * color, 0.0, 1.0)
        material_tile = make_seamless_tile(material_tile, max(8, min(material_tile.shape[:2]) // 16))
    prior = load_external_material_prior(face, image.shape[:2], args)
    if prior is None:
        y, x, _ = tile_box
        prior = tile_image(material_tile, image.shape[:2], offset_y=y, offset_x=x)
    target_mask = masks.get("material_ref", masks["high"] | masks["source"])
    prior = match_lab_statistics(
        prior,
        image,
        target_mask,
        max_std_ratio=args.wall_prior_max_std_ratio if face.startswith("wall") else None,
    )
    if face == "ceiling":
        color = np.median(prior.reshape(-1, 3), axis=0).reshape(1, 1, 3)
        prior = np.ones_like(prior) * color
    elif face.startswith("wall"):
        if args.wall_prior_blur_sigma > 0.0:
            prior = cv2.GaussianBlur(prior, (0, 0), args.wall_prior_blur_sigma)
    return np.clip(prior, 0.0, 1.0), material_tile, tile_box


def blend_soft(base, replacement, alpha):
    alpha3 = np.clip(alpha, 0.0, 1.0)[..., None]
    return np.clip(base * (1.0 - alpha3) + replacement * alpha3, 0.0, 1.0)


def blend_by_mask(base, replacement, mask, feather_px):
    if not np.any(mask):
        return base.copy(), np.zeros(mask.shape, dtype=np.float32)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    alpha = np.clip(dist / max(1, feather_px), 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.5, feather_px / 4.0))
    alpha[~mask] = 0.0
    return blend_soft(base, replacement, alpha), alpha


def seam_polish(image, edit_mask, high_mask, seam_band_px, seam_smooth_px):
    if seam_band_px <= 0 or not np.any(edit_mask):
        return image, np.zeros(edit_mask.shape, dtype=bool)
    band = dilate_bool(edit_mask, seam_band_px) & (~erode_bool(edit_mask, 1))
    if not np.any(band):
        return image, band
    img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    smooth = cv2.bilateralFilter(img_u8, d=0, sigmaColor=18, sigmaSpace=max(2, seam_smooth_px))
    smooth = smooth.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(band.astype(np.float32), (0, 0), max(1.0, seam_band_px / 3.0))
    alpha = np.clip(alpha * 0.42, 0.0, 0.42)
    alpha[high_mask & (~edit_mask)] *= 0.25
    out = blend_soft(image, smooth, alpha)
    return out, band


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(mask.astype(np.float32) * 255.0, 0, 255).astype(np.uint8))


def complete_planar_face(args, out_dir, face):
    image = load_rgb(source_image_path(args.input_dir, face, args.seed_texture))
    h, w = image.shape[:2]
    masks = compute_quality_masks(args, face, (h, w))
    masks = apply_face_material_identity(args, face, image, masks)
    if face == "floor":
        tile_size = args.floor_tile_size
        source_fraction = args.floor_min_source_fraction
        percentile = args.floor_anomaly_percentile
        normalize = True
    else:
        tile_size = args.ceiling_tile_size
        source_fraction = args.ceiling_min_source_fraction
        percentile = args.ceiling_anomaly_percentile
        normalize = True

    prior, material_tile, tile_box = build_material_prior(
        args, face, image, masks, tile_size, source_fraction, normalize=normalize
    )
    score, anomaly = color_anomaly_score(image, masks.get("material_ref", masks["source"] | masks["high"]), percentile)
    abnormal_mid = masks["mid"] & anomaly
    hole = masks["fill"] | (~masks["observed"])
    hole = dilate_bool(hole, 1)

    medium_alpha = np.zeros((h, w), dtype=np.float32)
    medium_alpha[abnormal_mid] = np.clip(
        0.20 + 0.42 * np.clip(score[abnormal_mid], 0.0, 1.0),
        0.0,
        args.medium_max_alpha,
    )
    medium_alpha *= 1.0 - 0.55 * masks["reliability"]

    hole_completed, hole_alpha = blend_by_mask(image, prior, hole, args.hole_feather_px)
    mid_completed = blend_soft(hole_completed, prior, medium_alpha)
    locked = masks["high"] & (~dilate_bool(hole | abnormal_mid, 1))
    mid_completed[locked] = image[locked]
    final, seam_mask = seam_polish(
        mid_completed,
        hole | abnormal_mid,
        masks["high"],
        args.seam_band_px,
        args.seam_smooth_px,
    )
    final[locked] = image[locked]

    tex_dir = out_dir / "textures"
    debug_dir = out_dir / "full_debug"
    material_dir = out_dir / "material_prior_input"
    tex_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    material_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(tex_dir / f"{face}.png", final)
    save_rgb(debug_dir / f"{face}_material_prior.png", prior)
    save_rgb(material_dir / f"{face}.png", material_tile)
    save_mask(debug_dir / f"{face}_high_mask.png", masks["high"])
    save_mask(debug_dir / f"{face}_medium_anomaly_mask.png", abnormal_mid)
    save_mask(debug_dir / f"{face}_hole_mask.png", hole)
    save_mask(debug_dir / f"{face}_hole_alpha.png", hole_alpha)
    save_mask(debug_dir / f"{face}_medium_alpha.png", medium_alpha)
    save_mask(debug_dir / f"{face}_seam_mask.png", seam_mask)
    save_mask(debug_dir / f"{face}_reliability.png", masks["reliability"])
    save_mask(debug_dir / f"{face}_face_material_outlier_mask.png", masks["face_material_outlier"])

    return {
        "face": face,
        "method": "three_tier_material_prior_seam_polish",
        "tile_box_yx_size": [int(tile_box[0]), int(tile_box[1]), int(tile_box[2])],
        "high_texels": int(np.count_nonzero(masks["high"])),
        "medium_abnormal_texels": int(np.count_nonzero(abnormal_mid)),
        "face_material_outlier_texels": int(np.count_nonzero(masks["face_material_outlier"])),
        "hole_texels": int(np.count_nonzero(hole)),
        "seam_texels": int(np.count_nonzero(seam_mask)),
        "total_texels": int(h * w),
    }


def ring_prefill(image, mask, ring_px, radius):
    if not np.any(mask) or ring_px <= 0:
        return image.copy(), mask.copy()
    mask_u8 = mask.astype(np.uint8) * 255
    bgr = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, mask_u8, radius, cv2.INPAINT_TELEA)
    inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    ring = mask & (dist <= ring_px)
    out = image.copy()
    out[ring] = inpainted[ring]
    residual = mask & (~ring)
    return out, residual


def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    step = max(1, tile_size - overlap)
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def feather_window(h, w, x0, y0, full_w, full_h, overlap):
    weight = np.ones((h, w), dtype=np.float32)
    ramp = max(1, overlap)
    if x0 > 0:
        width = min(ramp, w)
        weight[:, :width] *= np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    if x0 + w < full_w:
        width = min(ramp, w)
        weight[:, -width:] *= np.linspace(1.0, 0.0, width, dtype=np.float32)[None, :]
    if y0 > 0:
        height = min(ramp, h)
        weight[:height, :] *= np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    if y0 + h < full_h:
        height = min(ramp, h)
        weight[-height:, :] *= np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    return np.clip(weight, 1e-4, 1.0)


def pad_tile(image_u8, mask_u8, tile_size):
    h, w = image_u8.shape[:2]
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h == 0 and pad_w == 0:
        return image_u8, mask_u8, h, w
    image_pad = cv2.copyMakeBorder(image_u8, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    mask_pad = cv2.copyMakeBorder(mask_u8, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return image_pad, mask_pad, h, w


def load_diffusers_pipe(args):
    import torch
    from diffusers import AutoPipelineForInpainting

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if args.dtype == "fp16" else None,
        )
    except OSError:
        pipe = AutoPipelineForInpainting.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    if hasattr(pipe, "feature_extractor"):
        pipe.feature_extractor = None
    pipe = pipe.to(args.device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    return pipe, torch


def load_brushnet_pipe(args):
    import torch
    from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline, UniPCMultistepScheduler

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    brushnet = BrushNetModel.from_pretrained(str(args.brushnet_path), torch_dtype=dtype)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        str(args.brushnet_base_model),
        brushnet=brushnet,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)
    return pipe, torch


def run_diffusers_tile(pipe, torch, image_u8, mask_u8, face, args, seed):
    generator = torch.Generator(device=args.device).manual_seed(seed)
    result = pipe(
        prompt=FACE_PROMPTS.get(face, FACE_PROMPTS["wall_00"]),
        negative_prompt=NEGATIVE_PROMPT,
        image=Image.fromarray(image_u8),
        mask_image=Image.fromarray(mask_u8).convert("L"),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        generator=generator,
    ).images[0]
    return np.asarray(result.convert("RGB"), dtype=np.uint8)


def run_brushnet_tile(pipe, torch, image_u8, mask_u8, face, args, seed):
    mask_bool = mask_u8 > 0
    condition = image_u8.copy()
    condition[mask_bool] = 0
    mask_rgb = np.repeat(mask_bool[..., None], 3, axis=2).astype(np.uint8) * 255
    generator = torch.Generator(device=args.device).manual_seed(seed)
    result = pipe(
        prompt=FACE_PROMPTS.get(face, FACE_PROMPTS["wall_00"]),
        image=Image.fromarray(condition).convert("RGB"),
        mask=Image.fromarray(mask_rgb).convert("RGB"),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=NEGATIVE_PROMPT,
        generator=generator,
        brushnet_conditioning_scale=args.brushnet_conditioning_scale,
    ).images[0]
    return np.asarray(result.convert("RGB"), dtype=np.uint8)


def run_opencv_inpaint(image, mask, radius=5):
    if not np.any(mask):
        return image.copy()
    bgr = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def split_hole_components(mask, max_component_ratio):
    model_mask = np.zeros_like(mask, dtype=bool)
    large_mask = np.zeros_like(mask, dtype=bool)
    if not np.any(mask):
        return model_mask, large_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    total = float(mask.size)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        if area / max(total, 1.0) <= max_component_ratio:
            model_mask[component] = True
        else:
            large_mask[component] = True
    return model_mask, large_mask


def material_loss_fusion_beta(
    model_candidate,
    prior,
    reference_mask,
    percentile,
    base_beta,
    min_beta,
    args,
):
    model_score, _ = color_anomaly_score(model_candidate, reference_mask, percentile)
    prior_delta = material_prior_delta(model_candidate, prior)
    anomaly_penalty = args.material_loss_anomaly_scale * np.clip(model_score / 2.0, 0.0, 1.0)
    prior_penalty = args.material_loss_prior_scale * np.clip(
        (prior_delta - args.material_loss_prior_start) / 2.0,
        0.0,
        1.0,
    )
    beta = base_beta - anomaly_penalty - prior_penalty
    return np.clip(beta, min_beta, base_beta).astype(np.float32), model_score, prior_delta


def material_delta_gate_fusion(model_candidate, prior, model_beta, edit_mask, args):
    fused = blend_soft(prior, model_candidate, model_beta)
    delta = material_prior_delta(model_candidate, prior)
    gate_alpha = np.zeros(delta.shape, dtype=np.float32)
    if args.material_delta_gate_enable and np.any(edit_mask):
        width = max(1e-6, args.material_delta_gate_soft_width)
        gate_alpha = np.clip((delta - args.material_delta_gate_thresh) / width, 0.0, 1.0).astype(np.float32)
        gate_alpha[~edit_mask] = 0.0
        fused = blend_soft(fused, prior, gate_alpha)
    return fused, delta.astype(np.float32), gate_alpha


def run_tiled_wall_model(image, residual_mask, face, args, pipe=None, torch=None):
    if args.wall_model_backend == "none" or not np.any(residual_mask):
        return image.copy(), "none"
    if args.wall_model_backend == "opencv":
        return run_opencv_inpaint(image, residual_mask, args.wall_telea_radius), "opencv"

    source_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    mask_u8_full = residual_mask.astype(np.uint8) * 255
    h, w = residual_mask.shape
    accum = np.zeros_like(source_u8, dtype=np.float32)
    accum_w = np.zeros((h, w), dtype=np.float32)
    tile_index = 0
    for y0 in tile_starts(h, args.tile_size, args.tile_overlap):
        for x0 in tile_starts(w, args.tile_size, args.tile_overlap):
            y1 = min(y0 + args.tile_size, h)
            x1 = min(x0 + args.tile_size, w)
            if not np.any(residual_mask[y0:y1, x0:x1]):
                continue
            tile = source_u8[y0:y1, x0:x1]
            mask = mask_u8_full[y0:y1, x0:x1]
            tile_pad, mask_pad, real_h, real_w = pad_tile(tile, mask, args.tile_size)
            if args.wall_model_backend == "brushnet":
                generated = run_brushnet_tile(
                    pipe, torch, tile_pad, mask_pad, face, args, args.seed + 101 * tile_index
                )[:real_h, :real_w]
            else:
                generated = run_diffusers_tile(
                    pipe, torch, tile_pad, mask_pad, face, args, args.seed + 101 * tile_index
                )[:real_h, :real_w]
            feather = feather_window(y1 - y0, x1 - x0, x0, y0, w, h, args.tile_overlap)
            accum[y0:y1, x0:x1] += generated.astype(np.float32) * feather[..., None]
            accum_w[y0:y1, x0:x1] += feather
            tile_index += 1
    out = source_u8.astype(np.float32)
    covered = accum_w > 1e-8
    out[covered] = accum[covered] / accum_w[covered, None]
    return np.clip(out / 255.0, 0.0, 1.0), args.wall_model_backend


def complete_wall_face(args, out_dir, face, pipe=None, torch=None):
    image = load_rgb(source_image_path(args.input_dir, face, args.seed_texture))
    h, w = image.shape[:2]
    masks = compute_quality_masks(args, face, (h, w))
    masks = apply_face_material_identity(args, face, image, masks)
    if face == "floor":
        tile_size = args.floor_tile_size
        source_fraction = args.floor_min_source_fraction
        anomaly_percentile = args.floor_anomaly_percentile
        material_thresh = max(1.65, args.wall_material_anomaly_thresh)
    elif face == "ceiling":
        tile_size = args.ceiling_tile_size
        source_fraction = args.ceiling_min_source_fraction
        anomaly_percentile = args.ceiling_anomaly_percentile
        material_thresh = args.wall_material_anomaly_thresh
    else:
        tile_size = args.wall_tile_size
        source_fraction = args.wall_min_source_fraction
        anomaly_percentile = args.wall_anomaly_percentile
        material_thresh = args.wall_material_anomaly_thresh
    prior, material_tile, tile_box = build_material_prior(
        args, face, image, masks, tile_size, source_fraction, normalize=True
    )
    score, anomaly = color_anomaly_score(
        image,
        masks.get("material_ref", masks["source"] | masks["high"]),
        anomaly_percentile,
    )
    edge = cv2.GaussianBlur(edge_energy(image), (0, 0), 1.4)
    ref_edge = edge[masks.get("material_ref", masks["source"] | masks["high"])]
    edge_floor = 0.105 if face == "floor" else 0.070
    edge_percentile = 90.0 if face == "floor" else 82.0
    edge_thresh = max(edge_floor, float(np.percentile(ref_edge, edge_percentile)) if ref_edge.size else edge_floor)
    edge_is_suspicious = (
        (masks["object_risk"] > 0.34)
        | (masks["clean_score"] < args.source_clean_thresh)
        | (masks["mask_boundary_trust"] < max(0.30, args.mask_boundary_keep_thresh))
        | (masks["footprint_area"] < max(0.06, args.footprint_keep_min))
    )
    texture_bad = masks["mid"] & (edge > edge_thresh) & edge_is_suspicious & (~masks["contaminant"])
    prior_delta = material_prior_delta(image, prior)
    prior_edge = cv2.GaussianBlur(edge_energy(prior), (0, 0), 1.4)
    texture_excess = (edge > np.maximum(edge_thresh * 1.12, prior_edge + 0.035)) & (prior_delta > 0.55)
    hsv = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    ref_mask = masks.get("material_ref", masks["source"] | masks["high"])
    ref_sat = sat[ref_mask]
    sat_cap = max(0.18, float(np.percentile(ref_sat, 88.0)) + 0.08) if ref_sat.size else 0.18
    color_material_mismatch = prior_delta > (material_thresh * 1.35)
    saturated_object_mismatch = (prior_delta > 0.80) & (sat > sat_cap)
    material_inconsistent = (
        masks["observed"]
        & (~masks["contaminant"])
        & (
            (prior_delta > material_thresh)
            | texture_excess
            | color_material_mismatch
            | saturated_object_mismatch
        )
        & (
            (masks["reliability"] < 0.86)
            | (masks["object_risk"] > 0.22)
            | (masks["mask_boundary_trust"] < 0.68)
            | (edge > edge_thresh)
            | color_material_mismatch
            | saturated_object_mismatch
        )
    )
    abnormal_mid = (masks["mid"] & anomaly) | texture_bad | material_inconsistent
    medium_alpha = np.zeros((h, w), dtype=np.float32)
    medium_alpha[abnormal_mid] = np.clip(
        0.14 + 0.30 * np.clip(score[abnormal_mid], 0.0, 1.0),
        0.0,
        min(args.medium_max_alpha, 0.42),
    )
    medium_alpha[texture_bad] = np.maximum(
        medium_alpha[texture_bad],
        np.clip(0.22 + 2.2 * (edge[texture_bad] - edge_thresh), 0.0, 0.50),
    )
    medium_alpha[material_inconsistent] = np.maximum(
        medium_alpha[material_inconsistent],
        np.clip(
            0.30 + 0.18 * np.clip(prior_delta[material_inconsistent] - material_thresh, 0.0, 3.0),
            0.0,
            args.wall_material_anomaly_alpha,
        ),
    )
    medium_alpha *= 1.0 - 0.62 * masks["reliability"]
    medium_alpha[texture_bad] = np.maximum(medium_alpha[texture_bad], 0.18)
    medium_alpha[material_inconsistent] = np.maximum(
        medium_alpha[material_inconsistent],
        args.wall_material_anomaly_alpha,
    )
    medium_repaired = blend_soft(image, prior, medium_alpha)

    hole = masks["fill"] | (~masks["observed"])
    hole = dilate_bool(hole, args.wall_mask_dilate_px)
    ring_image, residual = ring_prefill(medium_repaired, hole, args.wall_ring_px, args.wall_telea_radius)
    small_model_mask, large_guidance_mask = split_hole_components(residual, args.max_model_component_ratio)
    model_mask = residual.copy()
    model_mask_ratio = float(np.count_nonzero(model_mask) / residual.size)
    model_method = "skipped"
    model_error = None
    completed = ring_image.copy()
    direct_prior_mask = np.zeros_like(residual, dtype=bool)
    model_run_mask = np.zeros_like(residual, dtype=bool)
    model_beta = np.zeros((h, w), dtype=np.float32)
    model_score = np.zeros((h, w), dtype=np.float32)
    model_prior_delta = np.zeros((h, w), dtype=np.float32)
    model_delta_gate_alpha = np.zeros((h, w), dtype=np.float32)
    hole_alpha = np.zeros((h, w), dtype=np.float32)

    if np.any(model_mask) and model_mask_ratio <= args.max_model_mask_ratio:
        try:
            model_candidate, model_method = run_tiled_wall_model(
                ring_image, model_mask, face, args, pipe=pipe, torch=torch
            )
        except Exception as exc:
            if not args.allow_model_fallback:
                raise
            model_error = repr(exc)
            model_candidate, model_method = prior.copy(), "material_prior_fallback"

        small_beta, model_score, model_prior_delta = material_loss_fusion_beta(
            model_candidate,
            prior,
            masks.get("material_ref", masks["source"] | masks["high"]),
            anomaly_percentile,
            args.small_brushnet_beta,
            args.small_brushnet_min_beta,
            args,
        )
        large_beta, _, _ = material_loss_fusion_beta(
            model_candidate,
            prior,
            masks.get("material_ref", masks["source"] | masks["high"]),
            anomaly_percentile,
            args.large_brushnet_beta,
            args.large_brushnet_min_beta,
            args,
        )
        model_beta[small_model_mask] = small_beta[small_model_mask]
        model_beta[large_guidance_mask] = large_beta[large_guidance_mask]
        model_beta[~model_mask] = 0.0
        fused_candidate, model_prior_delta, model_delta_gate_alpha = material_delta_gate_fusion(
            model_candidate,
            prior,
            model_beta,
            model_mask,
            args,
        )
        completed, hole_alpha = blend_by_mask(
            ring_image,
            fused_candidate,
            model_mask,
            args.wall_hole_feather_px,
        )
        model_run_mask = model_mask.copy()
        if np.any(large_guidance_mask):
            model_method = f"{model_method}+material_loss_large_guidance"
        if np.any(small_model_mask):
            model_method = f"{model_method}+small_hole_model"
    else:
        model_candidate = prior.copy() if np.any(model_mask) else completed.copy()
        if np.any(model_mask):
            direct_prior_mask = model_mask.copy()
            completed, hole_alpha = blend_by_mask(
                ring_image,
                prior,
                direct_prior_mask,
                args.wall_hole_feather_px,
            )
            model_method = "material_prior_fallback_mask_too_large"
        else:
            model_method = "none"

    if args.lock_all_high:
        locked = masks["high"]
    else:
        locked = masks["high"] & (~dilate_bool(hole | abnormal_mid, 1))
    completed[locked] = image[locked]
    final, seam_mask = seam_polish(
        completed,
        hole | abnormal_mid,
        masks["high"],
        args.seam_band_px,
        args.seam_smooth_px,
    )
    final[locked] = image[locked]

    tex_dir = out_dir / "textures"
    debug_dir = out_dir / "full_debug"
    material_dir = out_dir / "material_prior_input"
    tex_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    material_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(tex_dir / f"{face}.png", final)
    save_rgb(debug_dir / f"{face}_material_prior.png", prior)
    save_rgb(debug_dir / f"{face}_ring_prefill.png", ring_image)
    save_rgb(debug_dir / f"{face}_model_candidate.png", model_candidate)
    save_mask(debug_dir / f"{face}_model_beta.png", model_beta)
    save_mask(debug_dir / f"{face}_model_material_delta.png", np.clip(model_prior_delta / 2.0, 0.0, 1.0))
    save_mask(debug_dir / f"{face}_material_delta_gate_alpha.png", model_delta_gate_alpha)
    save_rgb(material_dir / f"{face}.png", material_tile)
    save_mask(debug_dir / f"{face}_high_mask.png", masks["high"])
    save_mask(debug_dir / f"{face}_medium_anomaly_mask.png", abnormal_mid)
    save_mask(debug_dir / f"{face}_material_inconsistent_mask.png", material_inconsistent)
    save_mask(debug_dir / f"{face}_hole_mask.png", hole)
    save_mask(debug_dir / f"{face}_residual_model_mask.png", model_run_mask)
    save_mask(debug_dir / f"{face}_small_model_mask.png", small_model_mask)
    save_mask(debug_dir / f"{face}_large_prior_mask.png", large_guidance_mask)
    save_mask(debug_dir / f"{face}_direct_prior_mask.png", direct_prior_mask)
    save_mask(debug_dir / f"{face}_hole_alpha.png", hole_alpha)
    save_mask(debug_dir / f"{face}_medium_alpha.png", medium_alpha)
    save_mask(debug_dir / f"{face}_seam_mask.png", seam_mask)
    save_mask(debug_dir / f"{face}_reliability.png", masks["reliability"])
    save_mask(debug_dir / f"{face}_face_material_outlier_mask.png", masks["face_material_outlier"])

    return {
        "face": face,
        "method": "surface_brushnet_material_loss_fusion",
        "tile_box_yx_size": [int(tile_box[0]), int(tile_box[1]), int(tile_box[2])],
        "high_texels": int(np.count_nonzero(masks["high"])),
        "medium_abnormal_texels": int(np.count_nonzero(abnormal_mid)),
        "face_material_outlier_texels": int(np.count_nonzero(masks["face_material_outlier"])),
        "hole_texels": int(np.count_nonzero(hole)),
        "residual_model_texels": int(np.count_nonzero(model_run_mask)),
        "small_model_texels": int(np.count_nonzero(small_model_mask & model_run_mask)),
        "large_material_guidance_texels": int(np.count_nonzero(large_guidance_mask & model_run_mask)),
        "direct_material_prior_texels": int(np.count_nonzero(direct_prior_mask)),
        "large_prior_texels": int(np.count_nonzero(large_guidance_mask)),
        "mean_model_beta": float(np.mean(model_beta[model_run_mask])) if np.any(model_run_mask) else 0.0,
        "material_delta_gate_texels": int(np.count_nonzero(model_delta_gate_alpha > 0.5)),
        "seam_texels": int(np.count_nonzero(seam_mask)),
        "model_method": model_method,
        "model_error": model_error,
        "total_texels": int(h * w),
    }


def prepare_wall_material_tile(args, out_dir, face):
    image = load_rgb(source_image_path(args.input_dir, face, args.seed_texture))
    h, w = image.shape[:2]
    masks = compute_quality_masks(args, face, (h, w))
    masks = apply_face_material_identity(args, face, image, masks)
    if face == "floor":
        tile_size = args.floor_tile_size
        source_fraction = args.floor_min_source_fraction
    elif face == "ceiling":
        tile_size = args.ceiling_tile_size
        source_fraction = args.ceiling_min_source_fraction
    else:
        tile_size = args.wall_tile_size
        source_fraction = args.wall_min_source_fraction
    _, material_tile, tile_box = build_material_prior(
        args, face, image, masks, tile_size, source_fraction, normalize=True
    )
    material_dir = out_dir / "material_prior_input"
    debug_dir = out_dir / "full_debug"
    material_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(material_dir / f"{face}.png", material_tile)
    save_mask(debug_dir / f"{face}_source_mask.png", masks["source"])
    save_mask(debug_dir / f"{face}_face_material_outlier_mask.png", masks["face_material_outlier"])
    return {
        "face": face,
        "method": "prepare_material_tile_only",
        "tile_box_yx_size": [int(tile_box[0]), int(tile_box[1]), int(tile_box[2])],
        "source_texels": int(np.count_nonzero(masks["source"])),
        "face_material_outlier_texels": int(np.count_nonzero(masks["face_material_outlier"])),
        "total_texels": int(h * w),
    }


def copy_mesh_files(input_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["room_empty.obj", "room_empty.mtl", "metadata.json", "README_unity.md"]:
        src = input_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)


def write_preview(out_dir, faces):
    thumbs = []
    for face in faces:
        path = out_dir / "textures" / f"{face}.png"
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        im.thumbnail((420, 260))
        canvas = Image.new("RGB", (420, 300), (30, 30, 30))
        canvas.paste(im, ((420 - im.width) // 2, 30 + (260 - im.height) // 2))
        ImageDraw.Draw(canvas).text((10, 8), face, fill=(255, 255, 255))
        thumbs.append(canvas)
    if not thumbs:
        return
    sheet = Image.new("RGB", (840, 300 * math.ceil(len(thumbs) / 2)), (18, 18, 18))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % 2) * 420, (i // 2) * 300))
    sheet.save(out_dir / "structured_full_texture_preview.jpg", quality=92)


def prepare_model(args):
    if args.wall_model_backend not in {"diffusers", "brushnet"}:
        return None, None, None
    try:
        if args.wall_model_backend == "brushnet":
            pipe, torch = load_brushnet_pipe(args)
        else:
            pipe, torch = load_diffusers_pipe(args)
        return pipe, torch, None
    except Exception as exc:
        if not args.allow_model_fallback:
            raise
        print(f"[warn] {args.wall_model_backend} backend failed to initialize; using material-prior fallback: {exc!r}")
        args.wall_model_backend = "none"
        return None, None, repr(exc)


def main():
    args = parse_args()
    faces = parse_faces(args.faces, args.input_dir)
    copy_mesh_files(args.input_dir, args.out_dir)
    stats = []
    if args.prepare_material_tiles_only:
        for face in faces:
            stats.append(prepare_wall_material_tile(args, args.out_dir, face))
        with open(args.out_dir / "metadata_material_tile_prepare.json", "w", encoding="utf-8") as f:
            json.dump({"source_export_dir": str(args.input_dir), "stats": stats}, f, indent=2)
        print("[prepare] material tiles written:", args.out_dir / "material_prior_input")
        return

    pipe, torch, init_error = prepare_model(args)
    for face in faces:
        stats.append(complete_wall_face(args, args.out_dir, face, pipe=pipe, torch=torch))
    write_preview(args.out_dir, faces)
    metadata = {
        "source_export_dir": str(args.input_dir),
        "method": "structured_full_material_prior_rad_fusion",
        "model_init_error": init_error,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"input_dir", "out_dir"}
        },
        "stats": stats,
    }
    with open(args.out_dir / "metadata_structured_full_completion.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("[done] Wrote:", args.out_dir)


if __name__ == "__main__":
    main()
