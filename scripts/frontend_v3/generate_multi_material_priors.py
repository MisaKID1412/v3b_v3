#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
PIX_DIR = PROJECT_DIR / "pipeline" / "pixcuboid" / "PixCuboid-main"
if str(PIX_DIR) not in sys.path:
    sys.path.insert(0, str(PIX_DIR))

import generate_material_priors as gmp  # noqa: E402
from complete_room_textures_full import (  # noqa: E402
    FACE_NAMES,
    apply_face_material_identity,
    compute_quality_masks,
    edge_energy,
    integral_sum,
    load_brushnet_pipe,
    make_seamless_tile,
    match_lab_statistics,
    normalize_tile_lighting,
    parse_faces,
    rect_sum,
    run_tiled_wall_model,
    save_mask,
    save_rgb,
    source_image_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate several Material Anything priors per surface from distinct high-confidence "
            "material regions, blend them into one smooth prior field per face, and keep the "
            "downstream v53/v60 material-observation fusion interface unchanged."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed-texture", choices=["raw", "texture"], default="raw")
    parser.add_argument("--backend", choices=["patch", "material_anything"], default="material_anything")
    parser.add_argument("--faces", nargs="*", default=None)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-stride", type=int, default=16)
    parser.add_argument("--floor-min-source-fraction", type=float, default=0.88)
    parser.add_argument("--ceiling-min-source-fraction", type=float, default=0.78)
    parser.add_argument("--wall-min-source-fraction", type=float, default=0.70)
    parser.add_argument("--max-candidates", type=int, default=54)
    parser.add_argument("--max-priors-per-face", type=int, default=6)
    parser.add_argument("--min-priors-per-face", type=int, default=1)
    parser.add_argument("--candidate-nms-iou", type=float, default=0.30)
    parser.add_argument("--candidate-min-center-frac", type=float, default=0.16)
    parser.add_argument("--cluster-lab-delta", type=float, default=0.72)
    parser.add_argument("--cluster-texture-delta", type=float, default=0.18)
    parser.add_argument("--blend-distance-frac", type=float, default=0.34)
    parser.add_argument("--blend-smooth-frac", type=float, default=0.018)
    parser.add_argument("--raw-compat-sigma", type=float, default=0.95)
    parser.add_argument("--raw-compat-strength", type=float, default=0.78)
    parser.add_argument("--support-dilate-frac", type=float, default=0.018)
    parser.add_argument(
        "--final-global-match-strength",
        type=float,
        default=0.0,
        help=(
            "Optional weak whole-face Lab correction after blending region priors. "
            "Keep this at 0 for multi-material faces; each region prior is already "
            "matched to its own support."
        ),
    )
    parser.add_argument("--surface-contaminant-mask-dir", type=Path, default=None)
    parser.add_argument("--surface-contaminant-dilate-px", type=int, default=8)
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
    parser.add_argument("--strict-ma-support", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-support-min-reliability", type=float, default=0.50)
    parser.add_argument("--strict-support-clean-thresh", type=float, default=0.62)
    parser.add_argument("--strict-support-object-risk-thresh", type=float, default=0.34)
    parser.add_argument("--strict-support-boundary-trust-thresh", type=float, default=0.60)
    parser.add_argument("--material-anything-dir", type=Path, default=Path("third_party/MaterialAnything"))
    parser.add_argument(
        "--material-anything-estimator",
        type=Path,
        default=Path("models/materialanything/material_estimator"),
    )
    parser.add_argument("--material-anything-size", type=int, default=512)
    parser.add_argument("--material-anything-steps", type=int, default=32)
    parser.add_argument("--material-anything-keep-thresh", type=float, default=0.58)
    parser.add_argument(
        "--non-tile-ma-seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Initialize Material Anything from a full-face, non-periodic support completion "
            "instead of a tiled patch prior. The MA keep mask still comes from the selected "
            "material support, so each prior may occupy an irregular observed/masked shape."
        ),
    )
    parser.add_argument("--non-tile-seed-fill-backend", choices=["brushnet", "lowfreq", "precomputed"], default="lowfreq")
    parser.add_argument("--precomputed-non-tile-seed-dir", type=Path, default=None)
    parser.add_argument("--non-tile-seed-inpaint-radius", type=float, default=7.0)
    parser.add_argument("--non-tile-seed-support-dilate-frac", type=float, default=0.012)
    parser.add_argument("--non-tile-seed-lowfreq-sigma-frac", type=float, default=0.030)
    parser.add_argument(
        "--material-anything-wall-mode",
        choices=["stat_prior", "observed", "projected_mask"],
        default="stat_prior",
    )
    parser.add_argument("--material-prior-quality-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--material-prior-max-lab-delta", type=float, default=12.0)
    parser.add_argument("--material-prior-max-edge-ratio", type=float, default=1.55)
    parser.add_argument("--material-prior-max-sat-ratio", type=float, default=1.12)
    parser.add_argument(
        "--apply-face-material-identity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the older single-material face outlier rejection before selecting MA supports. "
            "Default is false because multi-MA must preserve real secondary materials such as "
            "paint bands, accent tiles, or mixed wall finishes."
        ),
    )
    parser.add_argument("--face-material-outlier-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--face-material-source-reliability", type=float, default=0.44)
    parser.add_argument("--face-material-inlier-percentile", type=float, default=90.0)
    parser.add_argument("--face-material-outlier-dilate-px", type=int, default=2)
    parser.add_argument("--face-material-outlier-max-ratio", type=float, default=0.34)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--wall-model-backend", choices=["none", "opencv", "diffusers", "brushnet"], default="brushnet")
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
    parser.add_argument("--strength", type=float, default=0.36)
    parser.add_argument("--floor-strength", type=float, default=0.20)
    parser.add_argument("--ceiling-strength", type=float, default=0.20)
    parser.add_argument("--wall-strength", type=float, default=0.24)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--allow-fallback", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_lab_delta(a, b):
    d = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.sqrt((d[0] / 30.0) ** 2 + (d[1] / 11.0) ** 2 + (d[2] / 11.0) ** 2))


def tile_iou(a, b):
    ay, ax, asz = a
    by, bx, bsz = b
    y0 = max(ay, by)
    x0 = max(ax, bx)
    y1 = min(ay + asz, by + bsz)
    x1 = min(ax + asz, bx + bsz)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    inter = float((y1 - y0) * (x1 - x0))
    union = float(asz * asz + bsz * bsz - inter)
    return inter / max(union, 1.0)


def tile_center(box):
    y, x, size = box
    return np.asarray([x + 0.5 * size, y + 0.5 * size], dtype=np.float32)


def clean_material_support(args, masks):
    if args is None or not getattr(args, "strict_ma_support", False):
        return masks["source"] | masks["high"]
    strict = (
        masks["source"]
        & (masks["reliability"] >= args.strict_support_min_reliability)
        & (masks["clean_score"] >= args.strict_support_clean_thresh)
        & (masks["object_risk"] <= args.strict_support_object_risk_thresh)
        & (masks["mask_boundary_trust"] >= args.strict_support_boundary_trust_thresh)
        & (~masks["contaminant"])
    )
    strict |= masks["high"]
    strict &= ~masks["contaminant"]
    if np.count_nonzero(strict) < 64:
        strict = masks["high"] & (~masks["contaminant"])
    return strict


def tile_stats(face, image, masks, box, args=None):
    y, x, size = box
    sl = np.s_[y : y + size, x : x + size]
    support = clean_material_support(args, masks)[sl]
    if np.count_nonzero(support) < 64:
        support = (masks["source"][sl] | masks["high"][sl]) & (~masks["contaminant"][sl])
    if np.count_nonzero(support) < 64:
        support = np.ones((size, size), dtype=bool)
    tile = image[sl]
    lab = cv2.cvtColor(np.clip(tile * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_lab = np.median(lab[support], axis=0)
    edge = edge_energy(tile)
    edge_mean = float(np.mean(edge[support])) if np.any(support) else float(np.mean(edge))
    sat = cv2.cvtColor(np.clip(tile * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    sat_mean = float(np.mean(sat[support])) if np.any(support) else float(np.mean(sat))
    return {
        "mean_lab": mean_lab,
        "edge_mean": edge_mean,
        "sat_mean": sat_mean,
        "support_pixels": int(np.count_nonzero(support)),
        "face": face,
    }


def enumerate_candidates(args, face, image, masks, tile_size, min_source_fraction):
    h, w = image.shape[:2]
    tile_size = int(min(tile_size, h, w))
    if tile_size < 32:
        return [{"box": (0, 0, tile_size), "score": 0.0, **tile_stats(face, image, masks, (0, 0, tile_size), args)}]

    source = clean_material_support(args, masks) if args.strict_ma_support else masks["source"]
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

    candidates = []
    for y in ys:
        for x in xs:
            source_count = rect_sum(source_int, y, x, tile_size)
            if source_count < min_source:
                continue
            bad_frac = rect_sum(bad_int, y, x, tile_size) / float(tile_size * tile_size)
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
                # For multi-material walls, color/saturation differences are signal,
                # not necessarily contamination. Keep only a mild texture penalty.
                score -= 0.86 * float(np.mean(edge[sl]))
                score -= 0.54 * float(np.std(image[sl]))
                score -= 0.18 * float(np.mean(sat[sl]))
            box = (int(y), int(x), int(tile_size))
            candidates.append({"box": box, "score": float(score), **tile_stats(face, image, masks, box, args)})

    if not candidates and tile_size > 64:
        return enumerate_candidates(args, face, image, masks, max(64, tile_size // 2), max(0.34, min_source_fraction - 0.18))
    if not candidates:
        box = (0, 0, tile_size)
        return [{"box": box, "score": -1e6, **tile_stats(face, image, masks, box, args)}]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def select_spread_candidates(args, candidates, shape):
    selected = []
    min_dim = float(min(shape))
    for cand in candidates:
        if len(selected) >= args.max_candidates:
            break
        c0 = tile_center(cand["box"])
        too_close = False
        for prev in selected:
            c1 = tile_center(prev["box"])
            dist = float(np.linalg.norm(c0 - c1)) / max(min_dim, 1.0)
            if tile_iou(cand["box"], prev["box"]) > args.candidate_nms_iou or dist < args.candidate_min_center_frac:
                too_close = True
                break
        if not too_close:
            selected.append(cand)
    if not selected and candidates:
        selected.append(candidates[0])
    return selected


def cluster_candidates(args, candidates):
    clusters = []
    for cand in candidates:
        best_i = -1
        best_d = 1e9
        for i, cluster in enumerate(clusters):
            d_lab = normalize_lab_delta(cand["mean_lab"], cluster["mean_lab"])
            d_tex = abs(float(cand["edge_mean"]) - float(cluster["edge_mean"])) + 0.55 * abs(
                float(cand["sat_mean"]) - float(cluster["sat_mean"])
            )
            if d_lab < args.cluster_lab_delta and d_tex < args.cluster_texture_delta and d_lab < best_d:
                best_i = i
                best_d = d_lab
        if best_i < 0:
            clusters.append(
                {
                    "items": [cand],
                    "score": float(cand["score"]),
                    "mean_lab": cand["mean_lab"].copy(),
                    "edge_mean": float(cand["edge_mean"]),
                    "sat_mean": float(cand["sat_mean"]),
                    "representative": cand,
                }
            )
        else:
            cluster = clusters[best_i]
            cluster["items"].append(cand)
            weight_old = max(cluster["score"], 1e-3)
            weight_new = max(float(cand["score"]), 1e-3)
            total = weight_old + weight_new
            cluster["mean_lab"] = (cluster["mean_lab"] * weight_old + cand["mean_lab"] * weight_new) / total
            cluster["edge_mean"] = (cluster["edge_mean"] * weight_old + float(cand["edge_mean"]) * weight_new) / total
            cluster["sat_mean"] = (cluster["sat_mean"] * weight_old + float(cand["sat_mean"]) * weight_new) / total
            cluster["score"] = max(cluster["score"], float(cand["score"]))
            if float(cand["score"]) > float(cluster["representative"]["score"]):
                cluster["representative"] = cand

    clusters.sort(key=lambda item: item["score"], reverse=True)
    keep = max(args.min_priors_per_face, min(args.max_priors_per_face, len(clusters)))
    return clusters[:keep]


def prepare_tile_from_box(face, image, masks, box):
    y, x, size = box
    tile = image[y : y + size, x : x + size].copy()
    source_mask = masks["source"][y : y + size, x : x + size]
    tile = gmp.inpaint_tile_holes(tile, source_mask)
    if face in {"floor", "ceiling"}:
        tile = normalize_tile_lighting(tile)
    else:
        edge = cv2.Sobel(
            cv2.cvtColor(np.clip(tile * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
        edge = np.abs(edge) / 255.0
        if np.any(source_mask):
            edge_cap = max(0.035, float(np.percentile(edge[source_mask], 55.0)))
            reliable = source_mask & (edge <= edge_cap)
        else:
            reliable = source_mask
        if np.count_nonzero(reliable) < 64:
            reliable = source_mask
        if np.count_nonzero(reliable) >= 64:
            color = np.median(tile[reliable], axis=0).reshape(1, 1, 3)
        else:
            color = np.median(tile.reshape(-1, 3), axis=0).reshape(1, 1, 3)
        smooth = np.ones_like(tile) * color
        subtle = cv2.GaussianBlur(tile - cv2.GaussianBlur(tile, (0, 0), max(3.0, size / 18.0)), (0, 0), 2.0)
        tile = np.clip(smooth + 0.035 * subtle, 0.0, 1.0)
    return make_seamless_tile(tile), box


def lowfreq_non_tile_seed(face, image, masks, keep, args):
    shape = image.shape[:2]
    min_dim = max(1, min(shape))
    color = np.median(image[keep], axis=0).astype(np.float32)
    canvas = np.ones_like(image, dtype=np.float32) * color.reshape(1, 1, 3)
    canvas[keep] = image[keep]

    mask = (~keep).astype(np.uint8) * 255
    bgr = cv2.cvtColor(np.clip(canvas * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, mask, float(args.non_tile_seed_inpaint_radius), cv2.INPAINT_TELEA)
    seed = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    sigma = max(1.0, float(args.non_tile_seed_lowfreq_sigma_frac) * min_dim)
    low = cv2.GaussianBlur(seed, (0, 0), sigma)
    detail = seed - cv2.GaussianBlur(seed, (0, 0), max(1.0, sigma * 0.35))
    seed = np.clip(low + 0.20 * detail, 0.0, 1.0)
    keep_soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 2.0)
    keep_soft = np.clip(keep_soft, 0.0, 1.0)
    seed = seed * (1.0 - keep_soft[..., None]) + image * keep_soft[..., None]
    return np.clip(seed, 0.0, 1.0), keep


def non_tile_seed_from_support(face, image, masks, support, args, brushnet_pipe=None, brushnet_torch=None):
    shape = image.shape[:2]
    min_dim = max(1, min(shape))
    contaminant = masks.get("contaminant", np.zeros(shape, dtype=bool))
    observed = (masks["source"] | masks["high"]) & (~contaminant)
    keep = support & observed
    if np.count_nonzero(keep) < 256:
        keep = observed
    if np.count_nonzero(keep) < 256:
        color = np.median(image.reshape(-1, 3), axis=0).astype(np.float32)
        return np.ones_like(image) * color.reshape(1, 1, 3), keep, "flat_fallback"

    if args.non_tile_seed_support_dilate_frac > 0:
        radius = max(1, int(round(float(args.non_tile_seed_support_dilate_frac) * min_dim)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        keep_d = cv2.dilate(keep.astype(np.uint8), kernel).astype(bool) & observed
        if np.count_nonzero(keep_d) >= 256:
            keep = keep_d

    if args.non_tile_seed_fill_backend == "brushnet" and brushnet_pipe is not None and brushnet_torch is not None:
        condition = image.copy()
        color = np.median(image[keep], axis=0).astype(np.float32)
        condition[~keep] = color.reshape(1, 3)
        edit_mask = ~keep
        try:
            old_backend = getattr(args, "wall_model_backend", "brushnet")
            args.wall_model_backend = "brushnet"
            seed, method = run_tiled_wall_model(
                condition,
                edit_mask,
                face,
                args,
                pipe=brushnet_pipe,
                torch=brushnet_torch,
            )
            args.wall_model_backend = old_backend
            keep_soft = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 2.0)
            keep_soft = np.clip(keep_soft, 0.0, 1.0)
            seed = seed * (1.0 - keep_soft[..., None]) + image * keep_soft[..., None]
            return np.clip(seed, 0.0, 1.0), keep, f"brushnet_{method}"
        except Exception as exc:
            if not args.allow_fallback:
                raise
            print(f"[warn] BrushNet non-tile seed failed on {face}; using lowfreq fallback: {exc}")

    seed, keep = lowfreq_non_tile_seed(face, image, masks, keep, args)
    return seed, keep, "lowfreq_fallback"


def cluster_support_mask(shape, masks, cluster, args=None):
    support = np.zeros(shape, dtype=bool)
    clean_support = clean_material_support(args, masks)
    for item in cluster["items"]:
        y, x, size = item["box"]
        sl = np.s_[y : y + size, x : x + size]
        support[sl] |= clean_support[sl]
    if np.count_nonzero(support) < 64:
        rep = cluster["representative"]["box"]
        y, x, size = rep
        sl = np.s_[y : y + size, x : x + size]
        support[sl] = (masks["high"][sl] | clean_support[sl]) & (~masks["contaminant"][sl])
    if np.count_nonzero(support) < 64:
        rep = cluster["representative"]["box"]
        y, x, size = rep
        sl = np.s_[y : y + size, x : x + size]
        support[sl] = (masks["source"][sl] | masks["high"][sl]) & (~masks["contaminant"][sl])
    return support


def raw_lab_compatibility(raw, mean_lab, sigma):
    raw_lab = cv2.cvtColor(np.clip(raw * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    d = raw_lab - mean_lab.reshape(1, 1, 3)
    nd = np.sqrt((d[..., 0] / 30.0) ** 2 + (d[..., 1] / 11.0) ** 2 + (d[..., 2] / 11.0) ** 2)
    return np.exp(-(nd * nd) / max(1e-6, 2.0 * sigma * sigma)).astype(np.float32)


def blend_region_priors(args, face, raw, masks, clusters, region_priors, region_supports, region_valids=None):
    h, w = raw.shape[:2]
    weights = []
    min_dim = float(min(h, w))
    sigma = max(16.0, args.blend_distance_frac * min_dim)
    smooth_sigma = max(0.0, args.blend_smooth_frac * min_dim)
    support_dilate = max(1, int(round(args.support_dilate_frac * min_dim)))
    observed_gate = np.clip(masks["reliability"], 0.0, 1.0) * (masks["source"] | masks["high"]).astype(np.float32)
    observed_gate = cv2.GaussianBlur(observed_gate, (0, 0), 1.8)
    observed_gate = np.clip(observed_gate, 0.0, 1.0)

    score_values = np.asarray([float(c["score"]) for c in clusters], dtype=np.float32)
    if score_values.size:
        lo = float(np.min(score_values))
        hi = float(np.max(score_values))
    else:
        lo, hi = 0.0, 1.0

    fallback_weights = []
    for idx, (cluster, support) in enumerate(zip(clusters, region_supports)):
        support = support.copy()
        if support_dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * support_dilate + 1, 2 * support_dilate + 1))
            support = cv2.dilate(support.astype(np.uint8), kernel).astype(bool)
        dist = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        # Some completely unobserved faces contain OpenCV's large sentinel
        # distance.  Square in float64 so the mathematically intended exp(-inf)
        # limit is reached without a noisy float32 overflow warning.
        dist64 = dist.astype(np.float64)
        spatial = np.exp(-(dist64 * dist64) / max(1e-6, 2.0 * sigma * sigma)).astype(np.float32)
        score_q = 0.70 + 0.30 * ((float(cluster["score"]) - lo) / max(hi - lo, 1e-6))
        compat = raw_lab_compatibility(raw, cluster["mean_lab"], args.raw_compat_sigma)
        compat = (1.0 - observed_gate) + observed_gate * (
            (1.0 - args.raw_compat_strength) + args.raw_compat_strength * compat
        )
        weight = spatial * score_q * compat
        fallback_weights.append(np.clip(weight.copy(), 0.0, None))
        if region_valids is not None and idx < len(region_valids) and region_valids[idx] is not None:
            valid = region_valids[idx].astype(np.float32)
            if valid.shape == weight.shape:
                valid = cv2.GaussianBlur(valid, (0, 0), max(1.0, smooth_sigma))
                weight *= np.clip(valid, 0.0, 1.0)
        if smooth_sigma > 0:
            weight = cv2.GaussianBlur(weight, (0, 0), smooth_sigma)
        weights.append(np.clip(weight, 0.0, None))

    if not weights:
        return region_priors[0].copy(), []
    weight_stack = np.stack(weights, axis=0)
    denom_raw = np.sum(weight_stack, axis=0)
    if fallback_weights:
        fallback_stack = np.stack(fallback_weights, axis=0)
        no_owner = denom_raw <= 1e-6
        if np.any(no_owner):
            weight_stack[:, no_owner] = fallback_stack[:, no_owner]
            denom_raw = np.sum(weight_stack, axis=0)
    denom = np.maximum(denom_raw, 1e-6)
    out = np.zeros((h, w, 3), dtype=np.float32)
    for weight, prior in zip(weight_stack, region_priors):
        out += prior * (weight / denom)[..., None]
    return np.clip(out, 0.0, 1.0), [w / denom for w in weight_stack]


def load_material_pipe(args):
    if args.backend != "material_anything":
        return None, None, None
    try:
        pipe, torch = gmp.load_material_anything_pipe(args)
        return pipe, torch, None
    except Exception as exc:
        if not args.allow_fallback:
            raise
        return None, None, repr(exc)


def load_precomputed_non_tile_seed(seed_dir, face, region_i, shape):
    if seed_dir is None:
        raise FileNotFoundError("precomputed seed dir is not set")
    seed_path = seed_dir / "region_priors" / face / f"region_{region_i:02d}_non_tile_seed.png"
    keep_path = seed_dir / "debug" / f"{face}_region_{region_i:02d}_non_tile_seed_keep.png"
    valid_path = seed_dir / "debug" / f"{face}_region_{region_i:02d}_non_tile_seed_valid.png"
    if not seed_path.exists():
        raise FileNotFoundError(seed_path)
    seed = gmp.load_rgb(seed_path)
    if seed.shape[:2] != shape:
        seed = cv2.resize(seed, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    keep = np.zeros(shape, dtype=bool)
    if keep_path.exists():
        keep_u8 = cv2.imread(str(keep_path), cv2.IMREAD_GRAYSCALE)
        if keep_u8 is not None:
            if keep_u8.shape != shape:
                keep_u8 = cv2.resize(keep_u8, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            keep = keep_u8 > 127
    valid = np.ones(shape, dtype=bool)
    if valid_path.exists():
        valid_u8 = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
        if valid_u8 is not None:
            if valid_u8.shape != shape:
                valid_u8 = cv2.resize(valid_u8, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            valid = valid_u8 > 127
    return seed, keep, valid


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prior_dir = args.out_dir / "priors"
    region_dir = args.out_dir / "region_priors"
    debug_dir = args.out_dir / "debug"
    prior_dir.mkdir(parents=True, exist_ok=True)
    region_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    faces = args.faces if args.faces else parse_faces(None, args.input_dir)
    pipe, torch, init_error = load_material_pipe(args)
    if init_error is not None:
        print(f"[warn] Material Anything failed to initialize; using patch backend: {init_error}")
        args.backend = "patch"
    brushnet_pipe = None
    brushnet_torch = None
    brushnet_error = None
    if args.non_tile_ma_seed and args.non_tile_seed_fill_backend == "brushnet":
        try:
            brushnet_pipe, brushnet_torch = load_brushnet_pipe(args)
            print("[info] BrushNet loaded for non-tile MA seed filling")
        except Exception as exc:
            brushnet_error = repr(exc)
            if not args.allow_fallback:
                raise
            print(f"[warn] BrushNet failed to initialize; non-tile seeds will use lowfreq fallback: {brushnet_error}")

    stats = []
    for face_i, face in enumerate(faces):
        image = gmp.load_rgb(source_image_path(args.input_dir, face, args.seed_texture))
        shape = image.shape[:2]
        masks = compute_quality_masks(args, face, shape)
        if args.apply_face_material_identity:
            masks = apply_face_material_identity(args, face, image, masks)
        else:
            masks["face_material_outlier"] = np.zeros(shape, dtype=bool)
            masks["material_ref"] = masks["source"] | masks["high"]
        if face == "floor":
            min_source = args.floor_min_source_fraction
        elif face == "ceiling":
            min_source = args.ceiling_min_source_fraction
        else:
            min_source = args.wall_min_source_fraction

        all_candidates = enumerate_candidates(args, face, image, masks, args.tile_size, min_source)
        spread = select_spread_candidates(args, all_candidates, shape)
        clusters = cluster_candidates(args, spread)
        face_region_dir = region_dir / face
        face_region_dir.mkdir(parents=True, exist_ok=True)

        region_priors = []
        region_supports = []
        region_valids = []
        region_stats = []
        for region_i, cluster in enumerate(clusters):
            ref_tile, tile_box = prepare_tile_from_box(face, image, masks, cluster["representative"]["box"])
            support = cluster_support_mask(shape, masks, cluster, args)
            cluster_masks = dict(masks)
            cluster_masks["material_ref"] = support
            if args.non_tile_ma_seed:
                cluster_masks["source"] = masks["source"] & support
                cluster_masks["high"] = masks["high"] & support
                if "observed" in cluster_masks:
                    cluster_masks["observed"] = masks["observed"] & support
            patch_prior = gmp.build_full_prior(face, image, cluster_masks, ref_tile, tile_box)
            if args.non_tile_ma_seed:
                if args.non_tile_seed_fill_backend == "precomputed":
                    ma_seed_prior, non_tile_keep, non_tile_valid = load_precomputed_non_tile_seed(
                        args.precomputed_non_tile_seed_dir,
                        face,
                        region_i,
                        shape,
                    )
                    if np.count_nonzero(non_tile_keep) < 256:
                        non_tile_keep = support
                    non_tile_seed_method = "precomputed_brushnet"
                else:
                    ma_seed_prior, non_tile_keep, non_tile_seed_method = non_tile_seed_from_support(
                        face,
                        image,
                        masks,
                        support,
                        args,
                        brushnet_pipe=brushnet_pipe,
                        brushnet_torch=brushnet_torch,
                    )
                    non_tile_valid = np.ones(shape, dtype=bool)
            else:
                ma_seed_prior, non_tile_keep, non_tile_seed_method = patch_prior, np.zeros(shape, dtype=bool), "patch_prior"
                non_tile_valid = np.ones(shape, dtype=bool)
            prior = ma_seed_prior
            method = "non_tile_seed" if args.non_tile_ma_seed else "patch"
            quality = None
            error = None
            keep_mask = np.zeros(shape, dtype=bool)
            if args.backend == "material_anything":
                try:
                    ma_full, keep_mask = gmp.material_anything_prior(
                        pipe,
                        torch,
                        face,
                        image,
                        cluster_masks,
                        ma_seed_prior,
                        args,
                        args.seed + face_i * 137 + region_i * 997,
                    )
                    rejection, quality = gmp.reject_material_prior(face, ma_full, ma_seed_prior, args)
                    if rejection is not None:
                        prior = ma_seed_prior
                        method = "material_anything_rejected_to_non_tile_seed" if args.non_tile_ma_seed else "material_anything_rejected_to_patch"
                        error = rejection
                    else:
                        if args.non_tile_ma_seed:
                            # MA is only allowed to rewrite the high-confidence
                            # irregular support. The surrounding territory remains
                            # the BrushNet-grown non-periodic completion, and later
                            # region weights decide how far this seed occupies.
                            support_alpha = cv2.GaussianBlur(non_tile_keep.astype(np.float32), (0, 0), 1.6)
                            support_alpha = np.clip(support_alpha, 0.0, 1.0)
                            prior = ma_seed_prior * (1.0 - support_alpha[..., None]) + ma_full * support_alpha[..., None]
                            method = "material_anything_support_shape_brushnet_expansion"
                        else:
                            prior = ma_full
                            method = "material_anything"
                except Exception as exc:
                    if not args.allow_fallback:
                        raise
                    error = repr(exc)
                    prior = ma_seed_prior
                    method = "non_tile_seed_fallback" if args.non_tile_ma_seed else "patch_fallback"
            region_priors.append(prior)
            region_supports.append(support)
            region_valids.append(non_tile_valid)
            save_rgb(face_region_dir / f"region_{region_i:02d}.png", prior)
            save_rgb(face_region_dir / f"region_{region_i:02d}_patch_prior.png", patch_prior)
            if args.non_tile_ma_seed:
                save_rgb(face_region_dir / f"region_{region_i:02d}_non_tile_seed.png", ma_seed_prior)
                save_mask(debug_dir / f"{face}_region_{region_i:02d}_non_tile_seed_keep.png", non_tile_keep)
                save_mask(debug_dir / f"{face}_region_{region_i:02d}_non_tile_seed_valid.png", non_tile_valid)
            save_rgb(face_region_dir / f"region_{region_i:02d}_tile.png", ref_tile)
            save_mask(debug_dir / f"{face}_region_{region_i:02d}_support.png", support)
            if keep_mask is not None:
                save_mask(debug_dir / f"{face}_region_{region_i:02d}_ma_keep_mask.png", keep_mask)
            region_stats.append(
                {
                    "region": region_i,
                    "method": method,
                    "error": error,
                    "quality": quality,
                    "non_tile_seed_method": non_tile_seed_method,
                    "brushnet_init_error": brushnet_error,
                    "score": float(cluster["score"]),
                    "tile_box_yx_size": [int(v) for v in tile_box],
                    "cluster_items": len(cluster["items"]),
                    "support_texels": int(np.count_nonzero(support)),
                    "valid_texels": int(np.count_nonzero(non_tile_valid)),
                    "mean_lab": [float(x) for x in cluster["mean_lab"]],
                    "edge_mean": float(cluster["edge_mean"]),
                    "sat_mean": float(cluster["sat_mean"]),
                }
            )

        composite, weights = blend_region_priors(
            args,
            face,
            image,
            masks,
            clusters,
            region_priors,
            region_supports,
            region_valids=region_valids,
        )
        if args.final_global_match_strength > 0.0:
            matched = match_lab_statistics(composite, image, masks.get("material_ref", masks["source"] | masks["high"]))
            composite = np.clip(
                composite * (1.0 - args.final_global_match_strength)
                + matched * args.final_global_match_strength,
                0.0,
                1.0,
            )
        save_rgb(prior_dir / f"{face}.png", composite)
        save_rgb(debug_dir / f"{face}_multi_ma_composite.png", composite)
        save_rgb(debug_dir / f"{face}_raw_projected.png", image)
        save_mask(debug_dir / f"{face}_source_mask.png", masks["source"])
        save_mask(debug_dir / f"{face}_high_mask.png", masks["high"])
        save_mask(debug_dir / f"{face}_reliability.png", masks["reliability"])
        for idx, weight in enumerate(weights):
            save_mask(debug_dir / f"{face}_region_{idx:02d}_blend_weight.png", weight)
        stats.append(
            {
                "face": face,
                "shape_hw": [int(shape[0]), int(shape[1])],
                "candidate_count": int(len(all_candidates)),
                "spread_candidate_count": int(len(spread)),
                "prior_count": int(len(clusters)),
                "regions": region_stats,
            }
        )
        print(f"[multi-ma] {face}: priors={len(clusters)}, candidates={len(all_candidates)}")

    with open(args.out_dir / "metadata_multi_material_priors.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "multi_material_anything_prior_field_v1",
                "summary": (
                    "Each face is over-sampled for high-confidence material support regions. "
                    "Color/texture-similar supports are merged, each remaining material region "
                    "runs the same patch/Material Anything prior generation path, and the resulting "
                    "full-face priors are smoothly blended before downstream observation fusion."
                ),
                "source_export_dir": str(args.input_dir),
                "backend": args.backend,
                "init_error": init_error,
                "faces": faces,
                "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "stats": stats,
            },
            f,
            indent=2,
        )
    print("[done] multi material priors:", prior_dir)


if __name__ == "__main__":
    main()
