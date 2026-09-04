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
ROUTE3B_SCRIPTS_DIR = PROJECT_DIR / "routes" / "v3b_material_route" / "scripts"
if str(ROUTE3B_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTE3B_SCRIPTS_DIR))

import build_polygon_photo_source_from_colmap as proj  # noqa: E402
import generate_material_priors as gmp  # noqa: E402
import generate_multi_material_priors as mm  # noqa: E402
import prepare_patch_preserving_tileable_inputs as route3b_patch  # noqa: E402


PBR_ALIASES = {
    "basecolor": ["basecolor", "base_color", "albedo", "diffuse", "color"],
    "normal": ["normal", "normal_map", "bump"],
    "roughness": ["roughness", "rough"],
    "metallic": ["metallic", "metalness", "metal"],
    "irradiance": ["irradiance", "approxIrr", "approx_irr"],
    "rou_met": ["rou_met", "roughness_metallic", "roughnessMetallic"],
}

TRACEABLE_CANDIDATE_TYPES = frozenset({"view_contributor_rectified"})


def traceability_errors(candidate: dict) -> list[str]:
    """Return hard provenance violations for one CHORD input candidate."""
    errors: list[str] = []
    stem = str(candidate.get("stem", "<missing-stem>"))
    if candidate.get("type") not in TRACEABLE_CANDIDATE_TYPES:
        errors.append(f"{stem}: non-traceable type={candidate.get('type')!r}")
    if candidate.get("input_mode") != "atlas_rectified":
        errors.append(f"{stem}: non-rectified input_mode={candidate.get('input_mode')!r}")
    if not candidate.get("view_name"):
        errors.append(f"{stem}: missing source view_name")
    if candidate.get("image_id") is None:
        errors.append(f"{stem}: missing source image_id")
    for key in ("chord_input", "candidate_mask"):
        value = candidate.get(key)
        if not value:
            errors.append(f"{stem}: missing {key}")
        elif not Path(value).is_file():
            errors.append(f"{stem}: {key} does not exist: {value}")
    return errors


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
    parser.add_argument(
        "--material-cluster-chroma-merge-threshold",
        type=float,
        default=0.55,
        help=(
            "Merge initial Lab components whose normalized chroma distance is below "
            "this value. This suppresses exposure-driven splits while preserving "
            "data-driven material counts."
        ),
    )
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
            "Augment per-pixel material clustering with automatic detection of narrow, "
            "horizontally persistent wall materials. The detected atlas interval is only "
            "a support mask: its actual CHORD input is still traced back to an original "
            "camera image and rectified through the normal v3b path."
        ),
    )
    parser.add_argument("--wall-band-min-height-frac", type=float, default=0.018)
    parser.add_argument("--wall-band-max-height-frac", type=float, default=0.12)
    parser.add_argument("--wall-band-context-frac", type=float, default=0.075)
    parser.add_argument("--wall-band-min-score", type=float, default=1.10)
    parser.add_argument(
        "--wall-band-min-texture-delta",
        type=float,
        default=0.18,
        help=(
            "Require an automatically added wall band to differ from both neighbouring "
            "regions in row-wise texture statistics. Colour-defined materials already "
            "come from Atlas material clustering; this gate prevents smooth lighting "
            "gradients from being promoted to extra material territories."
        ),
    )
    parser.add_argument("--wall-band-min-tangent-coverage", type=float, default=0.48)
    parser.add_argument("--wall-band-max-count", type=int, default=4)
    parser.add_argument("--wall-band-edge-merge-frac", type=float, default=0.016)
    parser.add_argument(
        "--remove-tiny-material-islands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional cleanup of disconnected discovery-mask islands. Disabled in the "
            "strict v3b-compatible route so small Atlas territories remain evidence."
        ),
    )
    parser.add_argument(
        "--reject-cross-face-edge-singletons",
        "--wall-reject-lateral-edge-singletons",
        dest="reject_cross_face_edge_singletons",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reject a planar material class only when it has one spatial exemplar, is "
            "confined to a narrow face-atlas edge, and covers only a small part of the "
            "face. Wall tests use lateral edges; floor/ceiling tests use all edges. "
            "This removes cross-face projection leakage without fixing K."
        ),
    )
    parser.add_argument(
        "--edge-singleton-max-width-frac",
        "--wall-edge-singleton-max-width-frac",
        dest="edge_singleton_max_width_frac",
        type=float,
        default=0.24,
    )
    parser.add_argument(
        "--edge-singleton-max-material-frac",
        "--wall-edge-singleton-max-material-frac",
        dest="edge_singleton_max_material_frac",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--edge-singleton-margin-frac",
        "--wall-edge-singleton-margin-frac",
        dest="edge_singleton_margin_frac",
        type=float,
        default=0.015,
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
    parser.add_argument(
        "--wall-lowres-single-view-source-adaptation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When a wall has no normal multi-view final support, admit a sufficiently "
            "large single-view Atlas observation that still passes the depth/face, "
            "object-risk, mask-boundary, and footprint gates. The admitted texels are "
            "used only to trace a rectified patch back to the original RGB view."
        ),
    )
    parser.add_argument("--wall-lowres-adapt-trigger-final-keep-frac", type=float, default=0.006)
    parser.add_argument("--wall-lowres-adapt-min-support-frac", type=float, default=0.006)
    parser.add_argument("--wall-lowres-adapt-clean-thresh", type=float, default=0.48)
    parser.add_argument("--wall-lowres-adapt-min-valid-views", type=int, default=1)
    parser.add_argument("--wall-lowres-adapt-object-risk-thresh", type=float, default=0.05)
    parser.add_argument("--wall-lowres-adapt-boundary-trust-thresh", type=float, default=0.55)
    parser.add_argument("--wall-lowres-adapt-footprint-min", type=float, default=0.008)
    parser.add_argument("--wall-lowres-min-rectified-valid-frac", type=float, default=0.012)
    parser.add_argument("--wall-lowres-rectified-min-size", type=int, default=16)
    parser.add_argument("--wall-lowres-min-final-source-unique-pixels", type=int, default=128)
    parser.add_argument("--wall-lowres-min-final-source-bbox-short-side-px", type=float, default=8.0)
    parser.add_argument(
        "--wall-lowres-planar-color-delta",
        type=float,
        default=1.60,
        help=(
            "Maximum normalized Lab distance from the strictly observed material "
            "when expanding a low-resolution wall patch inside the same visible plane."
        ),
    )
    parser.add_argument(
        "--floor-lowres-planar-color-delta",
        type=float,
        default=1.80,
        help=(
            "Maximum normalized Lab distance from the strictly observed floor "
            "material during same-plane low-resolution patch expansion."
        ),
    )

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
    parser.add_argument(
        "--surface-normal-min-cos",
        type=float,
        default=0.0,
        help=(
            "Optional DA3 depth-normal gate applied before selecting a source patch. "
            "It rejects a geometrically projected floor/ceiling/wall sample when the "
            "local observed surface orientation belongs to another face type."
        ),
    )
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
    parser.add_argument(
        "--rectified-inner-min-safe-frac",
        type=float,
        default=None,
        help=(
            "Minimum fraction remaining after mask-boundary erosion. When omitted, "
            "the valid-fraction threshold is used for both checks."
        ),
    )
    parser.add_argument("--rectified-inner-safe-border-px", type=int, default=10)
    parser.add_argument("--rectified-inner-stride-frac", type=float, default=0.08)
    parser.add_argument(
        "--rectified-search-box-scale",
        type=float,
        default=1.0,
        help=(
            "Expand the atlas-space search box around a discovered exemplar before "
            "original-view rectification. The discovered material mask is still enforced."
        ),
    )
    parser.add_argument("--rectified-search-box-min-size", type=int, default=0)
    parser.add_argument(
        "--floor-rectified-search-box-scale",
        type=float,
        default=None,
        help="Optional floor-only expansion of the Atlas search box before original-view rectification.",
    )
    parser.add_argument(
        "--floor-rectified-search-box-min-size",
        type=int,
        default=None,
        help="Optional floor-only minimum Atlas search-box size.",
    )
    parser.add_argument(
        "--wall-lowres-rectified-search-box-scale",
        type=float,
        default=2.0,
        help=(
            "Atlas search expansion used only by the low-resolution single-view "
            "wall fallback; material and strict source masks remain enforced."
        ),
    )
    parser.add_argument("--wall-lowres-rectified-search-box-min-size", type=int, default=1024)
    parser.add_argument(
        "--rectified-inner-fallback-min-size",
        type=int,
        default=0,
        help=(
            "If the texture-preserving crop is unavailable, retry the same original-view "
            "rectification with this smaller minimum size. Zero disables the retry."
        ),
    )
    parser.add_argument("--rectified-inner-fallback-min-valid-frac", type=float, default=0.995)
    parser.add_argument("--rectified-inner-fallback-min-safe-frac", type=float, default=0.995)
    parser.add_argument("--chord-input-size", type=int, default=512)
    parser.add_argument("--include-atlas-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--strict-v3b-material-provenance",
        action="store_true",
        help=(
            "Require every material candidate to be rectified from a real source view of the "
            "same face. Atlas/target fallbacks are forbidden and missing face-local evidence "
            "is a hard failure."
        ),
    )
    parser.add_argument(
        "--resolution-aware-atlas-candidate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add the fused atlas target as an additional candidate when every "
            "rectified view reuses too few distinct source pixels. This avoids "
            "magnifying a low-resolution crop into a textureless 512px patch."
        ),
    )
    parser.add_argument(
        "--min-rectified-source-density",
        type=float,
        default=0.35,
        help=(
            "Minimum ratio of unique source-image pixels to valid rectified "
            "texels before the view crop is considered sufficiently resolved."
        ),
    )
    parser.add_argument(
        "--rectified-source-density-penalty",
        type=float,
        default=0.55,
        help=(
            "Maximum compose-score penalty for a rectified candidate whose "
            "unique-source-pixel density falls below the configured minimum."
        ),
    )
    parser.add_argument(
        "--min-final-source-unique-pixels",
        type=int,
        default=0,
        help=(
            "Reject a rectified CHORD candidate when its final inner crop samples "
            "fewer distinct pixels from the original source image. Zero disables it."
        ),
    )
    parser.add_argument(
        "--min-final-source-bbox-short-side-px",
        type=float,
        default=0.0,
        help=(
            "Reject a final rectified crop whose exact original-image footprint has "
            "a shorter bounding-box side below this value. Zero disables it."
        ),
    )
    parser.add_argument(
        "--thin-territory-min-span-frac",
        type=float,
        default=0.30,
        help=(
            "Minimum Atlas-axis span for an automatically detected elongated material "
            "territory to use the thin-territory source-resolution gate."
        ),
    )
    parser.add_argument(
        "--thin-territory-max-thickness-frac",
        type=float,
        default=0.10,
        help=(
            "Maximum Atlas-axis thickness for the elongated-territory source gate. "
            "This changes only source-patch eligibility, never the discovered territory."
        ),
    )
    parser.add_argument("--thin-territory-min-source-unique-pixels", type=int, default=128)
    parser.add_argument("--thin-territory-min-source-bbox-short-side-px", type=float, default=4.0)
    parser.add_argument("--thin-territory-min-rectified-valid-frac", type=float, default=0.08)
    parser.add_argument("--thin-territory-rectified-search-min-size", type=int, default=512)
    parser.add_argument(
        "--thin-territory-source-adaptation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Legacy experimental shortcut for elongated Atlas regions. Disabled in the "
            "strict v3b-compatible route because it relaxes square-patch rectification "
            "and must not replace evidence-driven material placement."
        ),
    )
    parser.add_argument(
        "--floor-min-final-source-unique-pixels",
        type=int,
        default=None,
        help=(
            "Optional floor-only override for --min-final-source-unique-pixels. "
            "This lets low-resolution datasets require a wider floor footprint "
            "without discarding thin but real wall-material bands."
        ),
    )
    parser.add_argument(
        "--floor-min-final-source-bbox-short-side-px",
        type=float,
        default=None,
        help=(
            "Optional floor-only override for "
            "--min-final-source-bbox-short-side-px."
        ),
    )
    parser.add_argument(
        "--floor-lowres-source-adaptation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When an otherwise valid floor crop uses too small an original-image footprint, "
            "retry a clean rectangular crop in the same rectified material territory and "
            "expand it with the Route3B direction-preserving patch quilt before CHORD. The "
            "source view, face rectification, and Atlas material identity remain unchanged."
        ),
    )
    parser.add_argument("--floor-lowres-retry-source-unique-pixels", type=int, default=1800)
    parser.add_argument("--floor-lowres-retry-source-bbox-short-side-px", type=float, default=56.0)
    parser.add_argument("--floor-lowres-rectified-min-size", type=int, default=256)
    parser.add_argument("--floor-lowres-rectified-max-side-frac", type=float, default=0.98)
    parser.add_argument("--floor-lowres-rectified-min-valid-frac", type=float, default=0.45)
    parser.add_argument("--floor-lowres-rectified-min-safe-frac", type=float, default=0.40)
    parser.add_argument(
        "--final-source-resolution-score-weight",
        type=float,
        default=0.0,
        help="Optional candidate-selection reward for retaining more real source pixels.",
    )
    parser.add_argument("--final-source-resolution-reference-side-px", type=float, default=96.0)
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


def source_region_metrics(
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    height: int,
) -> dict:
    """Measure the exact original-image footprint of the final CHORD crop."""
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    finite = np.isfinite(u) & np.isfinite(v)
    u = u[finite]
    v = v[finite]
    if u.size == 0:
        return {
            "final_source_sample_count": 0,
            "final_source_unique_pixels": 0,
            "final_source_sampling_density": 0.0,
            "final_source_effective_side_px": 0.0,
            "final_source_bbox_width_px": 0,
            "final_source_bbox_height_px": 0,
            "final_source_bbox_short_side_px": 0,
        }
    x = np.clip(np.rint(u).astype(np.int64), 0, max(0, int(width) - 1))
    y = np.clip(np.rint(v).astype(np.int64), 0, max(0, int(height) - 1))
    unique_pixels = int(np.unique(y * int(width) + x).size)
    bbox_width = int(np.max(x) - np.min(x) + 1)
    bbox_height = int(np.max(y) - np.min(y) + 1)
    return {
        "final_source_sample_count": int(u.size),
        "final_source_unique_pixels": unique_pixels,
        "final_source_sampling_density": float(unique_pixels / max(int(u.size), 1)),
        "final_source_effective_side_px": float(math.sqrt(unique_pixels)),
        "final_source_bbox_width_px": bbox_width,
        "final_source_bbox_height_px": bbox_height,
        "final_source_bbox_short_side_px": int(min(bbox_width, bbox_height)),
    }


def final_source_gate_thresholds(args: argparse.Namespace, face: str) -> tuple[int, float]:
    min_unique = int(args.min_final_source_unique_pixels)
    min_bbox_short = float(args.min_final_source_bbox_short_side_px)
    if face == "floor":
        if args.floor_min_final_source_unique_pixels is not None:
            min_unique = int(args.floor_min_final_source_unique_pixels)
        if args.floor_min_final_source_bbox_short_side_px is not None:
            min_bbox_short = float(args.floor_min_final_source_bbox_short_side_px)
    return min_unique, min_bbox_short


def save_source_region_provenance(
    image: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    overlay_path: Path,
    crop_path: Path,
    footprint_path: Path | None = None,
    masked_crop_path: Path | None = None,
) -> dict:
    """Save where the selected material support lies in the original camera image."""
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    finite = np.isfinite(u) & np.isfinite(v)
    u = u[finite]
    v = v[finite]
    height, width = image.shape[:2]
    source_metrics = source_region_metrics(u, v, width, height)
    if u.size == 0:
        return {
            "source_region_bbox_xyxy": None,
            "source_region_hull_xy": None,
            "source_region_overlay": None,
            "source_region_crop": None,
            "source_region_footprint": None,
            "source_region_masked_crop": None,
            **source_metrics,
        }

    points = np.stack(
        [
            np.clip(np.rint(u), 0, width - 1),
            np.clip(np.rint(v), 0, height - 1),
        ],
        axis=1,
    ).astype(np.int32)
    hull = cv2.convexHull(points).reshape(-1, 2)
    x0 = int(np.min(points[:, 0]))
    y0 = int(np.min(points[:, 1]))
    x1 = int(np.max(points[:, 0])) + 1
    y1 = int(np.max(points[:, 1])) + 1

    rgb8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    footprint = np.zeros((height, width), dtype=np.uint8)
    footprint[points[:, 1], points[:, 0]] = 255
    # The exact sampling footprint can be sub-pixel sparse after rectification.
    # Dilate only the visualization copy so the source pixels remain visible;
    # the saved binary footprint itself stays exact.
    display_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    display_footprint = cv2.dilate(footprint, display_kernel, iterations=1) > 0
    overlay_rgb = rgb8.copy()
    overlay_rgb[display_footprint] = np.clip(
        0.58 * overlay_rgb[display_footprint]
        + 0.42 * np.array([32, 235, 210], dtype=np.float32),
        0,
        255,
    ).astype(np.uint8)
    overlay = Image.fromarray(overlay_rgb)
    draw = ImageDraw.Draw(overlay)
    hull_list = [(int(x), int(y)) for x, y in hull]
    line_width = max(2, int(round(min(width, height) / 180.0)))
    if len(hull_list) >= 3:
        draw.line(hull_list + [hull_list[0]], fill=(255, 32, 24), width=line_width)
    draw.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=(32, 220, 64), width=line_width)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(overlay_path)

    if footprint_path is not None:
        footprint_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(footprint).save(footprint_path)

    pad = max(4, int(round(min(width, height) * 0.015)))
    cx0 = max(0, x0 - pad)
    cy0 = max(0, y0 - pad)
    cx1 = min(width, x1 + pad)
    cy1 = min(height, y1 + pad)
    save_rgb(crop_path, image[cy0:cy1, cx0:cx1])
    if masked_crop_path is not None:
        masked_crop = rgb8[cy0:cy1, cx0:cx1].copy()
        crop_display_mask = display_footprint[cy0:cy1, cx0:cx1]
        masked_crop[~crop_display_mask] = np.clip(
            masked_crop[~crop_display_mask].astype(np.float32) * 0.22,
            0,
            255,
        ).astype(np.uint8)
        masked_crop_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(masked_crop).save(masked_crop_path)
    return {
        "source_region_bbox_xyxy": [x0, y0, x1, y1],
        "source_region_hull_xy": [[int(x), int(y)] for x, y in hull],
        "source_region_overlay": str(overlay_path),
        "source_region_crop": str(crop_path),
        "source_region_footprint": str(footprint_path) if footprint_path is not None else None,
        "source_region_masked_crop": str(masked_crop_path) if masked_crop_path is not None else None,
        **source_metrics,
    }


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

    # Structure3D's 512 px perspective views can leave a clean wall supported by
    # only one camera even though the Atlas projection itself is geometrically
    # valid.  Do not relax the normal multi-view mask globally.  Only when a wall
    # has effectively no final support, build a separate fallback from the same
    # strict projection diagnostics.  A non-trivial contiguous-looking area is
    # required, so narrow projection fragments do not become materials.
    lowres_wall_adapted = np.zeros(shape, dtype=bool)
    final_keep_fraction = float(np.mean(final_keep))
    if (
        args.wall_lowres_single_view_source_adaptation
        and face.startswith("wall_")
        and final_keep_fraction < args.wall_lowres_adapt_trigger_final_keep_frac
    ):
        adapted_candidate = (
            (valid_count >= args.wall_lowres_adapt_min_valid_views)
            & (clean_score >= args.wall_lowres_adapt_clean_thresh)
            & (object_risk <= args.wall_lowres_adapt_object_risk_thresh)
            & (mask_boundary_trust >= args.wall_lowres_adapt_boundary_trust_thresh)
            & (footprint_area >= args.wall_lowres_adapt_footprint_min)
            & (weight_sum > 1e-8)
        )
        # Require one substantial connected observation instead of summing
        # scattered slivers from unrelated projections.
        component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
            adapted_candidate.astype(np.uint8),
            connectivity=8,
        )
        if component_count > 1:
            largest_component = 1 + int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA]))
            adapted_candidate = component_labels == largest_component
        else:
            adapted_candidate[:] = False
        adapted_fraction = float(np.mean(adapted_candidate))
        if adapted_fraction >= args.wall_lowres_adapt_min_support_frac:
            lowres_wall_adapted = adapted_candidate
            print(
                f"[wall-lowres-support] {face}: normal={final_keep_fraction:.6f} "
                f"adapted={adapted_fraction:.6f} "
                f"valid_views>={args.wall_lowres_adapt_min_valid_views}",
                flush=True,
            )
        elif np.count_nonzero(adapted_candidate):
            print(
                f"[wall-lowres-support-reject] {face}: normal={final_keep_fraction:.6f} "
                f"adapted={adapted_fraction:.6f} "
                f"required={args.wall_lowres_adapt_min_support_frac:.6f}",
                flush=True,
            )

    support_keep = final_keep | lowres_wall_adapted

    positive = weight_sum[support_keep & (weight_sum > 1e-8)]
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

    if np.any(lowres_wall_adapted):
        adapted_count_rel = np.clip(
            valid_count / max(1, args.wall_lowres_adapt_min_valid_views),
            0.0,
            1.0,
        )
        adapted_footprint_rel = np.clip(
            footprint_area / max(args.wall_lowres_adapt_footprint_min, 1e-6),
            0.0,
            1.0,
        )
        adapted_reliability = (
            np.sqrt(adapted_count_rel * weight_rel)
            * clean_score
            * (1.0 - 0.62 * object_risk)
            * np.sqrt(np.clip(mask_boundary_trust, 0.0, 1.0))
            * np.sqrt(adapted_footprint_rel)
        ).astype(np.float32)
        reliability[lowres_wall_adapted] = adapted_reliability[lowres_wall_adapted]

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
    # Adapted texels have already passed their explicit low-resolution gates.
    # Treat them as source/high support for region discovery and contributor
    # traceback without weakening any normal multi-view threshold.
    source |= lowres_wall_adapted
    high |= lowres_wall_adapted
    if np.count_nonzero(source) < 0.006 * source.size:
        source = (final_keep & (valid_count >= args.min_valid_views)) | lowres_wall_adapted
    return {
        "valid_count": valid_count,
        "candidate_count": candidate_count,
        "clean_score": clean_score,
        "object_risk": object_risk,
        "weight_sum": weight_sum,
        "mask_boundary_trust": mask_boundary_trust,
        "footprint_area": footprint_area,
        "contaminant": ~support_keep,
        "observed": support_keep,
        "source": source,
        "high": high,
        "fill": ~support_keep,
        "mid": support_keep & ~high,
        "reliability": reliability,
        "lowres_wall_adapted": lowres_wall_adapted,
    }


def expanded_square_box(
    box: tuple[int, int, int],
    shape: tuple[int, int],
    scale: float,
    minimum_size: int,
) -> tuple[int, int, int]:
    y, x, size = [int(value) for value in box]
    h, w = shape
    target = max(size, int(round(size * max(float(scale), 1.0))), int(max(0, minimum_size)))
    target = min(target, h, w)
    cy = y + 0.5 * size
    cx = x + 0.5 * size
    out_y = int(round(cy - 0.5 * target))
    out_x = int(round(cx - 0.5 * target))
    out_y = max(0, min(out_y, h - target))
    out_x = max(0, min(out_x, w - target))
    return out_y, out_x, int(target)


def representative_support(
    shape: tuple[int, int],
    masks: dict,
    cluster: dict,
    box: tuple[int, int, int] | None = None,
) -> np.ndarray:
    y, x, size = [int(v) for v in (box or cluster["representative"]["box"])]
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


def material_territory_shape(
    material_mask: np.ndarray,
    face_shape: tuple[int, int],
    min_span_frac: float,
    max_thickness_frac: float,
) -> dict:
    """Describe the Atlas-space shape without changing its material territory.

    Long, thin territories (for example a trim, border, or narrow panel) can be
    perfectly clear in the face Atlas while projecting to only a few pixels on
    the short axis of a low-resolution source view.  The shape description lets
    source tracing apply an appropriate resolution gate without inventing a
    material or expanding its territory.
    """
    height, width = [int(value) for value in face_shape]
    ys, xs = np.nonzero(np.asarray(material_mask, dtype=bool))
    if ys.size == 0:
        return {
            "bbox_yxhw": None,
            "axis_span_fractions": [0.0, 0.0],
            "major_span_fraction": 0.0,
            "minor_thickness_fraction": 0.0,
            "elongation": 0.0,
            "is_thin_territory": False,
        }
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bbox_h, bbox_w = y1 - y0, x1 - x0
    y_fraction = float(bbox_h / max(height, 1))
    x_fraction = float(bbox_w / max(width, 1))
    major_span = max(y_fraction, x_fraction)
    minor_thickness = min(y_fraction, x_fraction)
    elongation = float(max(bbox_h, bbox_w) / max(min(bbox_h, bbox_w), 1))
    is_thin = bool(
        major_span >= float(min_span_frac)
        and minor_thickness <= float(max_thickness_frac)
        and elongation >= 3.0
    )
    return {
        "bbox_yxhw": [int(y0), int(x0), int(bbox_h), int(bbox_w)],
        "axis_span_fractions": [y_fraction, x_fraction],
        "major_span_fraction": major_span,
        "minor_thickness_fraction": minor_thickness,
        "elongation": elongation,
        "is_thin_territory": is_thin,
    }


def remove_tiny_material_islands(material_mask: np.ndarray) -> np.ndarray:
    """Remove only negligible disconnected Atlas islands from a material class.

    Atlas colour clustering can attach a few isolated shadow/projection pixels to
    an otherwise coherent material territory.  Those specks must not change the
    territory bounding geometry or seed a remote material patch.  Large disjoint
    territories are retained; the threshold is relative to the largest connected
    component and therefore does not impose a room- or material-specific count.
    """
    mask_u8 = np.asarray(material_mask, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 2:
        return mask_u8 > 0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest = int(np.max(areas)) if areas.size else 0
    minimum = max(48, int(round(0.01 * largest)))
    keep_labels = np.flatnonzero(areas >= minimum) + 1
    if keep_labels.size == 0:
        keep_labels = np.asarray([int(np.argmax(areas)) + 1], dtype=np.int32)
    return np.isin(labels, keep_labels)


def complete_thin_atlas_territory(
    material_mask: np.ndarray,
    observed: np.ndarray,
    territory_shape: dict,
) -> np.ndarray:
    """Complete an elongated material territory along its Atlas tangent axis.

    Multi-view occlusion leaves a baseboard/border as several disjoint observed
    pieces even though their common Atlas height makes the territory unambiguous.
    Completion is restricted to the detected territory's own axis-aligned bbox
    and therefore fills evidence gaps without changing K or extrapolating the
    material to unrelated rows/columns.  The completed mask is allowed across
    unobserved texels inside that interval so downstream Atlas placement can
    preserve a coherent territory through occlusions.
    """
    if not bool(territory_shape.get("is_thin_territory", False)):
        return np.asarray(material_mask, dtype=bool)
    bbox = territory_shape.get("bbox_yxhw")
    if not bbox:
        return np.asarray(material_mask, dtype=bool)
    y0, x0, bbox_h, bbox_w = [int(value) for value in bbox]
    y1, x1 = y0 + bbox_h, x0 + bbox_w
    completed = np.asarray(material_mask, dtype=bool).copy()
    if bbox_w >= bbox_h:
        completed[y0:y1, :] = True
    else:
        completed[:, x0:x1] = True
    return completed


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
            if chroma < float(args.material_cluster_chroma_merge_threshold) and lightness < 90.0:
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

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy) / 255.0
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
        if args.remove_tiny_material_islands:
            material_mask = remove_tiny_material_islands(material_mask) & observed
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
        territory_shape = material_territory_shape(
            material_mask,
            image.shape[:2],
            args.thin_territory_min_span_frac,
            args.thin_territory_max_thickness_frac,
        )
        if args.thin_territory_source_adaptation:
            material_mask = complete_thin_atlas_territory(
                material_mask,
                observed,
                territory_shape,
            )
        fraction = float(np.count_nonzero(material_mask) / max(np.count_nonzero(observed), 1))
        territory_shape = material_territory_shape(
            material_mask,
            image.shape[:2],
            args.thin_territory_min_span_frac,
            args.thin_territory_max_thickness_frac,
        )
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
                "territory_shape": territory_shape,
                "discovery_index": discovery_index,
            }
        )
        candidates.append(representative)

    # Atlas territories are identities as well as spatial regions.  Once an
    # elongated territory has been completed along its tangent direction, it
    # owns those observed texels; broad colour classes must not overlap it.
    thin_masks = [
        np.asarray(cluster["material_mask"], dtype=bool)
        for cluster in clusters
        if args.thin_territory_source_adaptation
        and bool(cluster.get("territory_shape", {}).get("is_thin_territory", False))
    ]
    if thin_masks:
        thin_union = np.logical_or.reduce(thin_masks)
        for cluster in clusters:
            if (
                args.thin_territory_source_adaptation
                and bool(cluster.get("territory_shape", {}).get("is_thin_territory", False))
            ):
                continue
            updated_mask = np.asarray(cluster["material_mask"], dtype=bool) & ~thin_union
            cluster["material_mask"] = updated_mask
            cluster["material_fraction"] = float(
                np.count_nonzero(updated_mask) / max(np.count_nonzero(observed), 1)
            )
            cluster["territory_shape"] = material_territory_shape(
                updated_mask,
                image.shape[:2],
                args.thin_territory_min_span_frac,
                args.thin_territory_max_thickness_frac,
            )

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
    """Add narrow layered materials that per-pixel colour clustering cannot represent.

    A patterned border is commonly composed of several colours. Pixel-wise Lab k-means
    therefore assigns its pixels to the broad materials above and below even though the
    border is a coherent material. This detector uses robust row-distribution descriptors
    and two persistent appearance transitions to propose an arbitrary number of narrow
    wall bands. It does not create a CHORD tile from the atlas; downstream contributor
    tracing still locates and rectifies the corresponding source-image observation.
    """
    audit = {"enabled": bool(args.discover_persistent_wall_bands), "bands": []}
    if not args.discover_persistent_wall_bands or not face.startswith("wall_"):
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
        gray = np.mean(values, axis=1)
        descriptor[y, 9] = float(np.std(gray))
        descriptor[y, 10] = float(np.mean(np.abs(np.diff(gray)))) if gray.size > 1 else 0.0

    valid_rows = np.flatnonzero(np.isfinite(descriptor[:, 0]) & (row_coverage >= 0.20))
    if valid_rows.size < max(32, int(round(0.25 * h))):
        audit["reason"] = "insufficient_observed_rows"
        return clusters, audit
    row_ids = np.arange(h)
    for channel in range(descriptor.shape[1]):
        descriptor[:, channel] = np.interp(
            row_ids,
            valid_rows,
            descriptor[valid_rows, channel],
        )
    descriptor /= np.asarray([35.0, 15.0, 15.0] * 3 + [25.0, 25.0], dtype=np.float32)
    sigma = max(1.0, 0.004 * h)
    smooth = cv2.GaussianBlur(descriptor.reshape(h, 1, -1), (0, 0), sigmaX=0.0, sigmaY=sigma).reshape(h, -1)
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

    min_height = max(int(args.material_cluster_min_region_size), int(round(args.wall_band_min_height_frac * h)))
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
            context = max(min_height, min(base_context, band_height))
            if top - context < 4 or bottom + context >= h - 4:
                continue
            band_value = np.mean(smooth[top + 3 : max(top + 4, bottom - 2)], axis=0)
            above_value = np.mean(smooth[top - context : top - 3], axis=0)
            below_value = np.mean(smooth[bottom + 3 : bottom + context], axis=0)
            above_delta = float(np.linalg.norm(band_value - above_value))
            below_delta = float(np.linalg.norm(band_value - below_value))
            context_delta = float(np.linalg.norm(above_value - below_value))
            above_texture_delta = float(np.linalg.norm(band_value[-2:] - above_value[-2:]))
            below_texture_delta = float(np.linalg.norm(band_value[-2:] - below_value[-2:]))
            texture_delta = min(above_texture_delta, below_texture_delta)
            score = (
                min(above_delta, below_delta)
                + 0.20 * float(transition[top] + transition[bottom])
                - 0.15 * context_delta
            )
            coverage = float(np.median(row_coverage[top : bottom + 1]))
            if (
                score < float(args.wall_band_min_score)
                or coverage < float(args.wall_band_min_tangent_coverage)
                or texture_delta < float(args.wall_band_min_texture_delta)
            ):
                continue
            proposals.append(
                {
                    "top": int(top),
                    "bottom": int(bottom),
                    "score": score,
                    "coverage": coverage,
                    "above_delta": above_delta,
                    "below_delta": below_delta,
                    "context_delta": context_delta,
                    "above_texture_delta": above_texture_delta,
                    "below_texture_delta": below_texture_delta,
                    "texture_delta": texture_delta,
                }
            )

    proposals.sort(key=lambda item: float(item["score"]), reverse=True)
    selected = []
    for proposal in proposals:
        top, bottom = int(proposal["top"]), int(proposal["bottom"])
        if any(max(top, int(other["top"])) < min(bottom, int(other["bottom"])) for other in selected):
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
        # Edge snapping must not turn a valid narrow-band proposal into a
        # broad illumination strip. Re-apply the same data-independent width
        # limit after expansion, before creating a material identity.
        if bottom - top > max_height:
            continue
        material_mask = np.zeros((h, w), dtype=bool)
        material_mask[max(0, top) : min(h, bottom + 1)] = observed[max(0, top) : min(h, bottom + 1)]
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
                np.count_nonzero(material_mask[box[0] : box[0] + box[2], box[1] : box[1] + box[2]])
                / max(np.count_nonzero(observed[box[0] : box[0] + box[2], box[1] : box[1] + box[2]]), 1)
            ),
            "source": "persistent_wall_band",
        }
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
                "territory_shape": material_territory_shape(
                    material_mask,
                    image.shape[:2],
                    args.thin_territory_min_span_frac,
                    args.thin_territory_max_thickness_frac,
                ),
                "discovery_index": f"wall_band_{len(band_clusters):02d}",
            }
        )
        audit["bands"].append(
            {
                **proposal,
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

    # A validated band owns its interval. Remove those pixels from broad colour
    # classes so seed masks remain disjoint when arbitrary-K placement begins.
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
        updated["territory_shape"] = material_territory_shape(
            updated_mask,
            image.shape[:2],
            args.thin_territory_min_span_frac,
            args.thin_territory_max_thickness_frac,
        )
        cleaned.append(updated)
    return cleaned + band_clusters, audit


def filter_cross_face_edge_singletons(
    args: argparse.Namespace,
    face: str,
    clusters: list[dict],
    shape_hw: tuple[int, int],
) -> tuple[list[dict], dict]:
    """Remove small one-off classes caused by adjacent-face projection leakage.

    The filter does not prescribe a material count.  A class is rejected only
    when all three observations agree: it has one spatial exemplar, that
    exemplar is anchored to a relevant atlas boundary, and both its width and
    face coverage are small.  Repeated classes and broad/layered wall
    materials therefore remain eligible regardless of their appearance.
    """
    audit = {
        "enabled": bool(args.reject_cross_face_edge_singletons),
        "rejected": [],
    }
    if (
        not args.reject_cross_face_edge_singletons
        or not clusters
    ):
        return clusters, audit

    height, width = shape_hw
    edge_margin_x = float(args.edge_singleton_margin_frac) * float(width)
    edge_margin_y = float(args.edge_singleton_margin_frac) * float(height)
    kept = []
    for cluster_index, cluster in enumerate(clusters):
        items = cluster.get("items", [cluster["representative"]])
        y, x, side = [int(value) for value in cluster["representative"]["box"]]
        touches_lateral_edge = bool(
            float(x) <= edge_margin_x
            or float(x + side) >= float(width) - edge_margin_x
        )
        touches_vertical_edge = bool(
            float(y) <= edge_margin_y
            or float(y + side) >= float(height) - edge_margin_y
        )
        touches_relevant_edge = bool(
            touches_lateral_edge
            or (not face.startswith("wall_") and touches_vertical_edge)
        )
        width_fraction = float(side) / max(float(min(height, width)), 1.0)
        material_fraction = float(cluster.get("material_fraction", 1.0))
        reject = bool(
            len(items) == 1
            and touches_relevant_edge
            and width_fraction <= float(args.edge_singleton_max_width_frac)
            and material_fraction <= float(args.edge_singleton_max_material_frac)
        )
        if reject:
            audit["rejected"].append(
                {
                    "cluster_index": int(cluster_index),
                    "box_yx_size": [int(y), int(x), int(side)],
                    "exemplar_count": int(len(items)),
                    "width_fraction": float(width_fraction),
                    "material_fraction": float(material_fraction),
                    "mean_lab": [float(value) for value in cluster["mean_lab"]],
                    "edge": (
                        "lateral"
                        if touches_lateral_edge
                        else "vertical"
                    ),
                    "reason": "singleton_small_cross_face_edge_class",
                }
            )
            continue
        kept.append(cluster)

    # Discovery must never be made empty by a cleanup heuristic.
    if not kept:
        best = max(clusters, key=lambda item: float(item.get("score", 0.0)))
        kept = [best]
        audit["fallback_kept_best_cluster"] = True
    return kept, audit


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        if args.surface_normal_min_cos > 0.0:
            surface_normal_cos, surface_normal_valid = proj.sampled_da3_normal_cos(
                da3_view,
                du,
                dv,
                proj.face_normal(face, manifest, metas),
                raw_to_room_matrix=raw_to_room_matrix,
                raw_to_room_similarity=sim if raw_to_room_matrix is None else None,
            )
            surface_normal_cos[~surface_normal_valid] = 0.0
        else:
            surface_normal_cos = np.ones(u.shape, dtype=np.float32)
    else:
        has_depth = np.zeros(u.shape, dtype=bool)
        camera_depth = np.full(u.shape, np.inf, dtype=np.float32)
        projected_depth = z
        surface_distance = np.full(u.shape, np.inf, dtype=np.float32)
        surface_normal_cos = np.zeros(u.shape, dtype=np.float32)
    return has_depth, camera_depth, projected_depth, surface_distance, sampled_conf, surface_normal_cos


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
        has_depth, camera_depth, projected_depth, surface_distance, sampled_conf, surface_normal_cos = depth_surface_for_pose(
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
        normal_ok = (
            surface_normal_cos >= float(args.surface_normal_min_cos)
            if args.surface_normal_min_cos > 0.0
            else np.ones(idx0.shape, dtype=bool)
        )
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
            & normal_ok
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
    min_safe_frac: float | None = None,
) -> tuple[tuple[int, int, int, int], dict] | None:
    """Find a sharp, central square surrounded by valid rectified source pixels."""
    h, w = valid_mask.shape
    max_side = int(round(min(h, w) * np.clip(max_side_frac, 0.2, 1.0)))
    min_size = int(max(24, min(min_size, max_side)))
    if max_side < min_size or np.count_nonzero(valid_mask) < min_size * min_size:
        return None

    safe_min_frac = float(min_valid_frac if min_safe_frac is None else min_safe_frac)
    safe_mask = valid_mask.astype(np.uint8)
    border = int(max(0, safe_border_px))
    if border > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1))
        safe_mask = cv2.erode(safe_mask, kernel, iterations=1)
    if np.count_nonzero(safe_mask) < min_size * min_size * safe_min_frac:
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
                if valid_frac < min_valid_frac or safe_frac < safe_min_frac:
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
                            "inner_crop_required_valid_frac": float(min_valid_frac),
                            "inner_crop_required_safe_frac": float(safe_min_frac),
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
        args.rectified_inner_min_safe_frac,
    )
    crop_policy = "texture_preserving_source_rectified"
    if crop_info is None and args.rectified_inner_fallback_min_size > 0:
        crop_info = best_rectified_inner_crop(
            image,
            valid_mask,
            args.rectified_inner_fallback_min_size,
            args.rectified_inner_max_side_frac,
            args.rectified_inner_fallback_min_valid_frac,
            args.rectified_inner_safe_border_px,
            args.rectified_inner_stride_frac,
            args.rectified_inner_fallback_min_safe_frac,
        )
        crop_policy = "strict_small_source_rectified_fallback"
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
    info["inner_crop_policy"] = crop_policy
    return chord_input, mask_crop, info


def largest_true_axis_aligned_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the largest all-valid axis-aligned rectangle as y0,y1,x0,x1."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return None
    height, width = mask.shape
    histogram = np.zeros(width, dtype=np.int32)
    best_area = 0
    best_box: tuple[int, int, int, int] | None = None
    for y in range(height):
        histogram = np.where(mask[y], histogram + 1, 0)
        stack: list[int] = []
        for x in range(width + 1):
            current = int(histogram[x]) if x < width else 0
            while stack and int(histogram[stack[-1]]) > current:
                index = stack.pop()
                rect_height = int(histogram[index])
                x0 = int(stack[-1] + 1) if stack else 0
                rect_width = int(x - x0)
                area = rect_height * rect_width
                if area > best_area:
                    best_area = area
                    best_box = (int(y + 1 - rect_height), int(y + 1), x0, int(x))
            stack.append(x)
    return best_box


def crop_rectified_lowres_floor_input(
    args: argparse.Namespace,
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Build a larger CHORD reference only from a clean rectified floor strip.

    Low-resolution views can map a strict 128--160 texel Atlas crop to only a
    few dozen source pixels.  Enlarging that tiny crop after rectification
    erases plank seams before CHORD sees them.  This fallback searches a larger
    rectangle inside the *same* rectified material territory and resizes that
    one rectangle once for CHORD. No patch is repeated and no rejected hole is
    inpainted. It does not use a fused Atlas patch or a raw perspective crop.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    ys, xs = np.nonzero(valid_mask)
    if ys.size < 64:
        return None
    # A tiny erosion keeps the source away from object/mask boundaries.
    safe_mask = cv2.erode(valid_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    crop_box = largest_true_axis_aligned_rectangle(safe_mask)
    if crop_box is None:
        return None
    y0, y1, x0, x1 = crop_box
    # Keep the complete clean rectangle so broad plank seams survive.  A single
    # square cut can land entirely inside one plank and falsely look uniform.
    crop = image[y0:y1, x0:x1].copy()
    source_mask = safe_mask[y0:y1, x0:x1].astype(bool)
    observed_valid_fraction = float(np.mean(source_mask))
    required_valid_fraction = min(
        float(args.floor_lowres_rectified_min_valid_frac),
        float(args.floor_lowres_rectified_min_safe_frac),
    )
    short_side = min(crop.shape[:2])
    if (
        int(np.count_nonzero(source_mask)) < 64
        or short_side < 24
        or observed_valid_fraction < max(0.995, required_valid_fraction)
    ):
        return None
    out_size = int(args.chord_input_size)
    if out_size <= 0:
        out_size = max(crop.shape[:2])
    fixed = (
        cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
        if crop.shape[:2] != (out_size, out_size)
        else crop
    )
    mask_out = np.ones((out_size, out_size), dtype=bool)
    info = {
        "inner_crop_box_y0_y1_x0_x1": [int(v) for v in crop_box],
        "inner_crop_shape_hw": [int(crop.shape[0]), int(crop.shape[1])],
        "inner_crop_valid_frac": observed_valid_fraction,
        "inner_crop_safe_frac": observed_valid_fraction,
        "inner_crop_output_valid_frac": 1.0,
        "inner_crop_observed_valid_frac_before_inpaint": observed_valid_fraction,
        "inner_crop_policy": "lowres_floor_single_clean_rectified_rectangle_resize",
    }
    return np.clip(fixed, 0.0, 1.0), mask_out, info


def crop_rectified_lowres_wall_input(
    args: argparse.Namespace,
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Upscale one entirely valid square from a low-resolution wall view.

    This remains an original-view traceback: RGB is sampled from the source
    camera by ``rectified_tile_from_view`` and no Atlas colour is substituted.
    The largest all-valid rectangle prevents rejected door/window/object pixels
    from being filled into the material reference.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    safe_mask = cv2.erode(
        valid_mask.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    crop_box = largest_true_axis_aligned_rectangle(safe_mask)
    if crop_box is None:
        crop_box = largest_true_axis_aligned_rectangle(valid_mask)
    all_valid_square = False
    if crop_box is not None:
        y0, y1, x0, x1 = [int(value) for value in crop_box]
        side = min(y1 - y0, x1 - x0)
        if side >= int(args.wall_lowres_rectified_min_size):
            # Centre the square inside the all-valid rectangle; do not stretch
            # a thin strip into a false material texture.
            y0 += (y1 - y0 - side) // 2
            x0 += (x1 - x0 - side) // 2
            y1 = y0 + side
            x1 = x0 + side
            all_valid_square = True

    if not all_valid_square:
        # If the valid observation is oblique/irregular, retain its connected
        # rectified footprint and fill only holes inside that footprint.  This
        # is the same masked-source preparation used for thin Atlas materials;
        # it is still derived solely from the original camera pixels.
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            valid_mask.astype(np.uint8),
            connectivity=8,
        )
        if component_count <= 1:
            return None
        component_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = labels == component_id
        y0 = int(stats[component_id, cv2.CC_STAT_TOP])
        x0 = int(stats[component_id, cv2.CC_STAT_LEFT])
        y1 = y0 + int(stats[component_id, cv2.CC_STAT_HEIGHT])
        x1 = x0 + int(stats[component_id, cv2.CC_STAT_WIDTH])
        if min(y1 - y0, x1 - x0) < 4:
            return None
        crop_mask = component[y0:y1, x0:x1]
        if int(np.count_nonzero(crop_mask)) < 64 or float(np.mean(crop_mask)) < 0.18:
            return None
        crop = gmp.inpaint_tile_holes(image[y0:y1, x0:x1].copy(), crop_mask)
        crop_policy = "lowres_wall_connected_support_rectified_rectangle"
    else:
        crop = image[y0:y1, x0:x1].copy()
        crop_mask = valid_mask[y0:y1, x0:x1]
        if float(np.mean(crop_mask)) < 0.995:
            return None
        crop_policy = "lowres_wall_largest_all_valid_source_rectified_square"
    side = min(y1 - y0, x1 - x0)
    out_size = int(args.chord_input_size)
    if out_size > 0 and side != out_size:
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
        mask_out = cv2.resize(
            crop_mask.astype(np.uint8),
            (out_size, out_size),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    else:
        mask_out = crop_mask.astype(bool)
    return (
        np.clip(crop, 0.0, 1.0),
        mask_out,
        {
            "inner_crop_box_y0_y1_x0_x1": [int(y0), int(y1), int(x0), int(x1)],
            "inner_crop_side": int(side),
            "inner_crop_valid_frac": float(np.mean(crop_mask)),
            "inner_crop_safe_frac": float(np.mean(crop_mask)),
            "inner_crop_output_valid_frac": float(np.mean(mask_out)),
            "inner_crop_policy": crop_policy,
        },
    )


def crop_rectified_thin_territory_input(
    args: argparse.Namespace,
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Create a CHORD input from an elongated, Atlas-rectified source region.

    Requiring a large square of valid source pixels is appropriate for broad
    wall/floor materials but systematically deletes trims and narrow panels.
    For a territory already established in the Atlas, retain its rectangular
    rectified footprint, fill only holes inside that footprint, and map that
    rectangle to CHORD's square input.  Placement later restores the original
    Atlas territory, so this resize cannot change the material boundary.
    """
    mask_u8 = np.asarray(valid_mask, dtype=np.uint8)
    if int(np.count_nonzero(mask_u8)) < 64:
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return None
    best_label = max(
        range(1, count),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA])
        * max(
            int(stats[index, cv2.CC_STAT_WIDTH]),
            int(stats[index, cv2.CC_STAT_HEIGHT]),
        ),
    )
    component = labels == best_label
    ys, xs = np.nonzero(component)
    if ys.size < 64:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if min(y1 - y0, x1 - x0) < 3:
        return None
    crop = image[y0:y1, x0:x1].copy()
    crop_mask = valid_mask[y0:y1, x0:x1].astype(bool)
    source_valid_fraction = float(np.mean(crop_mask))
    if source_valid_fraction < 0.18:
        return None
    fixed = gmp.inpaint_tile_holes(crop, crop_mask)
    out_size = int(args.chord_input_size)
    if out_size > 0:
        fixed = cv2.resize(fixed, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
        mask_out = cv2.resize(
            crop_mask.astype(np.uint8),
            (out_size, out_size),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    else:
        mask_out = crop_mask
    return (
        np.clip(fixed, 0.0, 1.0),
        mask_out,
        {
            "inner_crop_box_y0_y1_x0_x1": [int(y0), int(y1), int(x0), int(x1)],
            "inner_crop_side": int(min(y1 - y0, x1 - x0)),
            "inner_crop_valid_frac": source_valid_fraction,
            "inner_crop_safe_frac": source_valid_fraction,
            "inner_crop_output_valid_frac": float(np.mean(mask_out)),
            "inner_crop_policy": "atlas_thin_territory_rectified_rectangle",
            "inner_crop_rectified_shape_hw": [int(y1 - y0), int(x1 - x0)],
        },
    )


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
    min_valid_fraction: float | None = None,
    lowres_mask_safe_planar_expansion: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray] | None:
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
    has_depth, camera_depth, projected_depth, surface_distance, sampled_conf, surface_normal_cos = depth_surface_for_pose(
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
    normal_ok = (
        surface_normal_cos >= float(args.surface_normal_min_cos)
        if args.surface_normal_min_cos > 0.0
        else np.ones(rows.shape, dtype=bool)
    )
    mask_safe_face_projection = (
        target_mask.reshape(-1).astype(bool)
        & in_frame
        & (sampled_face == face_idx)
        & (sampled_object == 0)
        & (sampled_object_risk <= args.object_risk_hard_thresh)
        & (sampled_boundary >= args.min_mask_boundary_trust)
    )
    if lowres_mask_safe_planar_expansion:
        # The strict contributor support has already selected this wall/view.
        # For a 512 px source, DA3 depth noise can collapse the usable patch to
        # a few dozen pixels. Expand only inside the same z-buffer-visible wall
        # and the same SAM-safe source region; never borrow another face or a
        # masked door/window/object pixel.
        valid = mask_safe_face_projection
    else:
        valid = (
            mask_safe_face_projection
            & has_depth
            & np.isfinite(depth_diff)
            & (sampled_conf >= args.min_conf)
            & (depth_diff <= depth_tol)
            & surface_ok
            & normal_ok
        )
    valid_mask = valid.reshape(tile_h, tile_w)
    target_count = max(1, int(np.count_nonzero(target_mask)))
    valid_frac = float(np.count_nonzero(valid_mask) / target_count)
    required_valid_fraction = (
        float(args.min_rectified_valid_frac)
        if min_valid_fraction is None
        else float(min_valid_fraction)
    )
    if valid_frac < required_valid_fraction:
        print(
            f"[rectified-valid-reject] {face} {pose.name}: rectified_valid="
            f"{valid_frac:.4f} required={required_valid_fraction:.4f} "
            f"target_texels={target_count}",
            flush=True,
        )
        return None

    valid_count = int(np.count_nonzero(valid))
    if valid_count > 0:
        source_x = np.clip(np.rint(u[valid]).astype(np.int64), 0, pose.width - 1)
        source_y = np.clip(np.rint(v[valid]).astype(np.int64), 0, pose.height - 1)
        unique_source_pixels = int(np.unique(source_y * int(pose.width) + source_x).size)
    else:
        unique_source_pixels = 0
    source_sampling_density = float(unique_source_pixels / max(valid_count, 1))

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
    diagnostic_depth = valid & has_depth & np.isfinite(depth_diff) & np.isfinite(depth_tol)
    diagnostic_surface = valid & np.isfinite(surface_distance)
    diagnostic_normal = valid & np.isfinite(surface_normal_cos)
    info = {
        "rectified_valid_frac": valid_frac,
        "rectified_required_valid_frac": required_valid_fraction,
        "rectified_valid_texels": valid_count,
        "rectified_unique_source_pixels": unique_source_pixels,
        "rectified_source_sampling_density": source_sampling_density,
        "rectified_effective_source_side": float(math.sqrt(unique_source_pixels)),
        "mean_view_cos": float(np.mean(cos[valid])) if np.any(valid) else 0.0,
        "mean_depth_residual": (
            float(np.mean(depth_diff[diagnostic_depth] / np.maximum(depth_tol[diagnostic_depth], 1e-6)))
            if np.any(diagnostic_depth)
            else None
        ),
        "mean_surface_distance": (
            float(np.mean(surface_distance[diagnostic_surface]))
            if np.any(diagnostic_surface)
            else None
        ),
        "mean_surface_normal_cos": (
            float(np.mean(surface_normal_cos[diagnostic_normal]))
            if np.any(diagnostic_normal)
            else None
        ),
        "rectified_validation_policy": (
            "strict_contributor_then_mask_safe_same_wall_planar_expansion"
            if lowres_mask_safe_planar_expansion
            else "strict_depth_surface_face_and_object"
        ),
    }
    # Keep the face-texel -> source-image map alongside the rectified RGB.
    # This is used only for provenance/QA: the CHORD input is still sampled
    # directly from the original view above.  Returning the map prevents the
    # audit image from accidentally drawing the bounding box of the entire
    # material support instead of the exact inner patch sent to CHORD.
    return (
        np.clip(colors, 0.0, 1.0),
        valid_mask_out,
        info,
        u.reshape(tile_h, tile_w).astype(np.float32),
        v.reshape(tile_h, tile_w).astype(np.float32),
    )


def write_region_view_inputs(
    args: argparse.Namespace,
    face: str,
    region_i: int,
    box: tuple[int, int, int],
    contributors: list[dict],
    target_tile: np.ndarray,
    target_mask: np.ndarray,
    territory_shape: dict | None,
    sim: proj.Similarity,
    manifest: dict,
    metas: dict,
    all_faces: list[str],
    da3_views: dict[int, proj.Da3View],
    raw_to_room_matrix: np.ndarray | None,
    caches: dict,
    out_dirs: dict,
    lowres_wall_source: bool = False,
) -> list[dict]:
    total_weight = sum(float(item.get("weight", 0.0)) for item in contributors)
    selected = []
    min_final_unique, min_final_bbox_short = final_source_gate_thresholds(args, face)
    source_gate_policy = "standard_material_region"
    is_thin_territory = bool(
        args.thin_territory_source_adaptation
        and territory_shape
        and territory_shape.get("is_thin_territory", False)
    )
    if is_thin_territory:
        min_final_unique = min(
            min_final_unique,
            int(args.thin_territory_min_source_unique_pixels),
        )
        min_final_bbox_short = min(
            min_final_bbox_short,
            float(args.thin_territory_min_source_bbox_short_side_px),
        )
        source_gate_policy = "atlas_thin_territory_adaptive"
    if lowres_wall_source:
        min_final_unique = max(
            int(min_final_unique),
            int(args.wall_lowres_min_final_source_unique_pixels),
        )
        min_final_bbox_short = max(
            float(min_final_bbox_short),
            float(args.wall_lowres_min_final_source_bbox_short_side_px),
        )
        source_gate_policy = "lowres_wall_single_view_strict_traceback"
    lowres_floor_source = bool(face == "floor" and args.floor_lowres_source_adaptation)
    lowres_planar_source = bool(lowres_wall_source or lowres_floor_source)

    # The strict Atlas support chooses the material and contributor.  For the
    # low-resolution fallback only, sample a broader same-plane window; after
    # projection it is intersected with the observed material's appearance.
    rectification_target_mask = (
        np.ones_like(target_mask, dtype=bool)
        if lowres_planar_source
        else target_mask
    )

    def append_atlas_candidate(reason: str, best_density: float | None) -> None:
        if args.strict_v3b_material_provenance:
            raise AssertionError(
                f"{face} region {region_i}: strict provenance forbids atlas fallback ({reason})"
            )
        stem = f"{face}_r{region_i:02d}_atlas_fallback"
        atlas_input = target_tile
        atlas_mask = target_mask.astype(bool)
        if args.chord_input_size > 0 and atlas_input.shape[:2] != (args.chord_input_size, args.chord_input_size):
            atlas_input = cv2.resize(
                atlas_input,
                (args.chord_input_size, args.chord_input_size),
                interpolation=cv2.INTER_CUBIC,
            )
            atlas_mask = cv2.resize(
                atlas_mask.astype(np.uint8),
                (args.chord_input_size, args.chord_input_size),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        save_rgb(out_dirs["chord_inputs"] / f"{stem}.png", atlas_input)
        save_mask(out_dirs["candidate_masks"] / f"{stem}_mask.png", atlas_mask)
        save_rgb(out_dirs["candidate_crops"] / f"{stem}_original_crop.png", atlas_input)
        selected.append(
            {
                "stem": stem,
                "type": "atlas_fallback",
                "view_name": None,
                "image_id": None,
                "input_mode": "fused_atlas_target",
                "weight": 0.0,
                "weight_frac": 0.0,
                "valid_sample_count": int(np.count_nonzero(atlas_mask)),
                "crop_box_y0_y1_x0_x1": None,
                "mask_pixels": int(np.count_nonzero(atlas_mask)),
                "mask_fraction_in_chord_input": float(np.mean(atlas_mask)),
                "chord_input": str(out_dirs["chord_inputs"] / f"{stem}.png"),
                "candidate_mask": str(out_dirs["candidate_masks"] / f"{stem}_mask.png"),
                "candidate_overlay": None,
                "mean_depth_residual": None,
                "mean_surface_distance": None,
                "atlas_fallback_reason": reason,
                "best_rectified_source_sampling_density": (
                    float(best_density) if best_density is not None else None
                ),
            }
        )

    if args.chord_input_mode == "atlas_rectified":
        evaluated = []
        rejection_counts = {
            "low_contributor_weight": 0,
            "rectification_failed": 0,
            "no_strict_square_crop": 0,
            "insufficient_unique_source_pixels": 0,
            "insufficient_source_bbox_short_side": 0,
        }
        for rank, item in enumerate(contributors[: max(args.max_source_views_eval, args.max_view_candidates)]):
            if total_weight > 0 and item.get("weight", 0.0) / total_weight < args.min_view_weight_frac * 0.45:
                rejection_counts["low_contributor_weight"] += 1
                continue
            pose: proj.ImagePose = item["pose"]
            image = load_rgb(pose.image_path)
            rectified = rectified_tile_from_view(
                args,
                face,
                box,
                rectification_target_mask,
                pose,
                image,
                sim,
                manifest,
                metas,
                all_faces,
                da3_views,
                raw_to_room_matrix,
                caches,
                (
                    min(float(args.min_rectified_valid_frac), float(args.thin_territory_min_rectified_valid_frac))
                    if is_thin_territory
                    else (
                        min(float(args.min_rectified_valid_frac), float(args.wall_lowres_min_rectified_valid_frac))
                        if lowres_wall_source
                        else (
                            min(
                                float(args.min_rectified_valid_frac),
                                float(args.floor_lowres_rectified_min_valid_frac),
                            )
                            if lowres_floor_source
                            else None
                        )
                    )
                ),
                lowres_mask_safe_planar_expansion=lowres_planar_source,
            )
            if rectified is None:
                rejection_counts["rectification_failed"] += 1
                continue
            rectified_full, rectified_mask, extra_info, source_u_grid, source_v_grid = rectified
            if lowres_planar_source:
                target_values = target_tile[target_mask.astype(bool)]
                if target_values.size:
                    target_rgb8 = np.clip(
                        target_values.reshape(-1, 1, 3) * 255.0,
                        0,
                        255,
                    ).astype(np.uint8)
                    target_lab = np.median(
                        cv2.cvtColor(target_rgb8, cv2.COLOR_RGB2LAB).reshape(-1, 3),
                        axis=0,
                    ).astype(np.float32)
                    rectified_lab = cv2.cvtColor(
                        np.clip(rectified_full * 255.0, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2LAB,
                    ).astype(np.float32)
                    delta = rectified_lab - target_lab.reshape(1, 1, 3)
                    normalized_delta = np.sqrt(
                        (delta[..., 0] / 30.0) ** 2
                        + (delta[..., 1] / 11.0) ** 2
                        + (delta[..., 2] / 11.0) ** 2
                    )
                    before_color_gate = int(np.count_nonzero(rectified_mask))
                    planar_color_delta = float(
                        args.wall_lowres_planar_color_delta
                        if lowres_wall_source
                        else args.floor_lowres_planar_color_delta
                    )
                    rectified_mask &= normalized_delta <= planar_color_delta
                    after_color_gate = int(np.count_nonzero(rectified_mask))
                    extra_info["lowres_planar_color_delta"] = planar_color_delta
                    extra_info["lowres_pixels_before_color_gate"] = before_color_gate
                    extra_info["lowres_pixels_after_color_gate"] = after_color_gate
                    print(
                        f"[lowres-planar-trace] {face} r{region_i:02d} {pose.name}: "
                        f"mask_safe={before_color_gate} color_compatible={after_color_gate}",
                        flush=True,
                    )
            if args.rectified_inner_crop:
                if is_thin_territory:
                    cropped = crop_rectified_thin_territory_input(args, rectified_full, rectified_mask)
                elif lowres_wall_source:
                    cropped = crop_rectified_lowres_wall_input(args, rectified_full, rectified_mask)
                else:
                    cropped = crop_rectified_chord_input(args, rectified_full, rectified_mask)
                if cropped is None:
                    rejection_counts["no_strict_square_crop"] += 1
                    if is_thin_territory:
                        print(
                            f"[thin-trace-reject] {face} r{region_i:02d} {pose.name}: "
                            f"no_valid_rectified_rectangle pixels={int(np.count_nonzero(rectified_mask))}",
                            flush=True,
                        )
                    continue
                chord_input, mask_crop, crop_info = cropped
                extra_info.update(crop_info)
                crop_y0, crop_y1, crop_x0, crop_x1 = crop_info["inner_crop_box_y0_y1_x0_x1"]
                provenance_mask = rectified_mask[crop_y0:crop_y1, crop_x0:crop_x1]
                provenance_u = source_u_grid[crop_y0:crop_y1, crop_x0:crop_x1][provenance_mask]
                provenance_v = source_v_grid[crop_y0:crop_y1, crop_x0:crop_x1][provenance_mask]
            else:
                chord_input, mask_crop = rectified_full, rectified_mask
                provenance_u = source_u_grid[rectified_mask]
                provenance_v = source_v_grid[rectified_mask]
            final_source_info = source_region_metrics(
                provenance_u,
                provenance_v,
                pose.width,
                pose.height,
            )
            needs_floor_resolution_retry = bool(
                face == "floor"
                and args.floor_lowres_source_adaptation
                and (
                    (
                        args.floor_lowres_retry_source_unique_pixels > 0
                        and final_source_info["final_source_unique_pixels"]
                        < int(args.floor_lowres_retry_source_unique_pixels)
                    )
                    or (
                        args.floor_lowres_retry_source_bbox_short_side_px > 0.0
                        and final_source_info["final_source_bbox_short_side_px"]
                        < float(args.floor_lowres_retry_source_bbox_short_side_px)
                    )
                )
            )
            if needs_floor_resolution_retry and args.rectified_inner_crop:
                adapted = crop_rectified_lowres_floor_input(args, rectified_full, rectified_mask)
                if adapted is not None:
                    adapted_input, adapted_mask_crop, adapted_crop_info = adapted
                    ay0, ay1, ax0, ax1 = adapted_crop_info["inner_crop_box_y0_y1_x0_x1"]
                    adapted_provenance_mask = rectified_mask[ay0:ay1, ax0:ax1]
                    adapted_u = source_u_grid[ay0:ay1, ax0:ax1][adapted_provenance_mask]
                    adapted_v = source_v_grid[ay0:ay1, ax0:ax1][adapted_provenance_mask]
                    adapted_source_info = source_region_metrics(
                        adapted_u,
                        adapted_v,
                        pose.width,
                        pose.height,
                    )
                    old_resolution = (
                        int(final_source_info["final_source_bbox_short_side_px"]),
                        int(final_source_info["final_source_unique_pixels"]),
                    )
                    new_resolution = (
                        int(adapted_source_info["final_source_bbox_short_side_px"]),
                        int(adapted_source_info["final_source_unique_pixels"]),
                    )
                    print(
                        f"[floor-lowres-trace] {face} r{region_i:02d} {pose.name}: "
                        f"bbox_short={old_resolution[0]}->{new_resolution[0]} "
                        f"unique={old_resolution[1]}->{new_resolution[1]} "
                        f"valid={float(adapted_crop_info.get('inner_crop_observed_valid_frac_before_inpaint', 0.0)):.3f}",
                        flush=True,
                    )
                    adapted_meets_acceptance_gate = bool(
                        (min_final_unique <= 0 or new_resolution[1] >= min_final_unique)
                        and (min_final_bbox_short <= 0.0 or new_resolution[0] >= min_final_bbox_short)
                    )
                    is_clean_lowres_reference = (
                        adapted_crop_info.get("inner_crop_policy")
                        == "lowres_floor_single_clean_rectified_rectangle_resize"
                    )
                    if adapted_meets_acceptance_gate and (
                        new_resolution > old_resolution or is_clean_lowres_reference
                    ):
                        chord_input = adapted_input
                        mask_crop = adapted_mask_crop
                        provenance_u = adapted_u
                        provenance_v = adapted_v
                        final_source_info = adapted_source_info
                        extra_info.update(adapted_crop_info)
                        extra_info["floor_lowres_resolution_before"] = {
                            "bbox_short_side_px": old_resolution[0],
                            "unique_source_pixels": old_resolution[1],
                        }
                        extra_info["floor_lowres_resolution_after"] = {
                            "bbox_short_side_px": new_resolution[0],
                            "unique_source_pixels": new_resolution[1],
                        }
            if (
                min_final_unique > 0
                and final_source_info["final_source_unique_pixels"]
                < min_final_unique
            ):
                rejection_counts["insufficient_unique_source_pixels"] += 1
                if is_thin_territory or lowres_wall_source:
                    print(
                        f"[source-resolution-reject] {face} r{region_i:02d} {pose.name}: "
                        f"unique={final_source_info['final_source_unique_pixels']} "
                        f"required={min_final_unique}",
                        flush=True,
                    )
                continue
            if (
                min_final_bbox_short > 0.0
                and final_source_info["final_source_bbox_short_side_px"]
                < min_final_bbox_short
            ):
                rejection_counts["insufficient_source_bbox_short_side"] += 1
                if is_thin_territory or lowres_wall_source:
                    print(
                        f"[source-resolution-reject] {face} r{region_i:02d} {pose.name}: "
                        f"bbox_short={final_source_info['final_source_bbox_short_side_px']} "
                        f"required={min_final_bbox_short}",
                        flush=True,
                    )
                continue
            extra_info.update(final_source_info)
            extra_info["source_resolution_gate_policy"] = source_gate_policy
            extra_info["source_resolution_gate_min_unique_pixels"] = int(min_final_unique)
            extra_info["source_resolution_gate_min_bbox_short_side_px"] = float(min_final_bbox_short)
            source_resolution_score = min(
                float(final_source_info["final_source_effective_side_px"])
                / max(float(args.final_source_resolution_reference_side_px), 1e-6),
                1.0,
            )
            q = (
                2.10 * float(extra_info.get("rectified_valid_frac", 0.0))
                + 0.80 * float(extra_info.get("mean_view_cos", 0.0))
                + 0.55 * float(item.get("weight_frac", 0.0))
                + 0.20 * float(extra_info.get("inner_crop_score", 0.0))
                + float(args.final_source_resolution_score_weight) * source_resolution_score
                - 0.35 * float(extra_info.get("mean_depth_residual") or 0.0)
                - 1.20 * max(0.0, args.min_rectified_valid_frac - float(extra_info.get("rectified_valid_frac", 0.0)))
            )
            # Keep provenance coordinates bound to the same candidate.  They
            # must survive sorting together with the rectified CHORD input.
            evaluated.append(
                (
                    q,
                    rank,
                    item,
                    pose,
                    chord_input,
                    mask_crop,
                    extra_info,
                    provenance_u.copy(),
                    provenance_v.copy(),
                )
            )
        evaluated.sort(key=lambda x: x[0], reverse=True)
        for (
            q,
            rank,
            item,
            pose,
            chord_input,
            mask_crop,
            extra_info,
            provenance_u,
            provenance_v,
        ) in evaluated[: args.max_view_candidates]:
            stem = f"{face}_r{region_i:02d}_v{len(selected):02d}_{Path(pose.name).stem}"
            save_rgb(out_dirs["chord_inputs"] / f"{stem}.png", chord_input)
            save_mask(out_dirs["candidate_masks"] / f"{stem}_mask.png", mask_crop)
            save_rgb(out_dirs["candidate_crops"] / f"{stem}_original_crop.png", chord_input)
            overlay = chord_input.copy()
            overlay[mask_crop] = 0.58 * overlay[mask_crop] + 0.42 * np.array([1.0, 0.08, 0.04], dtype=np.float32)
            save_rgb(out_dirs["candidate_overlays"] / f"{stem}_overlay.png", overlay)
            source_info = save_source_region_provenance(
                load_rgb(pose.image_path),
                provenance_u,
                provenance_v,
                out_dirs["source_region_overlays"] / f"{stem}_source_overlay.png",
                out_dirs["source_region_crops"] / f"{stem}_source_crop.png",
                out_dirs["source_region_footprints"] / f"{stem}_source_footprint.png",
                out_dirs["source_region_masked_crops"] / f"{stem}_source_masked_crop.png",
            )
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
                    **source_info,
                    **extra_info,
                    "chord_input": str(out_dirs["chord_inputs"] / f"{stem}.png"),
                    "candidate_mask": str(out_dirs["candidate_masks"] / f"{stem}_mask.png"),
                    "candidate_overlay": str(out_dirs["candidate_overlays"] / f"{stem}_overlay.png"),
                    "mean_depth_residual": float(item["mean_depth_residual"]),
                    "mean_surface_distance": float(item["mean_surface_distance"]),
                }
            )
        if selected:
            if args.resolution_aware_atlas_candidate:
                best_density = max(
                    float(item.get("rectified_source_sampling_density", 0.0))
                    for item in selected
                )
                if best_density < args.min_rectified_source_density:
                    append_atlas_candidate("low_rectified_source_density", best_density)
            return selected
        if not args.include_atlas_fallback:
            print(
                f"[trace-reject-summary] {face} r{region_i:02d} box={tuple(int(v) for v in box)} "
                f"contributors={len(contributors)} rejections={rejection_counts}",
                flush=True,
            )
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
                (
                    min(float(args.min_rectified_valid_frac), float(args.thin_territory_min_rectified_valid_frac))
                    if is_thin_territory
                    else None
                ),
            )
            if rectified is None:
                continue
            rectified_full, rectified_mask, extra_info, source_u_grid, source_v_grid = rectified
            if args.rectified_inner_crop:
                cropped = (
                    crop_rectified_thin_territory_input(args, rectified_full, rectified_mask)
                    if is_thin_territory
                    else crop_rectified_chord_input(args, rectified_full, rectified_mask)
                )
                if cropped is None:
                    continue
                chord_input, mask_crop, crop_info = cropped
                extra_info.update(crop_info)
                crop_box = tuple(extra_info["inner_crop_box_y0_y1_x0_x1"])
                crop_y0, crop_y1, crop_x0, crop_x1 = crop_box
                provenance_mask = rectified_mask[crop_y0:crop_y1, crop_x0:crop_x1]
                provenance_u = source_u_grid[crop_y0:crop_y1, crop_x0:crop_x1][provenance_mask]
                provenance_v = source_v_grid[crop_y0:crop_y1, crop_x0:crop_x1][provenance_mask]
            else:
                chord_input, mask_crop = rectified_full, rectified_mask
                provenance_u = source_u_grid[rectified_mask]
                provenance_v = source_v_grid[rectified_mask]
            final_source_info = source_region_metrics(
                provenance_u,
                provenance_v,
                pose.width,
                pose.height,
            )
            if (
                min_final_unique > 0
                and final_source_info["final_source_unique_pixels"]
                < min_final_unique
            ):
                continue
            if (
                min_final_bbox_short > 0.0
                and final_source_info["final_source_bbox_short_side_px"]
                < min_final_bbox_short
            ):
                continue
            extra_info.update(final_source_info)
            extra_info["source_resolution_gate_policy"] = source_gate_policy
            extra_info["source_resolution_gate_min_unique_pixels"] = int(min_final_unique)
            extra_info["source_resolution_gate_min_bbox_short_side_px"] = float(min_final_bbox_short)
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
        source_info = save_source_region_provenance(
            image,
            provenance_u if args.chord_input_mode == "atlas_rectified" else item["u"],
            provenance_v if args.chord_input_mode == "atlas_rectified" else item["v"],
            out_dirs["source_region_overlays"] / f"{stem}_source_overlay.png",
            out_dirs["source_region_crops"] / f"{stem}_source_crop.png",
            out_dirs["source_region_footprints"] / f"{stem}_source_footprint.png",
            out_dirs["source_region_masked_crops"] / f"{stem}_source_masked_crop.png",
        )
        selected.append(
            {
                "stem": stem,
                "type": (
                    "view_contributor_rectified"
                    if (
                        args.strict_v3b_material_provenance
                        and args.chord_input_mode == "atlas_rectified"
                    )
                    else "view_contributor_rectified_inner"
                    if args.chord_input_mode == "atlas_rectified"
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
                **source_info,
                **extra_info,
                "chord_input": str(out_dirs["chord_inputs"] / f"{stem}.png"),
                "candidate_mask": str(out_dirs["candidate_masks"] / f"{stem}_mask.png"),
                "candidate_overlay": str(out_dirs["candidate_overlays"] / f"{stem}_overlay.png"),
                "mean_depth_residual": float(item["mean_depth_residual"]),
                "mean_surface_distance": float(item["mean_surface_distance"]),
            }
        )
    if selected:
        if args.resolution_aware_atlas_candidate and args.chord_input_mode == "atlas_rectified":
            best_density = max(
                float(item.get("rectified_source_sampling_density", 0.0))
                for item in selected
            )
            if best_density < args.min_rectified_source_density:
                append_atlas_candidate("low_rectified_source_density", best_density)
        return selected
    if not args.include_atlas_fallback:
        return selected
    append_atlas_candidate("no_valid_view_candidate", None)
    return selected


def prepare_stage(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = {
        "chord_inputs": args.out_dir / "chord_inputs",
        "candidate_masks": args.out_dir / "candidate_masks",
        "candidate_crops": args.out_dir / "candidate_crops",
        "candidate_overlays": args.out_dir / "candidate_overlays",
        "source_region_overlays": args.out_dir / "source_region_overlays",
        "source_region_crops": args.out_dir / "source_region_crops",
        "source_region_footprints": args.out_dir / "source_region_footprints",
        "source_region_masked_crops": args.out_dir / "source_region_masked_crops",
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
        edge_filter_audit = {"enabled": False, "rejected": []}
        wall_band_audit = {"enabled": False, "bands": []}
        if args.material_cluster_discovery:
            candidates, clusters = discover_weighted_material_regions(args, face, image, masks)
            clusters, wall_band_audit = discover_persistent_wall_bands(
                args,
                face,
                image,
                masks,
                clusters,
            )
            for band in wall_band_audit.get("bands", []):
                print(
                    f"[prepare-wall-band] {face}: rows="
                    f"{band['expanded_top']}:{band['expanded_bottom']} "
                    f"score={band['score']:.3f} "
                    f"coverage={band['coverage']:.3f}",
                    flush=True,
                )
            clusters, edge_filter_audit = filter_cross_face_edge_singletons(
                args,
                face,
                clusters,
                image.shape[:2],
            )
            candidates = [
                exemplar
                for cluster in clusters
                for exemplar in cluster.get("items", [cluster["representative"]])
            ]
            for rejected in edge_filter_audit["rejected"]:
                print(
                    f"[prepare-edge-filter] {face}: rejected cluster "
                    f"{rejected['cluster_index']} box={tuple(rejected['box_yx_size'])} "
                    f"width={rejected['width_fraction']:.3f} "
                    f"fraction={rejected['material_fraction']:.3f}",
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
            original_box = tuple(int(v) for v in cluster["representative"]["box"])
            search_minimum_size = int(args.rectified_search_box_min_size)
            search_scale = float(args.rectified_search_box_scale)
            if face == "floor":
                if args.floor_rectified_search_box_min_size is not None:
                    search_minimum_size = max(
                        search_minimum_size,
                        int(args.floor_rectified_search_box_min_size),
                    )
                if args.floor_rectified_search_box_scale is not None:
                    search_scale = max(
                        search_scale,
                        float(args.floor_rectified_search_box_scale),
                    )
            if face.startswith("wall_") and np.any(masks["lowres_wall_adapted"]):
                search_minimum_size = max(
                    search_minimum_size,
                    int(args.wall_lowres_rectified_search_box_min_size),
                )
                search_scale = max(
                    search_scale,
                    float(args.wall_lowres_rectified_search_box_scale),
                )
            if (
                args.thin_territory_source_adaptation
                and bool(cluster.get("territory_shape", {}).get("is_thin_territory", False))
            ):
                search_minimum_size = max(
                    search_minimum_size,
                    int(args.thin_territory_rectified_search_min_size),
                )
            box = expanded_square_box(
                original_box,
                image.shape[:2],
                search_scale,
                search_minimum_size,
            )
            support = representative_support(image.shape[:2], masks, cluster, box)
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
                cluster.get("territory_shape"),
                sim,
                manifest,
                metas,
                all_faces,
                da3_views,
                hf_alignment,
                caches,
                out_dirs,
                lowres_wall_source=bool(np.any(masks["lowres_wall_adapted"])),
            )
            face_regions.append(
                {
                    "region": int(region_i),
                    "material_id": int(material_i),
                    "exemplar_index": int(exemplar_i),
                    "material_box_purity": float(cluster["representative"].get("material_purity", 1.0)),
                    "box_yx_size": [int(v) for v in box],
                    "discovery_box_yx_size": [int(v) for v in original_box],
                    "cluster_score": float(cluster["score"]),
                    "cluster_items": int(len(cluster["items"])),
                    "mean_lab": [float(x) for x in cluster["mean_lab"]],
                    "edge_mean": float(cluster["edge_mean"]),
                    "sat_mean": float(cluster["sat_mean"]),
                    "material_fraction": float(cluster.get("material_fraction", 1.0)),
                    "territory_shape": cluster.get("territory_shape"),
                    "discovery_index": cluster.get("discovery_index"),
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
            if args.strict_v3b_material_provenance:
                errors = [
                    error
                    for candidate in view_candidates
                    for error in traceability_errors(candidate)
                ]
                if errors:
                    raise RuntimeError(
                        f"{face} region {region_i}: invalid strict CHORD candidate provenance:\n"
                        + "\n".join(errors)
                    )
        if not args.include_atlas_fallback:
            unsupported = [r for r in face_regions if not r.get("view_candidates")]
            if unsupported:
                dropped = [int(r["region"]) for r in unsupported]
                print(
                    f"[prepare-drop] {face}: no traceable source candidate for regions={dropped}",
                    flush=True,
                )
                face_regions = [r for r in face_regions if r.get("view_candidates")]
        if args.strict_v3b_material_provenance and not face_regions:
            raise RuntimeError(
                f"{face}: no face-local material survived strict Atlas-to-source-view Trace-Back"
            )

        # A discovered appearance class is usable only when at least one of
        # its regions has a traceable original-image patch.  Keep IDs dense so
        # downstream placement never creates an empty/fallback-only material.
        surviving_material_ids = sorted({int(r["material_id"]) for r in face_regions})
        material_id_remap = {
            material_id: dense_id
            for dense_id, material_id in enumerate(surviving_material_ids)
        }
        for region in face_regions:
            region["material_id"] = int(material_id_remap[int(region["material_id"])])
        stats.append(
            {
                "face": face,
                "shape_hw": [int(image.shape[0]), int(image.shape[1])],
                "candidate_count": int(len(candidates)),
                "material_count": int(len(surviving_material_ids)),
                "region_count": int(len(face_regions)),
                "persistent_wall_band_discovery": wall_band_audit,
                "cross_face_edge_singleton_filter": edge_filter_audit,
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
        "strict_v3b_material_provenance": bool(args.strict_v3b_material_provenance),
        "params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "stats": stats,
    }
    (args.out_dir / "metadata_view_contributor_chord_inputs.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_prepare_contact_sheet(args.out_dir, stats)
    write_source_trace_contact_sheets(args.out_dir, stats)
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
    if args.chord_output_dir is None:
        raise ValueError("--chord-output-dir is required for --stage compose")
    chord_output_dir = args.chord_output_dir
    keys = pbr_keys(args)
    meta_path = args.out_dir / "metadata_view_contributor_chord_inputs.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if args.strict_v3b_material_provenance:
        params = metadata.get("params", {})
        if metadata.get("strict_v3b_material_provenance") is not True:
            raise RuntimeError("strict compose requires strict prepared metadata")
        if params.get("strict_v3b_material_provenance") is not True:
            raise RuntimeError("strict compose requires strict=true in prepared params")
        if bool(params.get("include_atlas_fallback", True)):
            raise RuntimeError("strict compose requires include_atlas_fallback=false in prepared metadata")
        if bool(params.get("resolution_aware_atlas_candidate", False)):
            raise RuntimeError("strict compose forbids resolution-aware atlas candidates")
        if params.get("chord_input_mode") != "atlas_rectified":
            raise RuntimeError("strict compose requires atlas_rectified prepared inputs")
        prepared_faces = metadata.get("stats", [])
        if not prepared_faces:
            raise RuntimeError("strict compose received no prepared faces")
        errors: list[str] = []
        for face_info in prepared_faces:
            face = str(face_info.get("face", "<missing-face>"))
            regions = face_info.get("regions", [])
            if not regions:
                errors.append(f"{face}: no face-local prepared material regions")
            for region in regions:
                candidates = region.get("view_candidates", [])
                if not candidates:
                    errors.append(f"{face} region {region.get('region')}: no CHORD candidates")
                for candidate in candidates:
                    errors.extend(f"{face}: {error}" for error in traceability_errors(candidate))
                    if load_chord_map(chord_output_dir, candidate.get("stem", ""), args.basecolor_key) is None:
                        errors.append(
                            f"{face}: missing CHORD {args.basecolor_key} for {candidate.get('stem')}"
                        )
        if errors:
            raise RuntimeError("strict compose provenance preflight failed:\n" + "\n".join(errors))
    resolution_aware = bool(metadata.get("params", {}).get("resolution_aware_atlas_candidate", False))
    min_source_density = float(metadata.get("params", {}).get("min_rectified_source_density", 0.35))
    source_density_penalty = float(metadata.get("params", {}).get("rectified_source_density_penalty", 0.55))
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
                resolution_penalty = 0.0
                if resolution_aware and candidate.get("type") != "atlas_fallback":
                    density = float(candidate.get("rectified_source_sampling_density", 1.0))
                    resolution_penalty = source_density_penalty * max(
                        0.0,
                        min_source_density - density,
                    ) / max(min_source_density, 1e-6)
                    score += resolution_penalty
                if candidate.get("type") == "atlas_fallback" and not candidate.get("atlas_fallback_reason"):
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
                        "resolution_penalty": float(resolution_penalty),
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
                if args.strict_v3b_material_provenance:
                    raise RuntimeError(
                        f"{face} region {region_i}: no valid CHORD output; target fallback is forbidden"
                    )
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
                            "source_sampling_density": item["candidate"].get(
                                "rectified_source_sampling_density"
                            ),
                            "resolution_penalty": float(item.get("resolution_penalty", 0.0)),
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
            if args.strict_v3b_material_provenance:
                raise RuntimeError(
                    f"{face}: no face-local CHORD material prior; full-face Atlas fallback is forbidden"
                )
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
        "strict_v3b_material_provenance": bool(args.strict_v3b_material_provenance),
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


def write_source_trace_contact_sheets(out_dir: Path, stats: list[dict]) -> None:
    """Write explicit original-region -> crop -> rectified-input audit sheets."""
    for face_info in stats:
        rows: list[list[Image.Image]] = []
        face_name = str(face_info["face"])
        for region in face_info["regions"]:
            candidates = region.get("view_candidates", [])
            if not candidates:
                continue
            candidate = candidates[0]
            paths_and_labels = [
                (candidate.get("source_region_overlay"), "original RGB + exact footprint"),
                (candidate.get("source_region_crop"), "original RGB crop"),
                (candidate.get("source_region_footprint"), "exact sampled source pixels"),
                (candidate.get("chord_input"), "same pixels rectified for CHORD"),
            ]
            tiles = []
            prefix = f"{face_name} r{int(region['region']):02d}"
            for path_value, label in paths_and_labels:
                path = Path(path_value) if path_value else None
                if path is not None and path.exists():
                    tiles.append(text_tile(f"{prefix} | {label}", load_rgb(path)))
                else:
                    blank = np.ones((32, 32, 3), dtype=np.float32)
                    tiles.append(text_tile(f"{prefix} | missing {label}", blank))
            rows.append(tiles)
        if not rows:
            continue
        tile_w, tile_h = 230, 270
        sheet = Image.new("RGB", (4 * tile_w, len(rows) * tile_h), (235, 235, 235))
        for row_i, row in enumerate(rows):
            for col_i, tile in enumerate(row):
                sheet.paste(tile, (col_i * tile_w, row_i * tile_h))
        sheet.save(out_dir / f"source_to_rectified_trace_{face_name}.jpg", quality=92)


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
    if args.strict_v3b_material_provenance:
        args.include_atlas_fallback = False
        args.resolution_aware_atlas_candidate = False
        if args.chord_input_mode != "atlas_rectified":
            raise ValueError("strict material provenance requires --chord-input-mode atlas_rectified")
    if args.stage == "prepare":
        return prepare_stage(args)
    return compose_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
