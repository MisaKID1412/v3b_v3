#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, median_filter


LABEL_COLORS = np.asarray(
    [
        [217, 91, 91],
        [74, 151, 211],
        [236, 184, 69],
        [91, 178, 126],
        [164, 107, 194],
        [83, 190, 190],
        [190, 142, 83],
        [95, 95, 95],
    ],
    dtype=np.uint8,
)

TRACEABLE_CANDIDATE_TYPES = frozenset({"view_contributor_rectified"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose a CHORD-only base material atlas from the frozen reconstruction-to-CHORD "
            "pipeline. Real observations are used only as label/color evidence, never pasted "
            "into the output texture."
        )
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("config/freeze_manifest.json"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--faces", nargs="*", default=None)
    parser.add_argument(
        "--completed-observed-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing completed_observed/<face>.png and optionally "
            "weights/<face>.png. Defaults to completed_observed_lama_dir in the freeze manifest."
        ),
    )
    parser.add_argument("--observed-confidence", type=float, default=0.58)
    parser.add_argument("--observed-margin", type=float, default=0.12)
    parser.add_argument(
        "--placement-min-reliability",
        type=float,
        default=0.03,
        help=(
            "Minimum projection/completion reliability allowed to influence hard material "
            "placement and axis-curve fitting. Low-confidence completion behind object, "
            "door, or window masks remains fill context but cannot bend a material boundary."
        ),
    )
    parser.add_argument("--color-calibration-strength", type=float, default=0.88)
    parser.add_argument("--axis-boundary-min-accuracy", type=float, default=0.84)
    parser.add_argument("--axis-boundary-min-class-accuracy", type=float, default=0.68)
    parser.add_argument("--axis-layer-min-accuracy", type=float, default=0.76)
    parser.add_argument("--axis-layer-min-class-accuracy", type=float, default=0.55)
    parser.add_argument("--axis-layer-min-tangent-span", type=float, default=0.45)
    parser.add_argument("--axis-layer-min-strict-fraction", type=float, default=0.008)
    parser.add_argument("--axis-layer-min-width-frac", type=float, default=0.010)
    parser.add_argument(
        "--axis-layer-energy-tie-margin",
        type=float,
        default=0.0,
        help=(
            "Prefer an already validated ordered architectural-layer candidate when "
            "its total energy is within this absolute margin of the unconstrained "
            "winner. Zero preserves the original pure-energy selection."
        ),
    )
    parser.add_argument(
        "--axis-layer-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional ordered-layer candidate family. Disabled in the strict "
            "v3b-compatible route; arbitrary K remains supported by the original "
            "nearest-seed candidate and common energy comparison."
        ),
    )
    parser.add_argument("--linear-min-accuracy", type=float, default=0.91)
    parser.add_argument("--linear-min-class-accuracy", type=float, default=0.84)
    parser.add_argument("--linear-min-tangent-span", type=float, default=0.50)
    parser.add_argument("--linear-min-strict-fraction", type=float, default=0.055)
    parser.add_argument("--axis-curve-max-deviation-frac", type=float, default=0.075)
    parser.add_argument("--axis-curve-smooth-frac", type=float, default=0.030)
    parser.add_argument("--axis-curve-max-tangent-samples", type=int, default=640)
    parser.add_argument(
        "--axis-curve-tangent-edge-ignore-frac",
        type=float,
        default=0.0,
        help=(
            "Keep an axis-curve boundary anchored to its robust global threshold near "
            "the two tangential atlas edges, where adjacent-face seams and corner "
            "shadows are not reliable evidence of a material-boundary bend."
        ),
    )
    parser.add_argument("--data-energy-weight", type=float, default=1.0)
    parser.add_argument("--seed-mismatch-weight", type=float, default=5.0)
    parser.add_argument("--boundary-complexity-weight", type=float, default=0.025)
    parser.add_argument(
        "--use-discovered-material-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use full-face material masks produced by the material-discovery stage as "
            "strong seeds on strictly observed Atlas texels. Missing and rejected texels "
            "remain assigned by the original v3b territory inference."
        ),
    )
    parser.add_argument("--soft-material-blend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--soft-probability-sigma", type=float, default=5.5)
    parser.add_argument("--soft-probability-power", type=float, default=0.82)
    parser.add_argument("--soft-confidence-margin", type=float, default=0.55)
    parser.add_argument("--soft-boundary-radius-frac", type=float, default=0.080)
    parser.add_argument("--soft-region-blur-frac", type=float, default=0.020)
    parser.add_argument(
        "--soft-weight-source",
        choices=["probability", "target_reconstruction", "hybrid"],
        default="target_reconstruction",
    )
    parser.add_argument("--target-reconstruction-blur-sigma", type=float, default=5.5)
    parser.add_argument("--target-reconstruction-temperature", type=float, default=0.62)
    parser.add_argument("--target-reconstruction-label-prior", type=float, default=0.18)
    parser.add_argument("--target-reconstruction-smooth-sigma", type=float, default=1.8)
    parser.add_argument("--target-reconstruction-min-weight", type=float, default=0.22)
    parser.add_argument("--target-reconstruction-weight-power", type=float, default=1.35)
    parser.add_argument("--label-soft-mix-base", type=float, default=0.10)
    parser.add_argument("--label-soft-mix-lowconf", type=float, default=0.50)
    parser.add_argument("--soft-boundary-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--soft-boundary-width-frac", type=float, default=0.045)
    parser.add_argument("--pairwise-boundary-blend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pairwise-target-mix", type=float, default=0.55)
    parser.add_argument("--target-lowfreq-transfer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--strict-v3b-material-provenance",
        action="store_true",
        help=(
            "Require every selected material to come from a traceable original-view "
            "candidate on the same face; forbid cross-face and ceiling donors."
        ),
    )
    parser.add_argument("--target-lowfreq-sigma-frac", type=float, default=0.032)
    parser.add_argument("--target-lowfreq-strength", type=float, default=0.82)
    parser.add_argument(
        "--allow-neutral-ceiling-wall-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For a wall with effectively zero strict Atlas observation and no traceable "
            "wall donor, allow the dominant traceable neutral ceiling-paint material as "
            "an explicitly recorded last-resort donor. This never promotes rejected wall "
            "pixels or completed/inpainted wall content to material evidence."
        ),
    )
    parser.add_argument("--unobserved-wall-max-observed-fraction", type=float, default=0.001)
    parser.add_argument("--neutral-ceiling-min-lab-lightness", type=float, default=128.0)
    parser.add_argument("--neutral-ceiling-max-lab-chroma", type=float, default=24.0)
    parser.add_argument(
        "--merge-small-neutral-ceiling-shading-clusters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Merge a weak near-neutral ceiling cluster into the dominant near-neutral "
            "ceiling paint when the strict Atlas evidence indicates a luminance/shading "
            "split rather than an independently supported material."
        ),
    )
    parser.add_argument("--neutral-ceiling-merge-max-chroma", type=float, default=8.0)
    parser.add_argument("--neutral-ceiling-merge-max-lightness-gap", type=float, default=55.0)
    parser.add_argument("--neutral-ceiling-merge-max-secondary-seed-fraction", type=float, default=0.025)
    parser.add_argument("--neutral-ceiling-merge-max-secondary-to-primary-ratio", type=float, default=0.35)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if image.shape != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return image > 127


def load_gray(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if image.shape != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def input_region_map(input_metadata: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (face["face"], int(region["region"])): region
        for face in input_metadata["stats"]
        for region in face["regions"]
    }


def input_candidate_map(input_metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for face_info in input_metadata["stats"]:
        face = str(face_info["face"])
        for region in face_info["regions"]:
            region_index = int(region["region"])
            for candidate in region.get("view_candidates", []):
                annotated = dict(candidate)
                annotated["_face"] = face
                annotated["_region"] = region_index
                annotated["_material_id"] = int(region.get("material_id", region_index))
                if annotated["stem"] in result:
                    raise RuntimeError(f"duplicate candidate stem: {annotated['stem']}")
                result[annotated["stem"]] = annotated
    return result


def strict_layout_preflight(
    args: argparse.Namespace,
    faces: list[str],
    material_faces: dict[str, dict[str, Any]],
    input_regions: dict[tuple[str, int], dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    material_dir: Path,
    input_metadata: dict[str, Any],
    material_metadata: dict[str, Any],
    freeze: dict[str, Any],
    freeze_path: Path,
) -> None:
    errors: list[str] = []
    if input_metadata.get("strict_v3b_material_provenance") is not True:
        errors.append("prepared input metadata is not marked strict")
    input_params = input_metadata.get("params", {})
    if input_params.get("strict_v3b_material_provenance") is not True:
        errors.append("prepared input params are not marked strict")
    if bool(input_params.get("include_atlas_fallback", True)):
        errors.append("prepared inputs enabled atlas fallback")
    if bool(input_params.get("resolution_aware_atlas_candidate", False)):
        errors.append("prepared inputs enabled resolution-aware atlas fallback")
    if input_params.get("chord_input_mode") != "atlas_rectified":
        errors.append("prepared inputs are not atlas_rectified")
    if material_metadata.get("strict_v3b_material_provenance") is not True:
        errors.append("composed material metadata is not marked strict")
    material_params = material_metadata.get("params", {})
    if material_params.get("strict_v3b_material_provenance") is not True:
        errors.append("composed material params are not marked strict")
    run_root_value = freeze.get("experiment_root")
    if not run_root_value:
        errors.append("freeze manifest has no experiment_root")
    else:
        run_root = Path(run_root_value).resolve()
        for label, path in {
            "freeze manifest": freeze_path,
            "candidate metadata": Path(freeze["chord_inputs_metadata"]),
            "material metadata": Path(freeze["chord_materials_metadata"]),
            "material directory": material_dir,
            "strict projection": Path(freeze["strict_observed_projection_dir"]),
        }.items():
            try:
                path.resolve().relative_to(run_root)
            except (OSError, ValueError):
                errors.append(f"{label} is outside experiment_root: {path}")
        if Path(freeze["chord_inputs_metadata"]).parent.resolve() != material_dir.resolve():
            errors.append("candidate metadata and chord_material_dir are from different stages")
        if Path(freeze["chord_materials_metadata"]).parent.resolve() != material_dir.resolve():
            errors.append("material metadata and chord_material_dir are from different stages")
    if abs(float(args.color_calibration_strength)) > 1e-12:
        errors.append("color_calibration_strength must be 0 in strict mode")
    if bool(args.target_lowfreq_transfer):
        errors.append("target_lowfreq_transfer must be disabled in strict mode")
    for stem, candidate in candidates.items():
        if candidate.get("type") not in TRACEABLE_CANDIDATE_TYPES:
            errors.append(f"{stem}: forbidden candidate type={candidate.get('type')!r}")
        if candidate.get("input_mode") != "atlas_rectified":
            errors.append(f"{stem}: forbidden input_mode={candidate.get('input_mode')!r}")
        if not candidate.get("view_name") or candidate.get("image_id") is None:
            errors.append(f"{stem}: missing real source-view provenance")
        for key in ("chord_input", "candidate_mask"):
            path = candidate.get(key)
            if not path or not Path(path).is_file():
                errors.append(f"{stem}: missing {key}: {path}")
    for face in faces:
        face_info = material_faces.get(face)
        if face_info is None:
            errors.append(f"{face}: missing composed material metadata")
            continue
        regions = face_info.get("regions", [])
        if not regions:
            errors.append(f"{face}: no face-local composed material region")
            continue
        for material_record in regions:
            region = int(material_record["region"])
            input_region = input_regions.get((face, region))
            if input_region is None:
                errors.append(f"{face} region {region}: missing prepared input region")
                continue
            stem = material_record.get("chosen_stem")
            candidate = candidates.get(stem)
            if candidate is None:
                errors.append(f"{face} region {region}: unknown chosen stem {stem}")
                continue
            if candidate.get("_face") != face or int(candidate.get("_region", -1)) != region:
                errors.append(
                    f"{face} region {region}: chosen stem {stem} belongs to "
                    f"{candidate.get('_face')} region {candidate.get('_region')}"
                )
            input_material_id = int(input_region.get("material_id", region))
            if int(candidate.get("_material_id", -1)) != input_material_id:
                errors.append(
                    f"{face} region {region}: chosen stem {stem} has wrong material_id"
                )
            if material_record.get("chosen_type") != candidate.get("type"):
                errors.append(
                    f"{face} region {region}: chosen_type does not match prepared candidate"
                )
            chosen_tile = material_dir / "region_priors" / face / f"region_{region:02d}_chosen_tile.png"
            if not chosen_tile.is_file():
                errors.append(f"{face} region {region}: missing composed chosen tile {chosen_tile}")
    if errors:
        raise RuntimeError("strict material-layout provenance preflight failed:\n" + "\n".join(errors))


def apply_discovered_material_mask_constraints(
    labels: np.ndarray,
    known_labels: np.ndarray,
    probability_maps: np.ndarray,
    selected_materials: list[dict[str, Any]],
    input_metadata_path: Path,
    face: str,
    observed: np.ndarray,
    lab: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Constrain Atlas placement with discovered material territories.

    Broad material masks remain evidence-only on observed texels.  A territory
    explicitly classified during discovery as a coherent elongated Atlas region
    may also constrain its completed interval through occlusions; otherwise a
    nearest-seed fill turns a baseboard/border into triangular islands.
    """
    debug_dir = input_metadata_path.parent / "debug"
    input_metadata = json.loads(input_metadata_path.read_text(encoding="utf-8"))
    region_metadata = {
        (face_info["face"], int(region["region"])): region
        for face_info in input_metadata.get("stats", [])
        for region in face_info.get("regions", [])
    }
    material_masks: list[np.ndarray] = []
    mask_paths: list[str | None] = []
    mask_domains: list[str] = []
    for material in selected_materials:
        region = material.get("selected_region")
        path = debug_dir / f"{face}_region_{int(region):02d}_material_mask.png" if region is not None else None
        if path is None or not path.exists():
            material_masks.append(np.zeros(labels.shape, dtype=bool))
            mask_paths.append(None)
            mask_domains.append("missing")
            continue
        discovered_mask = load_mask(path, labels.shape)
        region_info = region_metadata.get((face, int(region)), {})
        is_thin_territory = bool(
            region_info.get("territory_shape", {}).get("is_thin_territory", False)
        )
        if is_thin_territory:
            material_masks.append(discovered_mask)
            mask_domains.append("completed_thin_atlas_territory")
        else:
            material_masks.append(discovered_mask & observed)
            mask_domains.append("observed_discovery_evidence")
        mask_paths.append(str(path))

    if not material_masks:
        return labels, known_labels, probability_maps, {
            "enabled": True,
            "applied": False,
            "reason": "no_selected_materials",
            "override_fraction": 0.0,
        }
    stack = np.stack(material_masks, axis=0)
    coverage = np.any(stack, axis=0)
    if not np.any(coverage):
        return labels, known_labels, probability_maps, {
            "enabled": True,
            "applied": False,
            "reason": "no_discovery_masks_found",
            "mask_paths": mask_paths,
            "override_fraction": 0.0,
        }

    scales = np.array([45.0, 8.0, 8.0], dtype=np.float32) if face.startswith("wall_") else np.array([32.0, 10.0, 10.0], dtype=np.float32)
    costs = np.sum(((lab[None, ...] - prototypes[:, None, None, :]) / scales[None, None, None, :]) ** 2, axis=3)
    costs[~stack] = np.inf
    winners = np.argmin(costs, axis=0).astype(labels.dtype)

    constrained_labels = labels.copy()
    constrained_known = known_labels.copy()
    constrained_labels[coverage] = winners[coverage]
    constrained_known[coverage] = winners[coverage]
    constrained_probability = probability_maps.copy()
    constrained_probability[:, coverage] = 0.0
    for index in range(len(material_masks)):
        constrained_probability[index][coverage & (winners == index)] = 1.0

    # A completed elongated Atlas territory is both a positive and a negative
    # spatial constraint: it owns its interval and must not reappear as
    # nearest-seed triangles elsewhere on an unobserved part of the face.
    exclusive_indices = [
        index
        for index, domain in enumerate(mask_domains)
        if domain == "completed_thin_atlas_territory"
    ]
    for index in exclusive_indices:
        constrained_probability[index][~material_masks[index]] = 0.0
    probability_sum = np.sum(constrained_probability, axis=0)
    valid_probability = probability_sum > 1e-8
    constrained_probability[:, valid_probability] /= probability_sum[valid_probability][None, :]
    constrained_labels[valid_probability] = np.argmax(
        constrained_probability[:, valid_probability],
        axis=0,
    ).astype(constrained_labels.dtype)

    overlap = np.sum(stack, axis=0) > 1
    return constrained_labels, constrained_known, constrained_probability, {
        "enabled": True,
        "applied": True,
        "method": "discovered_material_masks_with_completed_thin_atlas_territories",
        "mask_paths": mask_paths,
        "mask_domains": mask_domains,
        "exclusive_completed_territory_material_indices": [int(index) for index in exclusive_indices],
        "override_fraction": float(np.mean(coverage)),
        "observed_override_fraction": float(np.count_nonzero(coverage) / max(np.count_nonzero(observed), 1)),
        "overlap_fraction": float(np.mean(overlap)),
        "per_material_fraction": [float(np.mean(mask)) for mask in material_masks],
    }


def gray_image(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def gradient_magnitude(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy), gx, gy


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = first.reshape(-1).astype(np.float32)
    b = second.reshape(-1).astype(np.float32)
    a -= float(np.mean(a))
    b -= float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def histogram_cosine(first: np.ndarray, second: np.ndarray, bins: int = 18) -> float:
    a = first.reshape(-1)
    b = second.reshape(-1)
    low = min(float(np.percentile(a, 2.0)), float(np.percentile(b, 2.0)))
    high = max(float(np.percentile(a, 98.0)), float(np.percentile(b, 98.0)))
    if high <= low + 1e-6:
        return 0.0
    hist_a, _ = np.histogram(a, bins=bins, range=(low, high), density=True)
    hist_b, _ = np.histogram(b, bins=bins, range=(low, high), density=True)
    denom = float(np.linalg.norm(hist_a) * np.linalg.norm(hist_b))
    return float(np.dot(hist_a, hist_b) / denom) if denom > 1e-8 else 0.0


def radial_frequency_profile(gray: np.ndarray, bins: int = 24) -> np.ndarray:
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    fft = np.fft.fftshift(np.fft.fft2(small - float(np.mean(small))))
    power = np.log1p(np.abs(fft) ** 2)
    yy, xx = np.indices(power.shape)
    radius = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2)
    bucket = np.clip((radius / (float(np.max(radius)) + 1e-6) * bins).astype(np.int32), 0, bins - 1)
    profile = np.bincount(bucket.reshape(-1), weights=power.reshape(-1), minlength=bins).astype(np.float32)
    counts = np.bincount(bucket.reshape(-1), minlength=bins).astype(np.float32)
    profile = profile / np.maximum(counts, 1.0)
    profile = profile[1:]
    norm = float(np.linalg.norm(profile))
    return profile / norm if norm > 1e-8 else profile


def texture_fidelity(input_path: Path | None, output_path: Path) -> dict[str, float]:
    if input_path is None or not input_path.exists() or not output_path.exists():
        return {"texture_fidelity": 0.5, "gradient_ratio": 1.0, "gradient_correlation": 0.0}
    source = load_rgb(input_path)
    output = cv2.resize(load_rgb(output_path), (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)
    source_gray = gray_image(source)
    output_gray = gray_image(output)
    source_mag, _, _ = gradient_magnitude(source_gray)
    output_mag, _, _ = gradient_magnitude(output_gray)
    source_freq = radial_frequency_profile(source_gray)
    output_freq = radial_frequency_profile(output_gray)
    gradient_ratio = float(np.mean(output_mag) / max(float(np.mean(source_mag)), 1e-8))
    gradient_balance = max(0.0, 1.0 - abs(float(np.log(max(gradient_ratio, 1e-4)))))
    grad_corr = normalized_correlation(source_mag, output_mag)
    hist = histogram_cosine(source_mag, output_mag)
    freq = float(np.dot(source_freq, output_freq) / max(float(np.linalg.norm(source_freq) * np.linalg.norm(output_freq)), 1e-8))
    fidelity = 0.42 * grad_corr + 0.28 * hist + 0.20 * freq + 0.10 * gradient_balance
    return {
        "texture_fidelity": float(fidelity),
        "gradient_ratio": gradient_ratio,
        "gradient_correlation": grad_corr,
        "gradient_histogram_cosine": hist,
        "frequency_cosine": freq,
    }


def lab_image(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_likelihood(lab_values: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    delta = lab_values[:, None, :] - prototypes[None, :, :]
    distance2 = (
        (delta[..., 0] / 34.0) ** 2
        + (delta[..., 1] / 9.0) ** 2
        + (delta[..., 2] / 9.0) ** 2
    )
    return np.exp(-0.5 * distance2).astype(np.float32)


def normalized_probabilities(values: np.ndarray) -> np.ndarray:
    denom = np.sum(values, axis=0, keepdims=True)
    return values / np.maximum(denom, 1e-8)


def observed_prototype(lab: np.ndarray, mask: np.ndarray, fallback_tile: np.ndarray) -> np.ndarray:
    if np.count_nonzero(mask) >= 64:
        return np.median(lab[mask], axis=0).astype(np.float32)
    tile_lab = lab_image(fallback_tile)
    return np.median(tile_lab.reshape(-1, 3), axis=0).astype(np.float32)


def calibrate_tile_to_observed(tile: np.ndarray, prototype: np.ndarray, strength: float) -> np.ndarray:
    tile_lab = lab_image(tile)
    tile_center = np.median(tile_lab.reshape(-1, 3), axis=0).astype(np.float32)
    delta = (prototype - tile_center) * float(np.clip(strength, 0.0, 1.0))
    out = tile_lab + delta.reshape(1, 1, 3)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def support_mask(material_dir: Path, face: str, region: int, shape: tuple[int, int]) -> np.ndarray:
    path = material_dir / "debug" / f"{face}_region_{region:02d}_support.png"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    return load_mask(path, shape)


def choose_region_representative(
    material_dir: Path,
    face: str,
    material_record: dict[str, Any],
    input_region: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    region = int(material_record["region"])
    stem = material_record["chosen_stem"]
    candidate = candidates.get(stem)
    input_path = Path(candidate["chord_input"]) if candidate else None
    output_path = material_dir / "region_priors" / face / f"region_{region:02d}_chosen_tile.png"
    fidelity = texture_fidelity(input_path, output_path)
    side = float(input_region.get("box_yx_size", [0, 0, 512])[2])
    purity = float(input_region.get("material_box_purity", 1.0))
    chosen_score = float(material_record.get("chosen_score", 0.0))
    support_texels = float(material_record.get("support_texels", 0))
    score = chosen_score
    score += 2.6 * max(0.0, 0.90 - fidelity["texture_fidelity"])
    score += 0.9 * max(0.0, 0.78 - purity)
    score += 3.0 * max(0.0, 1.0 - side / 512.0)
    score += 0.6 * max(0.0, fidelity["gradient_ratio"] - 1.55)
    score += 0.9 * max(0.0, 0.42 - fidelity["gradient_ratio"]) / 0.42
    score -= 0.10 * math.log1p(max(support_texels, 0.0) / 50000.0)
    metrics = {
        **fidelity,
        "base_chord_score": chosen_score,
        "material_box_purity": purity,
        "crop_side": side,
        "support_texels": support_texels,
        "selection_score": float(score),
    }
    return float(score), metrics


def fill_from_nearest(labels: np.ndarray, known: np.ndarray) -> np.ndarray:
    if not np.any(known):
        return np.zeros(labels.shape, dtype=np.int16)
    if np.all(known):
        return labels.copy()
    _, nearest = distance_transform_edt(~known, return_indices=True)
    filled = labels.copy()
    missing = ~known
    filled[missing] = labels[nearest[0][missing], nearest[1][missing]]
    return filled


def boundary_complexity(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    denom = max(horizontal.size + vertical.size, 1)
    return float((np.count_nonzero(horizontal) + np.count_nonzero(vertical)) / denom)


def score_label_map(
    labels: np.ndarray,
    probabilities: np.ndarray,
    observed: np.ndarray,
    reliability: np.ndarray,
    known_labels: np.ndarray,
    known: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float]:
    material_count = probabilities.shape[0]
    label_indices = np.clip(labels, 0, material_count - 1)
    gathered = np.take_along_axis(probabilities, label_indices[None, ...], axis=0)[0]
    observed_weight = observed.astype(np.float32) * (0.15 + 0.85 * np.clip(reliability, 0.0, 1.0))
    if np.any(observed_weight > 0.0):
        data_energy = float(
            np.sum(-np.log(np.maximum(gathered, 1e-7)) * observed_weight)
            / max(float(np.sum(observed_weight)), 1e-8)
        )
    else:
        data_energy = 0.0
    seed_mismatch = float(np.mean(labels[known] != known_labels[known])) if np.any(known) else 0.0
    complexity = boundary_complexity(labels)
    total = (
        float(args.data_energy_weight) * data_energy
        + float(args.seed_mismatch_weight) * seed_mismatch
        + float(args.boundary_complexity_weight) * complexity
    )
    return {
        "total_energy": float(total),
        "data_energy": data_energy,
        "seed_mismatch": seed_mismatch,
        "boundary_complexity": complexity,
    }


def fit_axis_boundary(
    known_labels: np.ndarray,
    known: np.ndarray,
    material_count: int,
    axis: int,
    min_accuracy: float,
    min_class_accuracy: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if material_count != 2 or not np.any(known):
        return None, {"method": "not_applicable"}
    h, w = known.shape
    ys, xs = np.nonzero(known)
    values = known_labels[known]
    if np.unique(values).size != 2:
        return None, {"method": "nearest_known", "reason": "only_one_known_label"}
    if axis == 0:
        normal_coord = ys
        tangent_coord = xs
        normal_size = h
        tangent_size = w
        axis_name = "y"
    else:
        normal_coord = xs
        tangent_coord = ys
        normal_size = w
        tangent_size = h
        axis_name = "x"
    tangent_span_by_label = []
    median_normal = []
    for label in (0, 1):
        selected = values == label
        label_tangent = tangent_coord[selected]
        label_normal = normal_coord[selected]
        if label_tangent.size < 64:
            return None, {"method": "nearest_known", "reason": "small_label_sample"}
        tangent_span_by_label.append(
            float((np.percentile(label_tangent, 95.0) - np.percentile(label_tangent, 5.0)) / max(tangent_size - 1, 1))
        )
        median_normal.append(float(np.median(label_normal) / max(normal_size - 1, 1)))
    if min(tangent_span_by_label) < 0.18 or abs(median_normal[0] - median_normal[1]) < 0.18:
        return None, {
            "method": "nearest_known",
            "reason": "not_a_valid_axis_boundary",
            "axis": axis_name,
            "tangent_span_min": float(min(tangent_span_by_label)),
            "median_normal_delta": float(abs(median_normal[0] - median_normal[1])),
        }

    best: tuple[float, float, int, int, float] | None = None
    candidate_thresholds = np.unique(
        np.clip(np.linspace(0, normal_size - 1, 180).astype(np.int32), 0, normal_size - 1)
    )
    for threshold in candidate_thresholds:
        for positive_side_label in (0, 1):
            negative_side_label = 1 - positive_side_label
            pred = np.where(normal_coord >= threshold, positive_side_label, negative_side_label).astype(np.int16)
            accuracy = float(np.mean(pred == values))
            class_acc = []
            for label in (0, 1):
                label_values = values == label
                class_acc.append(float(np.mean(pred[label_values] == label)) if np.any(label_values) else 0.0)
            min_acc = float(min(class_acc))
            score = accuracy + 0.25 * min_acc
            if best is None or score > best[0]:
                best = (score, accuracy, positive_side_label, int(threshold), min_acc)
    if best is None:
        return None, {"method": "nearest_known"}
    _, accuracy, positive_side_label, threshold, min_acc = best
    if accuracy < min_accuracy or min_acc < min_class_accuracy:
        return None, {
            "method": "nearest_known",
            "reason": "axis_boundary_failed_validation",
            "axis": axis_name,
            "axis_boundary_accuracy": accuracy,
            "axis_boundary_min_class_accuracy": min_acc,
        }
    yy, xx = np.indices(known.shape)
    coord = yy if axis == 0 else xx
    labels = np.where(coord >= threshold, positive_side_label, 1 - positive_side_label).astype(np.int16)
    return labels, {
        "method": "axis_boundary",
        "axis": axis_name,
        "threshold": int(threshold),
        "positive_side_label": int(positive_side_label),
        "accuracy": accuracy,
        "min_class_accuracy": min_acc,
        "tangent_span_min": float(min(tangent_span_by_label)),
        "median_normal_delta": float(abs(median_normal[0] - median_normal[1])),
    }


def axis_curve_labels(
    probabilities: np.ndarray,
    observed: np.ndarray,
    reliability: np.ndarray,
    axis: int,
    positive_side_label: int,
    base_threshold: int,
    max_deviation_frac: float,
    smooth_frac: float,
    max_tangent_samples: int,
    tangent_edge_ignore_frac: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    material_count, h, w = probabilities.shape
    if material_count != 2:
        raise ValueError("axis curve boundary currently supports exactly two materials")
    if axis == 0:
        normal_size, tangent_size = h, w
        axis_name = "y"
    else:
        normal_size, tangent_size = w, h
        axis_name = "x"

    negative_side_label = 1 - int(positive_side_label)
    observed_weight = observed.astype(np.float32) * (0.15 + 0.85 * np.clip(reliability, 0.0, 1.0))
    cost_negative = -np.log(np.maximum(probabilities[negative_side_label], 1e-7)) * observed_weight
    cost_positive = -np.log(np.maximum(probabilities[positive_side_label], 1e-7)) * observed_weight
    if axis == 1:
        cost_negative = cost_negative.T
        cost_positive = cost_positive.T
        observed_weight_axis = observed_weight.T
    else:
        observed_weight_axis = observed_weight

    step = max(1, int(math.ceil(tangent_size / max(1, int(max_tangent_samples)))))
    centers = np.arange(0, tangent_size, step, dtype=np.int32)
    max_deviation = max(2, int(round(float(max_deviation_frac) * normal_size)))
    low = max(0, int(base_threshold) - max_deviation)
    high = min(normal_size - 1, int(base_threshold) + max_deviation)
    thresholds = []
    weights = []
    tangent_edge_ignore = max(
        0,
        int(round(float(tangent_edge_ignore_frac) * float(tangent_size))),
    )
    for start in centers:
        end = min(tangent_size, int(start) + step)
        if (
            tangent_edge_ignore > 0
            and (int(start) < tangent_edge_ignore or int(end) > tangent_size - tangent_edge_ignore)
        ):
            thresholds.append(float(base_threshold))
            weights.append(0.0)
            continue
        neg_line = np.sum(cost_negative[:, start:end], axis=1)
        pos_line = np.sum(cost_positive[:, start:end], axis=1)
        weight_line = np.sum(observed_weight_axis[:, start:end], axis=1)
        total_weight = float(np.sum(weight_line))
        if total_weight <= 1e-6:
            thresholds.append(float(base_threshold))
            weights.append(0.0)
            continue
        prefix_neg = np.concatenate([[0.0], np.cumsum(neg_line)])
        prefix_pos = np.concatenate([[0.0], np.cumsum(pos_line)])
        candidate = np.arange(low, high + 1, dtype=np.int32)
        costs = prefix_neg[candidate] + (prefix_pos[-1] - prefix_pos[candidate])
        anchor = ((candidate.astype(np.float32) - float(base_threshold)) / max(float(normal_size), 1.0)) ** 2
        costs = costs + 0.22 * total_weight * anchor
        thresholds.append(float(candidate[int(np.argmin(costs))]))
        weights.append(total_weight)

    thresholds_arr = np.asarray(thresholds, dtype=np.float32)
    if thresholds_arr.size >= 5:
        kernel = max(3, int(round(float(smooth_frac) * thresholds_arr.size)))
        if kernel % 2 == 0:
            kernel += 1
        if kernel > thresholds_arr.size:
            kernel = thresholds_arr.size if thresholds_arr.size % 2 == 1 else thresholds_arr.size - 1
        if kernel >= 3:
            thresholds_arr = median_filter(thresholds_arr, size=kernel, mode="nearest")
        sigma = max(0.75, float(smooth_frac) * thresholds_arr.size)
        thresholds_arr = cv2.GaussianBlur(thresholds_arr.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
    tangent_coords = np.arange(tangent_size, dtype=np.float32)
    center_coords = np.minimum(centers.astype(np.float32) + 0.5 * step, tangent_size - 1)
    curve = np.interp(tangent_coords, center_coords, thresholds_arr).astype(np.float32)
    curve = np.clip(curve, low, high)

    yy, xx = np.indices((h, w))
    if axis == 0:
        normal_coord = yy.astype(np.float32)
        tangent_coord = xx
    else:
        normal_coord = xx.astype(np.float32)
        tangent_coord = yy
    labels = np.where(normal_coord >= curve[tangent_coord], positive_side_label, negative_side_label).astype(np.int16)
    return labels, {
        "method": "axis_curve_boundary",
        "axis": axis_name,
        "base_threshold": int(base_threshold),
        "positive_side_label": int(positive_side_label),
        "curve_min": float(np.min(curve)),
        "curve_max": float(np.max(curve)),
        "curve_std": float(np.std(curve)),
        "curve_step": int(step),
        "curve_mean_bin_weight": float(np.mean(weights)) if weights else 0.0,
        "curve_tangent_edge_ignore_frac": float(tangent_edge_ignore_frac),
        "curve_tangent_edge_ignore_pixels": int(tangent_edge_ignore),
    }


def fit_axis_layers(
    known_labels: np.ndarray,
    known: np.ndarray,
    anchor_labels: np.ndarray,
    anchor_known: np.ndarray,
    probabilities: np.ndarray,
    observed: np.ndarray,
    reliability: np.ndarray,
    axis: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit an arbitrary number of ordered material layers along one atlas axis.

    The two-material boundary fitter cannot represent a narrow third material between
    two broad wall finishes. Here the order is inferred from seed medians and all
    boundaries are solved jointly with dynamic programming. The solution is retained
    only when every discovered material has sufficient observed support and the
    resulting layered map explains the observed labels.
    """
    material_count, h, w = probabilities.shape
    if material_count < 3 or not np.any(known):
        return None, {"method": "not_applicable"}
    normal_size = h if axis == 0 else w
    tangent_size = w if axis == 0 else h
    axis_name = "y" if axis == 0 else "x"
    normal_coord, tangent_coord = np.indices((h, w))
    if axis == 1:
        normal_coord, tangent_coord = tangent_coord, normal_coord

    medians = []
    tangent_spans = []
    strict_fractions = []
    for label in range(material_count):
        selected = known & (known_labels == label)
        count = int(np.count_nonzero(selected))
        if count < 64:
            return None, {
                "method": "nearest_known",
                "reason": "axis_layers_missing_label_support",
                "axis": axis_name,
                "label": int(label),
            }
        medians.append(float(np.median(normal_coord[selected])))
        tangent_values = tangent_coord[selected].astype(np.float32)
        tangent_spans.append(
            float((np.percentile(tangent_values, 95.0) - np.percentile(tangent_values, 5.0)) / max(tangent_size - 1, 1))
        )
        strict_fractions.append(float(np.mean(selected)))
    order = np.argsort(np.asarray(medians)).astype(np.int16)
    ordered_medians = np.asarray(medians, dtype=np.float32)[order]
    if np.any(np.diff(ordered_medians) < 1.0):
        return None, {"method": "nearest_known", "reason": "axis_layers_ambiguous_order", "axis": axis_name}
    if min(tangent_spans) < float(args.axis_layer_min_tangent_span):
        return None, {
            "method": "nearest_known",
            "reason": "axis_layers_short_tangent_support",
            "axis": axis_name,
            "min_tangent_span": float(min(tangent_spans)),
        }
    if min(strict_fractions) < float(args.axis_layer_min_strict_fraction):
        return None, {
            "method": "nearest_known",
            "reason": "axis_layers_small_strict_support",
            "axis": axis_name,
            "min_strict_fraction": float(min(strict_fractions)),
        }

    valid = observed & (reliability >= float(args.placement_min_reliability))
    weights = valid.astype(np.float32) * (0.15 + 0.85 * np.clip(reliability, 0.0, 1.0))
    negative_log = -np.log(np.maximum(probabilities, 1e-7))
    if axis == 0:
        denom = np.sum(weights, axis=1)
        row_cost = np.sum(negative_log * weights[None, ...], axis=2) / np.maximum(denom[None, :], 1e-6)
        seed_count = np.sum(known, axis=1).astype(np.float32)
        seed_label_count = np.stack([np.sum(known & (known_labels == label), axis=1) for label in range(material_count)])
    else:
        denom = np.sum(weights, axis=0)
        row_cost = np.sum(negative_log * weights[None, ...], axis=1) / np.maximum(denom[None, :], 1e-6)
        seed_count = np.sum(known, axis=0).astype(np.float32)
        seed_label_count = np.stack([np.sum(known & (known_labels == label), axis=0) for label in range(material_count)])
    observed_rows = denom > 1e-6
    if np.count_nonzero(observed_rows) < max(16, int(0.20 * normal_size)):
        return None, {"method": "nearest_known", "reason": "axis_layers_insufficient_rows", "axis": axis_name}
    row_ids = np.arange(normal_size)
    for label in range(material_count):
        row_cost[label] = np.interp(row_ids, row_ids[observed_rows], row_cost[label, observed_rows])
    seed_fraction = seed_label_count / np.maximum(seed_count[None, :], 1.0)
    row_cost += 1.25 * (1.0 - seed_fraction) * (seed_count[None, :] > 0.0)

    ordered_cost = row_cost[order]
    prefix = np.concatenate(
        [np.zeros((material_count, 1), dtype=np.float64), np.cumsum(ordered_cost, axis=1, dtype=np.float64)],
        axis=1,
    )
    min_width = max(1, int(round(float(args.axis_layer_min_width_frac) * normal_size)))
    if material_count * min_width > normal_size:
        return None, {"method": "nearest_known", "reason": "axis_layers_min_width_infeasible", "axis": axis_name}
    infinity = np.inf
    back = np.full((material_count, normal_size + 1), -1, dtype=np.int32)
    previous = np.full(normal_size + 1, infinity, dtype=np.float64)
    max_first = normal_size - (material_count - 1) * min_width
    for end in range(min_width, max_first + 1):
        previous[end] = prefix[0, end]
        back[0, end] = 0
    for layer_index in range(1, material_count):
        current = np.full(normal_size + 1, infinity, dtype=np.float64)
        min_end = (layer_index + 1) * min_width
        max_end = normal_size - (material_count - layer_index - 1) * min_width
        best_value = infinity
        best_start = -1
        for end in range(min_end, max_end + 1):
            start = end - min_width
            candidate = previous[start] - prefix[layer_index, start]
            if candidate < best_value:
                best_value = candidate
                best_start = start
            if best_start >= 0:
                current[end] = prefix[layer_index, end] + best_value
                back[layer_index, end] = best_start
        previous = current
    if not np.isfinite(previous[normal_size]):
        return None, {"method": "nearest_known", "reason": "axis_layers_dp_failed", "axis": axis_name}

    boundaries = [normal_size]
    end = normal_size
    for layer_index in range(material_count - 1, 0, -1):
        end = int(back[layer_index, end])
        if end < 0:
            return None, {"method": "nearest_known", "reason": "axis_layers_backtrack_failed", "axis": axis_name}
        boundaries.append(end)
    boundaries.append(0)
    boundaries = sorted(boundaries)
    normal_labels = np.zeros(normal_size, dtype=np.int16)
    for layer_index, label in enumerate(order):
        normal_labels[boundaries[layer_index] : boundaries[layer_index + 1]] = int(label)
    labels = normal_labels[normal_coord].astype(np.int16)

    predicted = labels[known]
    expected = known_labels[known]
    accuracy = float(np.mean(predicted == expected))
    anchor_predicted = labels[anchor_known]
    anchor_expected = anchor_labels[anchor_known]
    class_accuracy = [
        float(np.mean(anchor_predicted[anchor_expected == label] == label))
        if np.any(anchor_expected == label)
        else 0.0
        for label in range(material_count)
    ]
    if accuracy < float(args.axis_layer_min_accuracy) or min(class_accuracy) < float(args.axis_layer_min_class_accuracy):
        return None, {
            "method": "nearest_known",
            "reason": "axis_layers_failed_validation",
            "axis": axis_name,
            "axis_layer_accuracy": accuracy,
            "axis_layer_min_class_accuracy": float(min(class_accuracy)),
            "axis_layer_anchor_class_accuracy": [float(value) for value in class_accuracy],
            "axis_layer_boundaries": [int(value) for value in boundaries[1:-1]],
        }
    return labels, {
        "method": "axis_layers",
        "axis": axis_name,
        "material_order": [int(value) for value in order],
        "boundaries": [int(value) for value in boundaries[1:-1]],
        "accuracy": accuracy,
        "min_class_accuracy": float(min(class_accuracy)),
        "anchor_class_accuracy": [float(value) for value in class_accuracy],
        "tangent_span_min": float(min(tangent_spans)),
        "strict_fraction_min": float(min(strict_fractions)),
        "min_layer_width": int(min_width),
    }


def balanced_indices(values: np.ndarray, max_per_label: int) -> np.ndarray:
    selected = []
    for label in np.unique(values):
        indices = np.flatnonzero(values == label)
        if indices.size > max_per_label:
            step = indices.size / float(max_per_label)
            indices = indices[np.floor(np.arange(max_per_label) * step).astype(np.int64)]
        selected.append(indices)
    return np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)


def fit_linear_boundary(
    known_labels: np.ndarray,
    known: np.ndarray,
    material_count: int,
    min_accuracy: float,
    min_class_accuracy: float,
    min_tangent_span: float,
    min_strict_fraction: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if material_count != 2 or not np.any(known):
        return None, {"method": "not_applicable"}
    h, w = known.shape
    ys, xs = np.nonzero(known)
    values = known_labels[known]
    unique = np.unique(values)
    if unique.size != 2:
        return None, {"method": "nearest_known", "reason": "only_one_known_label"}
    sample = balanced_indices(values, 50000)
    if sample.size < 256:
        return None, {"method": "nearest_known", "reason": "small_sample"}
    x_norm = 2.0 * xs[sample].astype(np.float32) / max(float(w - 1), 1.0) - 1.0
    y_norm = 2.0 * ys[sample].astype(np.float32) / max(float(h - 1), 1.0) - 1.0
    design = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=1)
    target = np.where(values[sample] == unique[0], -1.0, 1.0).astype(np.float32)
    ridge = np.diag([1e-3, 1e-3, 1e-5]).astype(np.float32)
    try:
        weights = np.linalg.solve(design.T @ design + ridge, design.T @ target)
    except np.linalg.LinAlgError:
        return None, {"method": "nearest_known", "reason": "singular"}

    eval_sample = balanced_indices(values, 120000)
    eval_x = 2.0 * xs[eval_sample].astype(np.float32) / max(float(w - 1), 1.0) - 1.0
    eval_y = 2.0 * ys[eval_sample].astype(np.float32) / max(float(h - 1), 1.0) - 1.0
    eval_scores = weights[0] * eval_x + weights[1] * eval_y + weights[2]
    eval_pred = np.where(eval_scores <= 0.0, unique[0], unique[1]).astype(np.int16)
    eval_values = values[eval_sample]
    accuracy = float(np.mean(eval_pred == eval_values))
    class_acc = [
        float(np.mean(eval_pred[eval_values == label] == label)) if np.any(eval_values == label) else 0.0
        for label in unique
    ]
    normal = weights[:2].astype(np.float32)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        return None, {"method": "nearest_known", "reason": "zero_normal"}
    normal /= norm
    tangent = np.array([-normal[1], normal[0]], dtype=np.float32)
    tangent_projection = eval_x * tangent[0] + eval_y * tangent[1]
    tangent_spans = []
    strict_fractions = []
    for label in unique:
        selected = eval_values == label
        vals = tangent_projection[selected]
        tangent_spans.append(float(np.percentile(vals, 95.0) - np.percentile(vals, 5.0)))
        strict_fractions.append(float(np.mean(known & (known_labels == label))))
    if (
        accuracy < min_accuracy
        or min(class_acc) < min_class_accuracy
        or min(tangent_spans) < min_tangent_span
        or min(strict_fractions) < min_strict_fraction
    ):
        return None, {
            "method": "nearest_known",
            "reason": "linear_failed_validation",
            "linear_accuracy": accuracy,
            "linear_min_class_accuracy": float(min(class_acc)),
            "linear_min_tangent_span": float(min(tangent_spans)),
            "linear_min_strict_fraction": float(min(strict_fractions)),
        }
    yy, xx = np.indices(known.shape)
    all_x = 2.0 * xx.astype(np.float32) / max(float(w - 1), 1.0) - 1.0
    all_y = 2.0 * yy.astype(np.float32) / max(float(h - 1), 1.0) - 1.0
    scores = weights[0] * all_x + weights[1] * all_y + weights[2]
    labels = np.where(scores <= 0.0, unique[0], unique[1]).astype(np.int16)
    return labels, {
        "method": "linear_boundary",
        "linear_accuracy": accuracy,
        "linear_min_class_accuracy": float(min(class_acc)),
        "linear_min_tangent_span": float(min(tangent_spans)),
        "linear_min_strict_fraction": float(min(strict_fractions)),
        "linear_weight_x": float(weights[0]),
        "linear_weight_y": float(weights[1]),
        "linear_bias": float(weights[2]),
    }


def infer_labels(
    face: str,
    raw: np.ndarray,
    observed: np.ndarray,
    reliability: np.ndarray,
    seed_masks: list[np.ndarray],
    prototypes: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    h, w = observed.shape
    material_count = len(seed_masks)
    if material_count == 1:
        labels = np.zeros((h, w), dtype=np.int16)
        return labels, {"method": "single_material"}, labels.copy(), np.ones((1, h, w), dtype=np.float32)

    lab = lab_image(raw)
    values = lab.reshape(-1, 3)
    likelihood = lab_likelihood(values, prototypes).T.reshape(material_count, h, w)
    probs = normalized_probabilities(np.maximum(likelihood, 1e-8))
    sorted_probs = np.sort(probs, axis=0)
    best = np.argmax(probs, axis=0).astype(np.int16)
    confidence = sorted_probs[-1]
    margin = sorted_probs[-1] - sorted_probs[-2]
    observed_confident = (
        observed
        & (reliability >= float(args.placement_min_reliability))
        & (confidence >= args.observed_confidence)
        & (margin >= args.observed_margin)
    )

    known_labels = np.full((h, w), -1, dtype=np.int16)
    known = np.zeros((h, w), dtype=bool)
    for index, seed in enumerate(seed_masks):
        core = seed & observed
        if np.count_nonzero(core) >= 128:
            known_labels[core] = index
            known[core] = True

    if not np.any(known):
        known = observed_confident.copy()
        known_labels[known] = best[known]
    if not np.any(known):
        return np.zeros((h, w), dtype=np.int16), {"method": "fallback_empty"}, known_labels, probs

    fit_labels = known_labels.copy()
    fit_known = known.copy()
    fit_labels[observed_confident] = best[observed_confident]
    fit_known[observed_confident] = True

    candidate_solutions: list[tuple[np.ndarray, dict[str, Any]]] = []
    rejected_solutions: list[dict[str, Any]] = []
    nearest = fill_from_nearest(known_labels, known)
    candidate_solutions.append((nearest, {"method": "nearest_known"}))

    for axis in (0, 1):
        labels_axis, stats_axis = fit_axis_boundary(
            fit_labels,
            fit_known,
            material_count,
            axis,
            args.axis_boundary_min_accuracy,
            args.axis_boundary_min_class_accuracy,
        )
        if labels_axis is not None:
            candidate_solutions.append((labels_axis, stats_axis))
            labels_curve, stats_curve = axis_curve_labels(
                probs,
                observed & (reliability >= float(args.placement_min_reliability)),
                reliability,
                axis,
                int(stats_axis["positive_side_label"]),
                int(stats_axis["threshold"]),
                args.axis_curve_max_deviation_frac,
                args.axis_curve_smooth_frac,
                args.axis_curve_max_tangent_samples,
                args.axis_curve_tangent_edge_ignore_frac,
            )
            stats_curve["axis_boundary_accuracy"] = float(stats_axis["accuracy"])
            stats_curve["axis_boundary_min_class_accuracy"] = float(stats_axis["min_class_accuracy"])
            stats_curve["axis_boundary_tangent_span_min"] = float(stats_axis["tangent_span_min"])
            candidate_solutions.append((labels_curve, stats_curve))

    if args.axis_layer_candidates and material_count > 2:
        for axis in (0, 1):
            labels_layers, stats_layers = fit_axis_layers(
                fit_labels,
                fit_known,
                known_labels,
                known,
                probs,
                observed,
                reliability,
                axis,
                args,
            )
            if labels_layers is not None:
                candidate_solutions.append((labels_layers, stats_layers))
            else:
                rejected_solutions.append(stats_layers)

    labels_linear, stats_linear = fit_linear_boundary(
        fit_labels,
        fit_known,
        material_count,
        args.linear_min_accuracy,
        args.linear_min_class_accuracy,
        args.linear_min_tangent_span,
        args.linear_min_strict_fraction,
    )
    if labels_linear is not None:
        candidate_solutions.append((labels_linear, stats_linear))

    scored = []
    for candidate_labels, candidate_stats in candidate_solutions:
        energy = score_label_map(candidate_labels, probs, observed, reliability, known_labels, known, args)
        scored.append((energy["total_energy"], candidate_labels, {**candidate_stats, **energy}))
    scored.sort(key=lambda item: item[0])
    best_score, best_labels, best_stats = scored[0]
    axis_layer_margin = max(0.0, float(args.axis_layer_energy_tie_margin))
    if axis_layer_margin > 0.0 and best_stats.get("method") != "axis_layers":
        near_tied_axis_layers = [
            item
            for item in scored
            if item[2].get("method") == "axis_layers"
            and item[0] <= best_score + axis_layer_margin
        ]
        if near_tied_axis_layers:
            original_method = str(best_stats.get("method", "unknown"))
            original_score = float(best_score)
            best_score, best_labels, best_stats = near_tied_axis_layers[0]
            best_stats["architectural_axis_layer_tie_preference"] = True
            best_stats["unconstrained_best_method"] = original_method
            best_stats["unconstrained_best_total_energy"] = original_score
            best_stats["axis_layer_energy_tie_margin"] = axis_layer_margin
    best_stats["known_fraction"] = float(np.mean(known))
    best_stats["fit_known_fraction"] = float(np.mean(fit_known))
    best_stats["observed_confident_fraction"] = float(np.mean(observed_confident))
    best_stats["candidate_solutions"] = [
        {
            key: value
            for key, value in stats.items()
            if isinstance(value, (int, float, str))
        }
        for _, _, stats in scored
    ]
    best_stats["rejected_solutions"] = rejected_solutions
    return best_labels, best_stats, known_labels, probs


def tiled_material(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape[:2]
    yy = np.arange(h) % th
    xx = np.arange(w) % tw
    return tile[yy[:, None], xx[None, :]]


def label_image(labels: np.ndarray) -> np.ndarray:
    labels_safe = np.maximum(labels, 0)
    return LABEL_COLORS[labels_safe % len(LABEL_COLORS)].astype(np.float32) / 255.0


def target_reconstruction_weights(
    target: np.ndarray,
    target_weight: np.ndarray,
    tiled_materials: list[np.ndarray],
    labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    material_count = len(tiled_materials)
    h, w = target.shape[:2]
    if material_count == 1 or not args.soft_material_blend:
        weights = np.zeros((material_count, h, w), dtype=np.float32)
        weights[0] = 1.0
        return weights, {"method": "target_reconstruction_single_or_disabled"}

    sigma = max(0.0, float(args.target_reconstruction_blur_sigma))
    target_cmp = cv2.GaussianBlur(target, (0, 0), sigma) if sigma > 0.0 else target
    target_lab = lab_image(target_cmp).astype(np.float32)
    costs = []
    for material in tiled_materials:
        material_cmp = cv2.GaussianBlur(material, (0, 0), sigma) if sigma > 0.0 else material
        material_lab = lab_image(material_cmp).astype(np.float32)
        delta = target_lab - material_lab
        cost = (
            (delta[..., 0] / 34.0) ** 2
            + (delta[..., 1] / 9.0) ** 2
            + (delta[..., 2] / 9.0) ** 2
        )
        costs.append(cost.astype(np.float32))
    cost_stack = np.stack(costs, axis=0)
    likelihood = np.exp(-cost_stack / max(float(args.target_reconstruction_temperature), 1e-4)).astype(np.float32)

    prior_strength = float(np.clip(args.target_reconstruction_label_prior, 0.0, 1.0))
    prior = np.full_like(likelihood, 1.0 / material_count, dtype=np.float32)
    for index in range(material_count):
        prior[index, labels == index] = 1.0
    confidence = np.clip(target_weight.astype(np.float32), 0.0, 1.0) ** float(args.target_reconstruction_weight_power)
    confidence = np.clip(
        (confidence - float(args.target_reconstruction_min_weight))
        / max(1.0 - float(args.target_reconstruction_min_weight), 1e-6),
        0.0,
        1.0,
    )
    confidence = np.clip(
        (confidence - float(args.target_reconstruction_min_weight))
        / max(1.0 - float(args.target_reconstruction_min_weight), 1e-6),
        0.0,
        1.0,
    )
    effective_prior_strength = 1.0 - confidence * (1.0 - prior_strength)
    likelihood = likelihood * ((1.0 - effective_prior_strength[None, ...]) + effective_prior_strength[None, ...] * prior)

    smooth_sigma = max(0.0, float(args.target_reconstruction_smooth_sigma))
    if smooth_sigma > 0.0:
        likelihood = np.stack(
            [cv2.GaussianBlur(likelihood[index], (0, 0), smooth_sigma) for index in range(material_count)],
            axis=0,
        ).astype(np.float32)
    weights = normalized_probabilities(np.maximum(likelihood, 1e-8))
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-8)), axis=0)
    reconstruction = np.zeros_like(target)
    for index, material in enumerate(tiled_materials):
        reconstruction += weights[index, ..., None] * material
    diff = np.abs(lab_image(reconstruction) - lab_image(target))
    return weights.astype(np.float32), {
        "method": "completed_observed_target_reconstruction",
        "target_reconstruction_blur_sigma": float(args.target_reconstruction_blur_sigma),
        "target_reconstruction_temperature": float(args.target_reconstruction_temperature),
        "target_reconstruction_label_prior": float(args.target_reconstruction_label_prior),
        "target_reconstruction_smooth_sigma": float(args.target_reconstruction_smooth_sigma),
        "target_reconstruction_min_weight": float(args.target_reconstruction_min_weight),
        "target_reconstruction_weight_power": float(args.target_reconstruction_weight_power),
        "mean_target_confidence": float(np.mean(confidence)),
        "mean_entropy": float(np.mean(entropy)),
        "mean_max_weight": float(np.mean(np.max(weights, axis=0))),
        "mean_lab_abs_error": [float(value) for value in np.mean(diff.reshape(-1, 3), axis=0)],
    }


def probability_soft_material_weights(
    probability_maps: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    material_count, h, w = probability_maps.shape
    if material_count == 1 or not args.soft_material_blend:
        weights = np.zeros((material_count, h, w), dtype=np.float32)
        weights[0] = 1.0
        return weights, {"method": "hard_single_or_disabled"}

    probabilities = np.maximum(probability_maps.astype(np.float32), 1e-8)
    probabilities = probabilities ** float(args.soft_probability_power)
    weights = normalized_probabilities(probabilities)
    sigma = max(0.0, float(args.soft_probability_sigma))
    if sigma > 0.0:
        weights = np.stack(
            [cv2.GaussianBlur(weights[index], (0, 0), sigma) for index in range(material_count)],
            axis=0,
        ).astype(np.float32)
        weights = normalized_probabilities(np.maximum(weights, 1e-8))

    hard = np.zeros_like(weights)
    for index in range(material_count):
        hard[index] = labels == index
    margin = np.sort(weights, axis=0)[-1] - np.sort(weights, axis=0)[-2]
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]
    if np.any(boundary):
        distance = distance_transform_edt(~boundary).astype(np.float32)
    else:
        distance = np.full((h, w), max(h, w), dtype=np.float32)
    radius = max(1.0, float(args.soft_boundary_radius_frac) * min(h, w))
    boundary_softness = np.exp(-((distance / radius) ** 2)).astype(np.float32)
    confidence_hardness = np.clip((margin - float(args.soft_confidence_margin)) / 0.25, 0.0, 1.0)
    keep_soft = np.maximum(boundary_softness, 1.0 - confidence_hardness)
    region_sigma = max(0.0, float(args.soft_region_blur_frac) * min(h, w))
    if region_sigma > 0.0:
        keep_soft = cv2.GaussianBlur(keep_soft, (0, 0), region_sigma)
    keep_soft = np.clip(keep_soft, 0.0, 1.0)
    weights = keep_soft[None, ...] * weights + (1.0 - keep_soft[None, ...]) * hard
    weights = normalized_probabilities(np.maximum(weights, 1e-8))
    entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-8)), axis=0)
    return weights.astype(np.float32), {
        "method": "completed_observed_soft_probability_blend",
        "soft_probability_sigma": float(args.soft_probability_sigma),
        "soft_probability_power": float(args.soft_probability_power),
        "soft_confidence_margin": float(args.soft_confidence_margin),
        "soft_boundary_radius_frac": float(args.soft_boundary_radius_frac),
        "soft_region_blur_frac": float(args.soft_region_blur_frac),
        "mean_entropy": float(np.mean(entropy)),
        "mean_max_weight": float(np.mean(np.max(weights, axis=0))),
        "mean_keep_soft": float(np.mean(keep_soft)),
    }


def soft_material_weights(
    probability_maps: np.ndarray,
    labels: np.ndarray,
    target: np.ndarray,
    target_weight: np.ndarray,
    tiled_materials: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    probability_weights, probability_stats = probability_soft_material_weights(probability_maps, labels, args)
    if args.soft_weight_source == "probability":
        return probability_weights, probability_stats
    target_weights, target_stats = target_reconstruction_weights(target, target_weight, tiled_materials, labels, args)
    if args.soft_weight_source == "target_reconstruction":
        weights = target_weights
        stats = target_stats
    elif args.soft_weight_source == "hybrid":
        weights = normalized_probabilities(np.maximum(np.sqrt(probability_weights * target_weights), 1e-8))
        entropy = -np.sum(weights * np.log(np.maximum(weights, 1e-8)), axis=0)
        stats = {
            "method": "hybrid_probability_and_completed_target_reconstruction",
            "probability_stats": probability_stats,
            "target_reconstruction_stats": target_stats,
            "mean_entropy": float(np.mean(entropy)),
            "mean_max_weight": float(np.mean(np.max(weights, axis=0))),
        }
    else:
        weights = probability_weights
        stats = probability_stats

    material_count = weights.shape[0]
    hard = np.zeros_like(weights)
    for index in range(material_count):
        hard[index] = labels == index
    confidence = np.clip(target_weight.astype(np.float32), 0.0, 1.0) ** float(args.target_reconstruction_weight_power)
    confidence = np.clip(
        (confidence - float(args.target_reconstruction_min_weight))
        / max(1.0 - float(args.target_reconstruction_min_weight), 1e-6),
        0.0,
        1.0,
    )
    label_mix = float(args.label_soft_mix_base) + float(args.label_soft_mix_lowconf) * (1.0 - confidence)
    label_mix = np.clip(label_mix, 0.0, 1.0).astype(np.float32)
    soft_fraction = 1.0 - label_mix
    if args.soft_boundary_only:
        boundary = np.zeros(labels.shape, dtype=bool)
        boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
        boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
        boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]
        if np.any(boundary):
            distance = distance_transform_edt(~boundary).astype(np.float32)
            radius = max(1.0, float(args.soft_boundary_width_frac) * min(labels.shape))
            boundary_gate = np.exp(-((distance / radius) ** 2)).astype(np.float32)
        else:
            boundary_gate = np.zeros(labels.shape, dtype=np.float32)
        soft_fraction *= boundary_gate
    else:
        boundary_gate = np.ones(labels.shape, dtype=np.float32)
    if args.soft_boundary_only and args.pairwise_boundary_blend and material_count > 1:
        source_weights = weights
        pairwise_weights = hard.copy()
        distance_to_label = np.stack(
            [distance_transform_edt(labels != index).astype(np.float32) for index in range(material_count)],
            axis=0,
        )
        nearest = np.argsort(distance_to_label, axis=0)
        first = nearest[0]
        second = nearest[1]
        pairwise_target_mix = float(np.clip(args.pairwise_target_mix, 0.0, 1.0))
        pair_counts: dict[str, int] = {}
        active = boundary_gate > 1e-4
        for first_index in range(material_count):
            for second_index in range(first_index + 1, material_count):
                pair_mask = active & (
                    ((first == first_index) & (second == second_index))
                    | ((first == second_index) & (second == first_index))
                )
                if not np.any(pair_mask):
                    continue
                pair_counts[f"{first_index}-{second_index}"] = int(np.count_nonzero(pair_mask))
                distance_first = distance_to_label[first_index][pair_mask]
                distance_second = distance_to_label[second_index][pair_mask]
                distance_sum = distance_first + distance_second + 1e-6
                distance_weight_first = distance_second / distance_sum
                source_first = source_weights[first_index][pair_mask]
                source_second = source_weights[second_index][pair_mask]
                source_sum = source_first + source_second + 1e-6
                source_weight_first = source_first / source_sum
                mix = pairwise_target_mix * confidence[pair_mask]
                blended_first = mix * source_weight_first + (1.0 - mix) * distance_weight_first
                fraction = soft_fraction[pair_mask]
                hard_first = (labels[pair_mask] == first_index).astype(np.float32)
                hard_second = 1.0 - hard_first
                pairwise_weights[:, pair_mask] = 0.0
                pairwise_weights[first_index][pair_mask] = (
                    (1.0 - fraction) * hard_first + fraction * blended_first
                )
                pairwise_weights[second_index][pair_mask] = (
                    (1.0 - fraction) * hard_second + fraction * (1.0 - blended_first)
                )
        weights = pairwise_weights
    else:
        pair_counts = {}
        weights = (1.0 - soft_fraction[None, ...]) * hard + soft_fraction[None, ...] * weights
    weights = normalized_probabilities(np.maximum(weights, 1e-8))
    stats = {
        **stats,
        "label_soft_mix_base": float(args.label_soft_mix_base),
        "label_soft_mix_lowconf": float(args.label_soft_mix_lowconf),
        "soft_boundary_only": bool(args.soft_boundary_only),
        "soft_boundary_width_frac": float(args.soft_boundary_width_frac),
        "pairwise_boundary_blend": bool(args.pairwise_boundary_blend),
        "pairwise_target_mix": float(args.pairwise_target_mix),
        "pairwise_boundary_pair_counts": pair_counts,
        "mean_label_mix": float(np.mean(label_mix)),
        "mean_soft_fraction": float(np.mean(soft_fraction)),
        "mean_boundary_gate": float(np.mean(boundary_gate)),
        "mean_label_mix_observed_confident": float(np.mean(label_mix[confidence > 0.5])) if np.any(confidence > 0.5) else 0.0,
        "mean_label_mix_low_confidence": float(np.mean(label_mix[confidence <= 0.5])) if np.any(confidence <= 0.5) else 0.0,
    }
    return weights.astype(np.float32), stats


def soft_weight_image(weights: np.ndarray) -> np.ndarray:
    colors = LABEL_COLORS[np.arange(weights.shape[0]) % len(LABEL_COLORS)].astype(np.float32) / 255.0
    return np.einsum("khw,kc->hwc", weights, colors).astype(np.float32)


def transfer_low_frequency_to_target(
    atlas: np.ndarray,
    target: np.ndarray,
    target_weight: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not args.target_lowfreq_transfer:
        return atlas, {"method": "disabled"}
    sigma = max(1.0, float(args.target_lowfreq_sigma_frac) * min(atlas.shape[:2]))
    strength = float(np.clip(args.target_lowfreq_strength, 0.0, 1.0))
    atlas_lab = lab_image(atlas).astype(np.float32)
    target_lab = lab_image(target).astype(np.float32)
    atlas_low = np.stack(
        [cv2.GaussianBlur(atlas_lab[..., channel], (0, 0), sigma) for channel in range(3)],
        axis=-1,
    )
    target_low = np.stack(
        [cv2.GaussianBlur(target_lab[..., channel], (0, 0), sigma) for channel in range(3)],
        axis=-1,
    )
    residual = target_low - atlas_low
    confidence = np.clip(target_weight.astype(np.float32), 0.0, 1.0) ** float(args.target_reconstruction_weight_power)
    out_lab = atlas_lab + (strength * confidence)[..., None] * residual
    out = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    err_before = np.mean(np.abs(atlas_lab - target_lab).reshape(-1, 3), axis=0)
    err_after = np.mean(np.abs(lab_image(out) - target_lab).reshape(-1, 3), axis=0)
    return out, {
        "method": "low_frequency_lab_residual_to_completed_observed",
        "sigma": float(sigma),
        "strength": strength,
        "mean_target_confidence": float(np.mean(confidence)),
        "mean_lab_abs_error_before": [float(value) for value in err_before],
        "mean_lab_abs_error_after": [float(value) for value in err_after],
    }


def thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = image.copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    return result


def save_overview(out_dir: Path, faces: list[str], material_counts: dict[str, int]) -> None:
    columns = ["strict observed", "completed target", "hard labels", "soft weights", "CHORD soft base"]
    panel_w, panel_h, gap, header, row_h = 430, 250, 14, 76, 292
    canvas = Image.new(
        "RGB",
        (gap + len(columns) * (panel_w + gap), header + len(faces) * row_h),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 14), "material placement: completed-observed soft material weights", fill=(20, 20, 20), font=font)
    for col, title in enumerate(columns):
        draw.text((gap + col * (panel_w + gap), 44), title, fill=(20, 20, 20), font=font)
    for row, face in enumerate(faces):
        paths = [
            out_dir / "observed_reference" / f"{face}.png",
            out_dir / "completed_observed_reference" / f"{face}.png",
            out_dir / "labels" / f"{face}.png",
            out_dir / "soft_weights" / f"{face}.png",
            out_dir / "textures_base" / f"{face}.png",
        ]
        y = header + row * row_h
        draw.text((gap, y), f"{face} ({material_counts[face]} materials)", fill=(20, 20, 20), font=font)
        for col, path in enumerate(paths):
            img = thumbnail(Image.open(path).convert("RGB"), (panel_w, panel_h))
            x = gap + col * (panel_w + gap) + (panel_w - img.width) // 2
            image_y = y + 24 + (panel_h - img.height) // 2
            canvas.paste(img, (x, image_y))
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_dir / "material_placement_overview.jpg", quality=94)


def save_candidate_sheet(out_dir: Path, face_records: list[dict[str, Any]]) -> None:
    panel, gap, row_h, header = 170, 12, 220, 58
    max_mats = max(len(face["materials"]) for face in face_records)
    width = gap + (max_mats + 1) * (panel + gap)
    height = header + len(face_records) * row_h
    canvas = Image.new("RGB", (width, height), (246, 246, 246))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 16), "selected CHORD material tiles after observed-color calibration", fill=(20, 20, 20), font=font)
    for row, record in enumerate(face_records):
        y = header + row * row_h
        draw.text((gap, y + 4), record["face"], fill=(20, 20, 20), font=font)
        for col, material in enumerate(record["materials"]):
            path = Path(material["tile"])
            img = thumbnail(Image.open(path).convert("RGB"), (panel, panel))
            x = gap + (col + 1) * (panel + gap)
            canvas.paste(img, (x, y + 24))
            draw.text(
                (x, y + 4),
                f"m{material['material_id']} r{material['selected_region']}",
                fill=(20, 20, 20),
                font=font,
            )
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(preview_dir / "selected_materials.jpg", quality=94)


def select_unobserved_wall_donor(
    face_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Choose a traceable consensus wall material for a fully unobserved wall.

    This is intentionally restricted to wall faces.  Every donor must already
    have survived the normal Atlas discovery and original-view traceback.  A
    weighted Lab medoid favors the material that best agrees with the other
    observed walls, while observed seed support breaks otherwise similar ties.
    """
    donors: list[tuple[dict[str, Any], dict[str, Any], np.ndarray, float]] = []
    for record in face_records:
        if not str(record.get("face", "")).startswith("wall_"):
            continue
        observed_fraction = max(0.0, float(record.get("observed_fraction", 0.0)))
        for material in record.get("materials", []):
            prototype = material.get("observed_lab_prototype")
            if not isinstance(prototype, list) or len(prototype) != 3:
                continue
            seed_fraction = max(0.0, float(material.get("observed_seed_fraction", 0.0)))
            if observed_fraction <= 0.0 or seed_fraction <= 0.0:
                # Do not chain a previously synthesized/unobserved fallback
                # wall into another wall's provenance.
                continue
            support = max(1e-6, observed_fraction * max(seed_fraction, 1e-6))
            donors.append(
                (
                    record,
                    material,
                    np.asarray(prototype, dtype=np.float32),
                    support,
                )
            )
    if not donors:
        return None

    supports = np.asarray([item[3] for item in donors], dtype=np.float64)
    supports /= max(float(np.sum(supports)), 1e-12)
    costs = []
    for _, _, prototype, support in donors:
        distances = np.asarray(
            [float(np.linalg.norm(prototype - other[2])) for other in donors],
            dtype=np.float64,
        )
        consensus_cost = float(np.sum(supports * distances))
        support_bonus = 0.05 * float(np.log1p(1e6 * support))
        costs.append(consensus_cost - support_bonus)
    donor_index = int(np.argmin(np.asarray(costs, dtype=np.float64)))
    donor_record, donor_material, _, donor_support = donors[donor_index]
    diagnostics = {
        "method": "traceable_cross_wall_consensus_medoid",
        "donor_face": donor_record["face"],
        "donor_stem": donor_material["chosen_stem"],
        "donor_support": float(donor_support),
        "donor_consensus_cost": float(costs[donor_index]),
        "eligible_donor_count": len(donors),
    }
    return donor_material, diagnostics


def select_neutral_ceiling_wall_donor(
    face_records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Select dominant traceable neutral ceiling paint for an unobserved wall.

    This is deliberately a last-resort Structure3D transfer for rooms whose
    furnished views expose no usable wall texels.  The donor still comes from
    the normal strict Atlas -> original-view traceback -> CHORD route; no LaMa
    completion, rejected wall crop, or synthetic constant color is admitted.
    """
    eligible: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
    for record in face_records:
        if record.get("face") != "ceiling":
            continue
        observed_fraction = max(0.0, float(record.get("observed_fraction", 0.0)))
        for material in record.get("materials", []):
            prototype = material.get("observed_lab_prototype")
            if not isinstance(prototype, list) or len(prototype) != 3:
                continue
            lightness = float(prototype[0])
            chroma = float(math.hypot(float(prototype[1]) - 128.0, float(prototype[2]) - 128.0))
            seed_fraction = max(0.0, float(material.get("observed_seed_fraction", 0.0)))
            territory_fraction = max(0.0, float(material.get("territory_fraction", 0.0)))
            if seed_fraction <= 0.0 or observed_fraction <= 0.0:
                continue
            if lightness < float(args.neutral_ceiling_min_lab_lightness):
                continue
            if chroma > float(args.neutral_ceiling_max_lab_chroma):
                continue
            # Prefer the dominant ceiling territory; strict seed support and
            # lower chroma only break close ties.
            score = territory_fraction + 0.15 * seed_fraction - 0.001 * chroma
            eligible.append((score, chroma, record, material))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (-item[0], item[1]))
    score, chroma, donor_record, donor_material = eligible[0]
    diagnostics = {
        "method": "traceable_dominant_neutral_ceiling_paint_for_fully_unobserved_wall",
        "donor_face": donor_record["face"],
        "donor_stem": donor_material["chosen_stem"],
        "donor_material_id": int(donor_material["material_id"]),
        "donor_territory_fraction": float(donor_material.get("territory_fraction", 0.0)),
        "donor_seed_fraction": float(donor_material.get("observed_seed_fraction", 0.0)),
        "donor_lab_prototype": [float(value) for value in donor_material["observed_lab_prototype"]],
        "donor_lab_chroma": float(chroma),
        "donor_score": float(score),
        "eligible_donor_count": len(eligible),
    }
    return donor_material, diagnostics


def merge_small_neutral_ceiling_shading_clusters(
    face: str,
    selected_materials: list[dict[str, Any]],
    seed_masks: list[np.ndarray],
    raw_tiles: list[np.ndarray],
    prototypes: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, Any] | None]:
    """Collapse only a weak, neutral, luminance-only ceiling split.

    Material count remains data driven.  This guard fires only when every
    secondary cluster has very small strict Atlas support relative to the
    dominant cluster and all prototypes are near-neutral.  Colored, textured,
    or independently supported ceiling materials remain separate.
    """
    if (
        face != "ceiling"
        or not args.merge_small_neutral_ceiling_shading_clusters
        or len(selected_materials) <= 1
    ):
        return selected_materials, seed_masks, raw_tiles, prototypes, None
    chromas = [
        float(math.hypot(float(proto[1]) - 128.0, float(proto[2]) - 128.0))
        for proto in prototypes
    ]
    if max(chromas) > float(args.neutral_ceiling_merge_max_chroma):
        return selected_materials, seed_masks, raw_tiles, prototypes, None
    seed_fractions = [float(np.mean(mask)) for mask in seed_masks]
    dominant_index = int(np.argmax(np.asarray(seed_fractions, dtype=np.float64)))
    dominant_support = max(seed_fractions[dominant_index], 1e-12)
    dominant_lightness = float(prototypes[dominant_index][0])
    secondary_indices = [index for index in range(len(selected_materials)) if index != dominant_index]
    for index in secondary_indices:
        if seed_fractions[index] > float(args.neutral_ceiling_merge_max_secondary_seed_fraction):
            return selected_materials, seed_masks, raw_tiles, prototypes, None
        if seed_fractions[index] / dominant_support > float(args.neutral_ceiling_merge_max_secondary_to_primary_ratio):
            return selected_materials, seed_masks, raw_tiles, prototypes, None
        if abs(float(prototypes[index][0]) - dominant_lightness) > float(args.neutral_ceiling_merge_max_lightness_gap):
            return selected_materials, seed_masks, raw_tiles, prototypes, None
    merged_seed = np.zeros_like(seed_masks[dominant_index], dtype=bool)
    for mask in seed_masks:
        merged_seed |= mask
    dominant = dict(selected_materials[dominant_index])
    diagnostics = {
        "method": "merge_weak_neutral_ceiling_luminance_shading_clusters",
        "dominant_material_id": int(dominant["material_id"]),
        "dominant_stem": dominant["chosen_stem"],
        "dominant_seed_fraction": float(seed_fractions[dominant_index]),
        "merged_material_ids": [
            int(selected_materials[index]["material_id"]) for index in secondary_indices
        ],
        "merged_stems": [selected_materials[index]["chosen_stem"] for index in secondary_indices],
        "seed_fractions_before": [float(value) for value in seed_fractions],
        "lab_prototypes_before": [[float(value) for value in proto] for proto in prototypes],
        "lab_chromas_before": [float(value) for value in chromas],
    }
    dominant["material_index"] = 0
    dominant["observed_seed_fraction"] = float(np.mean(merged_seed))
    dominant["observed_seed_source"] = "strict_observed_atlas_material_masks_merged_shading_invariant"
    dominant["neutral_ceiling_shading_merge"] = diagnostics
    return (
        [dominant],
        [merged_seed],
        [raw_tiles[dominant_index]],
        [prototypes[dominant_index]],
        diagnostics,
    )


def main() -> int:
    args = parse_args()
    if args.strict_v3b_material_provenance and args.freeze_manifest == Path(
        "config/freeze_manifest.json"
    ):
        raise RuntimeError("strict mode requires an explicit current-run --freeze-manifest")
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    material_dir = Path(freeze["chord_material_dir"])
    observed_dir = Path(freeze["strict_observed_projection_dir"])
    completed_observed_root = args.completed_observed_dir
    if completed_observed_root is None and freeze.get("completed_observed_lama_dir"):
        completed_observed_root = Path(freeze["completed_observed_lama_dir"])
    completed_observed_texture_dir: Path | None = None
    completed_observed_weight_dir: Path | None = None
    if completed_observed_root is not None:
        completed_observed_root = Path(completed_observed_root)
        completed_observed_texture_dir = completed_observed_root / "completed_observed"
        if not completed_observed_texture_dir.exists():
            completed_observed_texture_dir = completed_observed_root
        completed_observed_weight_dir = completed_observed_root / "weights"
    material_metadata = json.loads(Path(freeze["chord_materials_metadata"]).read_text(encoding="utf-8"))
    input_metadata_path = Path(freeze["chord_inputs_metadata"])
    input_metadata = json.loads(input_metadata_path.read_text(encoding="utf-8"))
    input_regions = input_region_map(input_metadata)
    candidates = input_candidate_map(input_metadata)
    material_faces = {face["face"]: face for face in material_metadata["stats"]}
    faces = args.faces or [face["face"] for face in material_metadata["stats"]]
    face_order = {face: index for index, face in enumerate(faces)}
    # Fully unobserved walls inherit only from materials that survived the
    # normal Atlas discovery and original-view traceback.  Process traceable
    # faces first so that this existing fallback is independent of wall index
    # or polygon traversal order; restore the requested order before export.
    if args.strict_v3b_material_provenance:
        args.allow_neutral_ceiling_wall_fallback = False
        args.merge_small_neutral_ceiling_shading_clusters = False
        args.use_discovered_material_masks = False
        args.axis_layer_candidates = False
        args.axis_curve_tangent_edge_ignore_frac = 0.0
        strict_layout_preflight(
            args,
            faces,
            material_faces,
            input_regions,
            candidates,
            material_dir,
            input_metadata,
            material_metadata,
            freeze,
            args.freeze_manifest,
        )
        processing_faces = list(faces)
    else:
        processing_faces = sorted(
            faces,
            key=lambda face: (
                not bool(material_faces[face].get("regions")),
                face_order[face],
            ),
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    face_records: list[dict[str, Any]] = []
    material_counts: dict[str, int] = {}

    for face in processing_faces:
        face_info = material_faces[face]
        strict_raw = load_rgb(observed_dir / "debug" / f"{face}_raw_projected.png")
        shape = strict_raw.shape[:2]
        observed = load_mask(observed_dir / "debug" / f"{face}_final_keep_mask.png", shape)
        reliability_path = observed_dir / "debug" / f"{face}_reliability.png"
        reliability = load_rgb(reliability_path)[..., 0] if reliability_path.exists() else observed.astype(np.float32)
        placement_raw = strict_raw.copy()
        target_weight = observed.astype(np.float32)
        if completed_observed_texture_dir is not None:
            target_path = completed_observed_texture_dir / f"{face}.png"
            if target_path.exists():
                placement_raw = load_rgb(target_path)
                if placement_raw.shape[:2] != shape:
                    placement_raw = cv2.resize(placement_raw, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            if completed_observed_weight_dir is not None:
                weight_path = completed_observed_weight_dir / f"{face}.png"
                if weight_path.exists():
                    target_weight = load_gray(weight_path, shape)
        target_weight = np.clip(target_weight, 0.0, 1.0).astype(np.float32)
        placement_valid = target_weight > 1e-4
        placement_reliability = np.where(
            observed,
            np.maximum(reliability.astype(np.float32), target_weight),
            target_weight,
        ).astype(np.float32)

        observed_preview = strict_raw.copy()
        observed_preview[~observed] = 0.0
        save_rgb(args.out_dir / "observed_reference" / f"{face}.png", observed_preview)
        save_rgb(args.out_dir / "completed_observed_reference" / f"{face}.png", placement_raw)
        save_rgb(args.out_dir / "target_weights" / f"{face}.png", np.repeat(target_weight[..., None], 3, axis=2))

        grouped_records: dict[int, list[tuple[float, dict[str, Any], dict[str, Any], dict[str, float]]]] = {}
        for material_record in face_info["regions"]:
            region = int(material_record["region"])
            input_region = input_regions[(face, region)]
            material_id = int(input_region.get("material_id", region))
            score, metrics = choose_region_representative(material_dir, face, material_record, input_region, candidates)
            grouped_records.setdefault(material_id, []).append((score, material_record, input_region, metrics))

        lab = lab_image(placement_raw)
        selected_materials = []
        seed_masks = []
        raw_tiles = []
        prototypes = []
        for index, (material_id, group) in enumerate(sorted(grouped_records.items())):
            group.sort(key=lambda item: item[0])
            score, material_record, input_region, metrics = group[0]
            selected_region = int(material_record["region"])
            raw_tile_path = material_dir / "region_priors" / face / f"region_{selected_region:02d}_chosen_tile.png"
            raw_tile = load_rgb(raw_tile_path)
            seed = np.zeros(shape, dtype=bool)
            atlas_seed_paths = []
            for _, exemplar_record, _, _ in group:
                region_index = int(exemplar_record["region"])
                seed |= support_mask(material_dir, face, region_index, shape)
                if args.use_discovered_material_masks:
                    mask_path = material_dir / "debug" / f"{face}_region_{region_index:02d}_material_mask.png"
                    if mask_path.exists():
                        atlas_seed_paths.append(str(mask_path))
            seed_source = "representative_exemplar_support"
            if atlas_seed_paths:
                atlas_seed = np.zeros(shape, dtype=bool)
                for mask_path in atlas_seed_paths:
                    atlas_seed |= load_mask(Path(mask_path), shape)
                atlas_seed &= observed
                if np.count_nonzero(atlas_seed) >= 128:
                    seed = atlas_seed
                    seed_source = "strict_observed_atlas_material_mask"
            seed &= observed
            if args.strict_v3b_material_provenance and not np.any(seed):
                raise RuntimeError(
                    f"{face} material {material_id}: no strict observed seed support"
                )
            prototype = observed_prototype(lab, seed, raw_tile)
            if args.color_calibration_strength <= 0.0:
                calibrated = raw_tile.copy()
            else:
                calibrated = calibrate_tile_to_observed(raw_tile, prototype, args.color_calibration_strength)
            material_path = args.out_dir / "materials" / face / f"material_{index:02d}_id{material_id}.png"
            raw_material_path = args.out_dir / "materials_raw" / face / f"material_{index:02d}_id{material_id}.png"
            save_rgb(material_path, calibrated)
            save_rgb(raw_material_path, raw_tile)
            selected_materials.append(
                {
                    "material_index": index,
                    "material_id": material_id,
                    "selected_region": selected_region,
                    "chosen_stem": material_record["chosen_stem"],
                    "selection_score": score,
                    "selection_metrics": metrics,
                    "candidate_regions": [
                        {
                            "region": int(item[1]["region"]),
                            "chosen_stem": item[1]["chosen_stem"],
                            "selection_score": float(item[0]),
                            "metrics": item[3],
                        }
                        for item in group
                    ],
                    "observed_seed_fraction": float(np.mean(seed)),
                    "observed_seed_source": seed_source,
                    "observed_atlas_material_mask_paths": atlas_seed_paths,
                    "observed_lab_prototype": [float(v) for v in prototype],
                    "tile": str(material_path),
                    "raw_tile": str(raw_material_path),
                }
            )
            seed_masks.append(seed)
            raw_tiles.append(calibrated)
            prototypes.append(prototype)
        (
            selected_materials,
            seed_masks,
            raw_tiles,
            prototypes,
            neutral_ceiling_shading_merge,
        ) = merge_small_neutral_ceiling_shading_clusters(
            face,
            selected_materials,
            seed_masks,
            raw_tiles,
            prototypes,
            args,
        )
        if neutral_ceiling_shading_merge is not None:
            print(
                f"[material-placement-v1] {face}: merged weak neutral luminance cluster(s) "
                f"{neutral_ceiling_shading_merge['merged_material_ids']} into material "
                f"{neutral_ceiling_shading_merge['dominant_material_id']}",
                flush=True,
            )
        if not raw_tiles:
            if args.strict_v3b_material_provenance:
                raise RuntimeError(
                    f"{face}: no face-local traceable CHORD material; "
                    "cross-face and ceiling donors are forbidden"
                )
            donor_selection = select_unobserved_wall_donor(face_records) if face.startswith("wall_") else None
            if (
                donor_selection is None
                and face.startswith("wall_")
                and args.allow_neutral_ceiling_wall_fallback
                and float(np.mean(observed)) <= float(args.unobserved_wall_max_observed_fraction)
            ):
                donor_selection = select_neutral_ceiling_wall_donor(face_records, args)
            if donor_selection is None:
                raise RuntimeError(
                    f"{face}: no traceable CHORD material survived strict Atlas discovery; "
                    "no traceable observed-wall donor is available"
                )
            donor, donor_diagnostics = donor_selection
            raw_tile = load_rgb(Path(donor["raw_tile"]))
            material_path = args.out_dir / "materials" / face / "material_00_id0.png"
            raw_material_path = args.out_dir / "materials_raw" / face / "material_00_id0.png"
            save_rgb(material_path, raw_tile)
            save_rgb(raw_material_path, raw_tile)
            prototype = np.asarray(donor["observed_lab_prototype"], dtype=np.float32)
            selected_materials.append(
                {
                    "material_index": 0,
                    "material_id": 0,
                    "selected_region": int(donor["selected_region"]),
                    "chosen_stem": donor["chosen_stem"],
                    "selection_score": float(donor.get("selection_score", 0.0)),
                    "selection_metrics": {
                        **(donor.get("selection_metrics") or {}),
                        "unobserved_wall_fallback": donor_diagnostics,
                    },
                    "candidate_regions": [],
                    "observed_seed_fraction": 0.0,
                    "observed_seed_source": (
                        "fully_unobserved_neutral_ceiling_paint_donor"
                        if donor_diagnostics.get("donor_face") == "ceiling"
                        else "fully_unobserved_cross_wall_consensus"
                    ),
                    "observed_atlas_material_mask_paths": [],
                    "observed_lab_prototype": [float(v) for v in prototype],
                    "tile": str(material_path),
                    "raw_tile": str(raw_material_path),
                    "unobserved_wall_fallback": donor_diagnostics,
                }
            )
            seed_masks.append(np.zeros(shape, dtype=bool))
            raw_tiles.append(raw_tile)
            prototypes.append(prototype)
            print(
                f"[material-placement-v1] {face}: zero observed material; "
                f"inherited {donor_diagnostics['donor_stem']} from "
                f"{donor_diagnostics['donor_face']}",
                flush=True,
            )
        prototypes_array = np.asarray(prototypes, dtype=np.float32)
        labels, placement_stats, known_labels, probability_maps = infer_labels(
            face,
            placement_raw,
            placement_valid,
            placement_reliability,
            seed_masks,
            prototypes_array,
            args,
        )
        discovery_mask_stats = {
            "enabled": bool(args.use_discovered_material_masks),
            "applied": bool(
                args.use_discovered_material_masks
                and any(
                    material.get("observed_seed_source") == "strict_observed_atlas_material_mask"
                    for material in selected_materials
                )
            ),
            "method": "strict_observed_atlas_material_masks_as_territory_seeds",
            "per_material_seed_fraction": [
                float(material["observed_seed_fraction"])
                for material in selected_materials
            ],
            "override_fraction": 0.0,
        }

        tiled_materials = [tiled_material(tile, shape) for tile in raw_tiles]
        material_weights, soft_stats = soft_material_weights(
            probability_maps,
            labels,
            placement_raw,
            target_weight,
            tiled_materials,
            args,
        )
        atlas = np.zeros((*shape, 3), dtype=np.float32)
        for index, tiled in enumerate(tiled_materials):
            atlas += material_weights[index, ..., None] * tiled
            selected_materials[index]["territory_fraction"] = float(np.mean(labels == index))
            selected_materials[index]["soft_weight_fraction"] = float(np.mean(material_weights[index]))
        atlas_unmatched = atlas.copy()
        atlas, lowfreq_stats = transfer_low_frequency_to_target(atlas, placement_raw, target_weight, args)
        save_rgb(args.out_dir / "textures_base" / f"{face}.png", atlas)
        save_rgb(args.out_dir / "textures_base_before_lowfreq" / f"{face}.png", atlas_unmatched)
        save_rgb(args.out_dir / "labels" / f"{face}.png", label_image(labels))
        save_rgb(args.out_dir / "soft_weights" / f"{face}.png", soft_weight_image(material_weights))
        labels_npy = args.out_dir / "labels_npy"
        labels_npy.mkdir(parents=True, exist_ok=True)
        np.save(labels_npy / f"{face}.npy", labels.astype(np.int16))
        np.save(labels_npy / f"{face}_known_labels.npy", known_labels.astype(np.int16))
        np.save(labels_npy / f"{face}_soft_weights.npy", material_weights.astype(np.float32))

        evidence_dir = args.out_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for index, probability in enumerate(probability_maps):
            save_rgb(evidence_dir / f"{face}_material_{index:02d}_probability.png", np.repeat(probability[..., None], 3, axis=2))

        record = {
            "face": face,
            "provenance_scope": (
                "face_local_traceback_only"
                if args.strict_v3b_material_provenance
                else "legacy_with_recorded_fallbacks"
            ),
            "material_count": len(selected_materials),
            "materials": selected_materials,
            "placement_stats": placement_stats,
            "discovered_material_mask_constraints": discovery_mask_stats,
            "soft_blend_stats": soft_stats,
            "lowfreq_transfer_stats": lowfreq_stats,
            "observed_fraction": float(np.mean(observed)),
            "completed_observed_target_fraction": float(np.mean(placement_valid)),
            "completed_observed_mean_weight": float(np.mean(target_weight)),
            "known_fraction": float(np.mean(known_labels >= 0)),
            "neutral_ceiling_shading_merge": neutral_ceiling_shading_merge,
        }
        face_records.append(record)
        material_counts[face] = len(selected_materials)
        print(
            f"[material-placement-v1] {face}: materials={len(selected_materials)} "
            f"placement={placement_stats.get('method')} fractions="
            f"{[round(float(np.mean(labels == i)), 4) for i in range(len(selected_materials))]}",
            flush=True,
        )

    face_records.sort(key=lambda record: face_order[record["face"]])
    save_overview(args.out_dir, faces, material_counts)
    save_candidate_sheet(args.out_dir, face_records)
    metadata = {
        "method": "material_placement_v4_unified_curve_boundary",
        "freeze_manifest": str(args.freeze_manifest),
        "summary": (
            "CHORD-generated material tiles are assigned to frozen mesh face atlases. "
            "The completed_observed target is used as placement evidence; strict frozen real "
            "projections remain the seed/reference source. "
            "real high-frequency observations are not pasted into textures_base."
        ),
        "strict_observed_projection_dir": str(observed_dir),
        "completed_observed_root": str(completed_observed_root) if completed_observed_root is not None else None,
        "completed_observed_texture_dir": str(completed_observed_texture_dir) if completed_observed_texture_dir is not None else None,
        "completed_observed_weight_dir": str(completed_observed_weight_dir) if completed_observed_weight_dir is not None else None,
        "parameters": {
            "strict_v3b_material_provenance": bool(args.strict_v3b_material_provenance),
            "observed_confidence": args.observed_confidence,
            "observed_margin": args.observed_margin,
            "placement_min_reliability": args.placement_min_reliability,
            "color_calibration_strength": args.color_calibration_strength,
            "use_discovered_material_masks": bool(args.use_discovered_material_masks),
            "axis_boundary_min_accuracy": args.axis_boundary_min_accuracy,
            "axis_boundary_min_class_accuracy": args.axis_boundary_min_class_accuracy,
            "axis_layer_min_accuracy": args.axis_layer_min_accuracy,
            "axis_layer_min_class_accuracy": args.axis_layer_min_class_accuracy,
            "axis_layer_min_tangent_span": args.axis_layer_min_tangent_span,
            "axis_layer_min_strict_fraction": args.axis_layer_min_strict_fraction,
            "axis_layer_min_width_frac": args.axis_layer_min_width_frac,
            "axis_layer_candidates": bool(args.axis_layer_candidates),
            "linear_min_accuracy": args.linear_min_accuracy,
            "linear_min_class_accuracy": args.linear_min_class_accuracy,
            "linear_min_tangent_span": args.linear_min_tangent_span,
            "linear_min_strict_fraction": args.linear_min_strict_fraction,
            "axis_curve_max_deviation_frac": args.axis_curve_max_deviation_frac,
            "axis_curve_smooth_frac": args.axis_curve_smooth_frac,
            "axis_curve_max_tangent_samples": args.axis_curve_max_tangent_samples,
            "axis_curve_tangent_edge_ignore_frac": args.axis_curve_tangent_edge_ignore_frac,
            "data_energy_weight": args.data_energy_weight,
            "seed_mismatch_weight": args.seed_mismatch_weight,
            "boundary_complexity_weight": args.boundary_complexity_weight,
            "soft_material_blend": args.soft_material_blend,
            "soft_probability_sigma": args.soft_probability_sigma,
            "soft_probability_power": args.soft_probability_power,
            "soft_confidence_margin": args.soft_confidence_margin,
            "soft_boundary_radius_frac": args.soft_boundary_radius_frac,
            "soft_region_blur_frac": args.soft_region_blur_frac,
            "soft_weight_source": args.soft_weight_source,
            "target_reconstruction_blur_sigma": args.target_reconstruction_blur_sigma,
            "target_reconstruction_temperature": args.target_reconstruction_temperature,
            "target_reconstruction_label_prior": args.target_reconstruction_label_prior,
            "target_reconstruction_smooth_sigma": args.target_reconstruction_smooth_sigma,
            "target_reconstruction_min_weight": args.target_reconstruction_min_weight,
            "target_reconstruction_weight_power": args.target_reconstruction_weight_power,
            "label_soft_mix_base": args.label_soft_mix_base,
            "label_soft_mix_lowconf": args.label_soft_mix_lowconf,
            "soft_boundary_only": args.soft_boundary_only,
            "soft_boundary_width_frac": args.soft_boundary_width_frac,
            "target_lowfreq_transfer": args.target_lowfreq_transfer,
            "target_lowfreq_sigma_frac": args.target_lowfreq_sigma_frac,
            "target_lowfreq_strength": args.target_lowfreq_strength,
            "allow_neutral_ceiling_wall_fallback": bool(args.allow_neutral_ceiling_wall_fallback),
            "unobserved_wall_max_observed_fraction": args.unobserved_wall_max_observed_fraction,
            "neutral_ceiling_min_lab_lightness": args.neutral_ceiling_min_lab_lightness,
            "neutral_ceiling_max_lab_chroma": args.neutral_ceiling_max_lab_chroma,
            "merge_small_neutral_ceiling_shading_clusters": bool(args.merge_small_neutral_ceiling_shading_clusters),
            "neutral_ceiling_merge_max_chroma": args.neutral_ceiling_merge_max_chroma,
            "neutral_ceiling_merge_max_lightness_gap": args.neutral_ceiling_merge_max_lightness_gap,
            "neutral_ceiling_merge_max_secondary_seed_fraction": args.neutral_ceiling_merge_max_secondary_seed_fraction,
            "neutral_ceiling_merge_max_secondary_to_primary_ratio": args.neutral_ceiling_merge_max_secondary_to_primary_ratio,
        },
        "faces": face_records,
    }
    (args.out_dir / "metadata_material_placement.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
