#!/usr/bin/env python3
"""Regenerate CHORD inputs by running v3b trace-back after MatSeg grouping.

This is not a selector over old CHORD patches.  MatSeg supplies only material
identity.  For each resulting material group, this script merges the old strict
atlas supports, searches the merged atlas projection-weight field for the
highest-weight window, and then calls the original v3b geometric trace-back and
rectification functions to create a new CHORD input from the source image.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import generate_chord_view_contributor_region_priors as chord_trace  # noqa: E402


PATH_PARAMETER_KEYS = {
    "source_dir",
    "polygon_source_dir",
    "dataset_dir",
    "colmap_model_dir",
    "da3_dir",
    "object_mask_dir",
    "out_dir",
    "chord_output_dir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-metadata", type=Path, required=True)
    parser.add_argument("--region-assets-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--trace-log", type=Path, required=True)
    parser.add_argument("--faces", nargs="*", default=None)
    parser.add_argument(
        "--generator-script",
        type=Path,
        default=None,
        help="Optional newer compatible v3b generator implementation for regression testing.",
    )
    parser.add_argument(
        "--geometry-fallback-generator-script",
        type=Path,
        default=None,
        help=(
            "Optional compatible generator used only after the strict generator "
            "exhausts every weight-ordered original-view rectification attempt."
        ),
    )
    parser.add_argument(
        "--native-chord-input-size",
        type=int,
        default=None,
        help=(
            "Optional exact native rectified crop size. When set, the selected "
            "rectified crop and the saved CHORD input are both this size, so no "
            "spatial resize is performed."
        ),
    )
    parser.add_argument(
        "--skip-untraceable-materials",
        action="store_true",
        help=(
            "Drop a proposal group only after every descending-weight and "
            "geometry-adaptive original-view trace-back attempt fails."
        ),
    )
    parser.add_argument(
        "--strict-observed-support-only",
        action="store_true",
        help=(
            "Disable appearance-based expansion of thin territories during source "
            "trace-back. Geometry-derived window-size retries remain enabled."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
    if mask.shape != shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    return mask


def crop_exact_native_chord_input(
    params: argparse.Namespace,
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Select one exact native-size rectified crop without any resampling."""
    side = int(params.chord_input_size)
    height, width = valid_mask.shape
    if side > height or side > width:
        return None
    safe = valid_mask.astype(np.uint8)
    border = int(max(0, params.rectified_inner_safe_border_px))
    if border:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * border + 1, 2 * border + 1)
        )
        safe = cv2.erode(safe, kernel, iterations=1)
    valid_integral = cv2.integral(valid_mask.astype(np.float32))
    safe_integral = cv2.integral(safe.astype(np.float32))
    rgb8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    sharp = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)) / 255.0
    sharp_integral = cv2.integral(sharp.astype(np.float32))
    stride = max(1, int(round(side * float(params.rectified_inner_stride_frac))))
    y_positions = list(range(0, height - side + 1, stride))
    x_positions = list(range(0, width - side + 1, stride))
    if y_positions[-1] != height - side:
        y_positions.append(height - side)
    if x_positions[-1] != width - side:
        x_positions.append(width - side)

    def rect_sum(integral: np.ndarray, y: int, x: int) -> float:
        return float(
            integral[y + side, x + side]
            - integral[y, x + side]
            - integral[y + side, x]
            + integral[y, x]
        )

    area = float(side * side)
    best: tuple[float, int, int, float, float, float] | None = None
    for y0 in y_positions:
        for x0 in x_positions:
            valid_fraction = rect_sum(valid_integral, y0, x0) / area
            safe_fraction = rect_sum(safe_integral, y0, x0) / area
            if valid_fraction < float(params.rectified_inner_min_valid_frac):
                continue
            if safe_fraction < float(params.rectified_inner_min_valid_frac):
                continue
            sharpness = rect_sum(sharp_integral, y0, x0) / area
            centre_distance = float(
                np.hypot(
                    (y0 + 0.5 * side - 0.5 * height) / max(height, 1),
                    (x0 + 0.5 * side - 0.5 * width) / max(width, 1),
                )
            )
            score = 0.12 * np.log1p(30.0 * sharpness) - 0.45 * centre_distance
            row = (float(score), int(y0), int(x0), valid_fraction, safe_fraction, sharpness)
            if best is None or row[0] > best[0]:
                best = row
    if best is None:
        return None
    score, y0, x0, valid_fraction, safe_fraction, sharpness = best
    crop = np.clip(image[y0 : y0 + side, x0 : x0 + side].copy(), 0.0, 1.0)
    mask = valid_mask[y0 : y0 + side, x0 : x0 + side].astype(bool)
    return crop, mask, {
        "inner_crop_side": side,
        "inner_crop_valid_frac": float(valid_fraction),
        "inner_crop_safe_frac": float(safe_fraction),
        "inner_crop_sharpness": float(sharpness),
        "inner_crop_safe_border_px": border,
        "inner_crop_score": float(score),
        "inner_crop_box_y0_y1_x0_x1": [y0, y0 + side, x0, x0 + side],
        "inner_crop_output_valid_frac": float(np.mean(mask)),
        "native_chord_input_no_resize": True,
    }


def load_generator_module(path: Path, module_name: str):
    selected_scripts_dir = str(path.resolve().parent)
    if selected_scripts_dir in sys.path:
        sys.path.remove(selected_scripts_dir)
    sys.path.insert(0, selected_scripts_dir)
    for dependency in (
        "build_polygon_photo_source_from_colmap",
        "generate_material_priors",
        "generate_multi_material_priors",
        "polygon_projection_utils",
        "prepare_patch_preserving_tileable_inputs",
    ):
        sys.modules.pop(dependency, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def namespace_from_metadata(
    metadata: dict[str, Any], package_dir: Path, trace_module
) -> argparse.Namespace:
    original = copy.deepcopy(metadata["params"])
    # Build the namespace from the selected generator's own parser defaults,
    # then overlay the provenance parameters stored by the source run.  This
    # prevents a newer generalized generator from silently missing newly added
    # safety/adaptation fields when replaying older v3b metadata.
    required = {
        "source_dir": original["source_dir"],
        "polygon_source_dir": original["polygon_source_dir"],
        "dataset_dir": original["dataset_dir"],
        "colmap_model_dir": original["colmap_model_dir"],
        "da3_dir": original["da3_dir"],
        "object_mask_dir": original["object_mask_dir"],
        "out_dir": str(package_dir),
    }
    saved_argv = sys.argv
    try:
        sys.argv = [
            str(Path(trace_module.__file__)),
            "--stage",
            "prepare",
        ]
        for key, value in required.items():
            sys.argv.extend([f"--{key.replace('_', '-')}", str(value)])
        values = vars(trace_module.parse_args())
    finally:
        sys.argv = saved_argv
    values.update(original)
    values["out_dir"] = str(package_dir)
    values["stage"] = "prepare"
    for key in PATH_PARAMETER_KEYS:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    return argparse.Namespace(**values)


def start_positions(length: int, side: int, stride: int) -> list[int]:
    maximum = max(0, length - side)
    positions = list(range(0, maximum + 1, max(1, stride)))
    if not positions or positions[-1] != maximum:
        positions.append(maximum)
    return positions


def highest_weight_window(
    material_support: np.ndarray,
    projection_weight: np.ndarray,
    tile_size: int,
    tile_stride: int,
) -> tuple[tuple[int, int, int], np.ndarray, list[dict[str, float | int]]]:
    h, w = material_support.shape
    side = min(int(tile_size), h, w)
    weighted = np.where(material_support, projection_weight, 0.0).astype(np.float64)
    integral_weight = cv2.integral(weighted)
    integral_support = cv2.integral(material_support.astype(np.uint8))
    rows = []
    for y in start_positions(h, side, tile_stride):
        for x in start_positions(w, side, tile_stride):
            y1, x1 = y + side, x + side
            mass = (
                integral_weight[y1, x1]
                - integral_weight[y, x1]
                - integral_weight[y1, x]
                + integral_weight[y, x]
            )
            texels = int(
                integral_support[y1, x1]
                - integral_support[y, x1]
                - integral_support[y1, x]
                + integral_support[y, x]
            )
            if texels <= 0:
                continue
            values = weighted[y:y1, x:x1][material_support[y:y1, x:x1]]
            rows.append(
                {
                    "y": int(y),
                    "x": int(x),
                    "side": int(side),
                    "atlas_projection_weight_mass": float(mass),
                    "material_support_texels": texels,
                    "material_support_fraction": float(texels / max(side * side, 1)),
                    "atlas_projection_weight_mean": float(np.mean(values)),
                    "atlas_projection_weight_peak": float(np.max(values)),
                }
            )
    if not rows:
        raise RuntimeError("material support has no non-empty search window")
    rows.sort(
        key=lambda item: (
            float(item["atlas_projection_weight_mass"]),
            float(item["atlas_projection_weight_peak"]),
            int(item["material_support_texels"]),
        ),
        reverse=True,
    )
    # Projection mass is the primary rule.  On a broad surface, however, two
    # windows can be numerically tied because one contains more low-confidence
    # texels while the other contains fewer, much better observed texels.  A
    # hard total-mass sort then flips at tiny proposal-mask perturbations.  Only
    # inside a 1% mass plateau, and only for a >=20% gain in mean projection
    # confidence, promote the more reliable window.  This is a data-derived
    # tie break, not a room/face/material profile.
    mass_leader = rows[0]
    mass_max = float(mass_leader["atlas_projection_weight_mass"])
    near_ties = [
        row
        for row in rows
        if float(row["atlas_projection_weight_mass"]) >= 0.99 * mass_max
    ]
    confidence_leader = max(
        near_ties,
        key=lambda item: (
            float(item["atlas_projection_weight_mean"]),
            float(item["atlas_projection_weight_peak"]),
        ),
    )
    original_mean = float(mass_leader["atlas_projection_weight_mean"])
    promoted_mean = float(confidence_leader["atlas_projection_weight_mean"])
    if (
        confidence_leader is not mass_leader
        and promoted_mean >= 1.20 * max(original_mean, 1e-9)
    ):
        rows.remove(confidence_leader)
        rows.insert(0, confidence_leader)
        confidence_leader["near_tie_promotion"] = {
            "rule": (
                "within_1pct_of_max_projection_mass_and_at_least_20pct_"
                "higher_mean_projection_confidence"
            ),
            "mass_leader_y_x": [int(mass_leader["y"]), int(mass_leader["x"])],
            "mass_ratio_to_leader": float(
                float(confidence_leader["atlas_projection_weight_mass"])
                / max(mass_max, 1e-9)
            ),
            "mean_confidence_ratio_to_leader": float(
                promoted_mean / max(original_mean, 1e-9)
            ),
        }
    winner = rows[0]
    box = (int(winner["y"]), int(winner["x"]), int(winner["side"]))
    y, x, side = box
    selected_support = np.zeros_like(material_support)
    selected_support[y : y + side, x : x + side] = material_support[y : y + side, x : x + side]
    return box, selected_support, rows


def support_adaptive_retry_sides(
    material_support: np.ndarray,
    default_side: int,
) -> list[int]:
    """Return data-derived square sizes for a failed thin/compact territory.

    The ordinary v3b window is always attempted first.  Smaller windows are
    considered only after that window cannot produce a valid source-view
    rectification.  Their sizes come from connected support geometry, never
    from the face name or a room-specific setting.
    """
    binary = material_support.astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for index in range(1, count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area:
            components.append((area, min(width, height)))
    components.sort(reverse=True)
    sides = [min(int(default_side), *material_support.shape)]
    for _, short_side in components[:4]:
        side = max(24, min(int(short_side), *material_support.shape))
        if side < sides[0] and side not in sides:
            sides.append(side)
    return sides


def diverse_weight_windows(
    rows: list[dict[str, float | int]], max_attempts: int = 12
) -> list[dict[str, float | int]]:
    """Keep descending-weight windows while avoiding redundant overlap retries."""
    selected: list[dict[str, float | int]] = []
    for row in rows:
        y0, x0, side = int(row["y"]), int(row["x"]), int(row["side"])
        area = float(side * side)
        redundant = False
        for other in selected:
            oy, ox, oside = int(other["y"]), int(other["x"]), int(other["side"])
            overlap_h = max(0, min(y0 + side, oy + oside) - max(y0, oy))
            overlap_w = max(0, min(x0 + side, ox + oside) - max(x0, ox))
            intersection = float(overlap_h * overlap_w)
            union = area + float(oside * oside) - intersection
            if intersection / max(union, 1.0) >= 0.50:
                redundant = True
                break
        if not redundant:
            selected.append(row)
            if len(selected) >= max_attempts:
                break
    return selected


def material_groups(face_record: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for region in face_record["regions"]:
        groups.setdefault(int(region["material_id"]), []).append(region)
    return groups


def trace_and_rectify_highest_contributor(
    trace_module,
    params: argparse.Namespace,
    face: str,
    material_id: int,
    box: tuple[int, int, int],
    selected_support: np.ndarray,
    selected_territory: np.ndarray,
    poses: list[Any],
    sim: Any,
    manifest: dict[str, Any],
    metas: dict[str, Any],
    all_faces: list[str],
    da3_views: dict[int, Any],
    raw_to_room_matrix: np.ndarray | None,
    caches: dict[str, dict[Any, Any]],
    package_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contributors = trace_module.trace_region_contributors(
        params,
        face,
        material_id,
        selected_support,
        poses,
        sim,
        manifest,
        metas,
        all_faces,
        da3_views,
        raw_to_room_matrix,
        caches,
    )
    if not contributors:
        raise RuntimeError(f"{face} material {material_id}: trace-back found no contributors")

    y, x, side = box
    # The full material territory belongs to atlas assignment, not to source
    # patch validation.  Trace-back must rectify only the strict, actually
    # observed support inside the selected maximum-weight window.  Feeding the
    # full territory here makes unobserved texels invalidate otherwise clean
    # wall/band crops and can change the selected source view.
    target_mask = selected_support[y : y + side, x : x + side]
    attempts = []
    selected_candidate = None
    selected_pose = None
    selected_input = None
    selected_mask = None
    selected_rectified_full = None
    selected_rectified_mask = None
    selected_info: dict[str, Any] | None = None
    selected_rank = -1

    for rank, contributor in enumerate(contributors[: int(params.max_source_views_eval)]):
        pose = contributor["pose"]
        source_image = trace_module.load_rgb(pose.image_path)
        rectified = trace_module.rectified_tile_from_view(
            params,
            face,
            box,
            target_mask,
            pose,
            source_image,
            sim,
            manifest,
            metas,
            all_faces,
            da3_views,
            raw_to_room_matrix,
            caches,
        )
        if rectified is None:
            attempts.append(
                {
                    "rank_by_reverse_projection_weight": int(rank),
                    "view_name": pose.name,
                    "weight": float(contributor["weight"]),
                    "status": "failed_original_rectified_validity_gate",
                }
            )
            continue
        rectified_full, rectified_mask, info = rectified[:3]
        if params.rectified_inner_crop:
            if bool(getattr(params, "native_chord_input_no_resize", False)):
                cropped = crop_exact_native_chord_input(
                    params, rectified_full, rectified_mask
                )
            else:
                cropped = trace_module.crop_rectified_chord_input(
                    params, rectified_full, rectified_mask
                )
            if cropped is None:
                attempts.append(
                    {
                        "rank_by_reverse_projection_weight": int(rank),
                        "view_name": pose.name,
                        "weight": float(contributor["weight"]),
                        "status": (
                            "failed_native_exact_crop_gate"
                            if bool(getattr(params, "native_chord_input_no_resize", False))
                            else "failed_original_inner_crop_gate"
                        ),
                    }
                )
                continue
            chord_input, mask_crop, crop_info = cropped
            info.update(crop_info)
        else:
            chord_input, mask_crop = rectified_full, rectified_mask
        attempts.append(
            {
                "rank_by_reverse_projection_weight": int(rank),
                "view_name": pose.name,
                "weight": float(contributor["weight"]),
                "status": (
                    "selected_first_valid_after_weight_sort_native_exact_no_resize"
                    if bool(getattr(params, "native_chord_input_no_resize", False))
                    else "selected_first_valid_after_weight_sort"
                ),
            }
        )
        selected_candidate = contributor
        selected_pose = pose
        selected_input = chord_input
        selected_mask = mask_crop
        selected_rectified_full = rectified_full
        selected_rectified_mask = rectified_mask
        selected_info = info
        selected_rank = rank
        break

    if (
        selected_candidate is None
        or selected_pose is None
        or selected_info is None
        or selected_rectified_full is None
        or selected_rectified_mask is None
    ):
        raise RuntimeError(
            f"{face} material {material_id}: all high-weight contributors failed original rectification gates"
        )

    stem = f"{face}_m{material_id:02d}_material_trace_{Path(selected_pose.name).stem}"
    input_path = package_dir / "chord_inputs" / f"{stem}.png"
    mask_path = package_dir / "candidate_masks" / f"{stem}_mask.png"
    overlay_path = package_dir / "candidate_overlays" / f"{stem}_overlay.png"
    context_path = package_dir / "rectified_contexts" / f"{stem}_full.png"
    context_mask_path = (
        package_dir / "rectified_context_masks" / f"{stem}_full_mask.png"
    )
    save_rgb(input_path, selected_input)
    save_mask(mask_path, selected_mask)
    # Preserve the full, unresized rectification from the very same selected
    # contributor.  It is structure evidence only: CHORD still receives the
    # exact native crop saved at input_path.
    save_rgb(context_path, selected_rectified_full)
    save_mask(context_mask_path, selected_rectified_mask)
    overlay = selected_input.copy()
    overlay[selected_mask] = (
        0.58 * overlay[selected_mask]
        + 0.42 * np.asarray([1.0, 0.08, 0.04], dtype=np.float32)
    )
    save_rgb(overlay_path, overlay)

    candidate = {
        "stem": stem,
        "type": (
            "matseg_material_level_v3b_traceback_rectified_native_exact"
            if bool(getattr(params, "native_chord_input_no_resize", False))
            else "matseg_material_level_v3b_traceback_rectified"
        ),
        "view_name": selected_pose.name,
        "image_id": int(selected_pose.image_id),
        "input_mode": params.chord_input_mode,
        "source_rank_by_weight": int(selected_rank),
        "weight": float(selected_candidate["weight"]),
        "weight_frac": float(selected_candidate.get("weight_frac", 0.0)),
        "valid_sample_count": int(selected_candidate["count"]),
        "crop_box_y0_y1_x0_x1": selected_info.get("inner_crop_box_y0_y1_x0_x1"),
        "mask_pixels": int(np.count_nonzero(selected_mask)),
        "mask_fraction_in_chord_input": float(np.mean(selected_mask)),
        **selected_info,
        "chord_input": str(input_path),
        "candidate_mask": str(mask_path),
        "candidate_overlay": str(overlay_path),
        "rectified_structure_context": str(context_path),
        "rectified_structure_context_mask": str(context_mask_path),
        "rectified_structure_context_shape_hw": [
            int(selected_rectified_full.shape[0]),
            int(selected_rectified_full.shape[1]),
        ],
        "rectified_structure_context_resized": False,
        "mean_depth_residual": float(selected_candidate["mean_depth_residual"]),
        "mean_surface_distance": float(selected_candidate["mean_surface_distance"]),
    }
    trace_log = {
        "contributors": [
            {
                "rank_by_reverse_projection_weight": int(rank),
                "view_name": item["pose"].name,
                "image_id": int(item["pose"].image_id),
                "weight": float(item["weight"]),
                "weight_frac": float(item.get("weight_frac", 0.0)),
                "valid_sample_count": int(item["count"]),
            }
            for rank, item in enumerate(contributors)
        ],
        "rectification_attempts_in_weight_order": attempts,
        "selected_stem": stem,
        "selected_view_name": selected_pose.name,
        "selected_rank_by_reverse_projection_weight": int(selected_rank),
        "selected_reverse_projection_weight": float(selected_candidate["weight"]),
        "selected_input": str(input_path),
        "selected_input_sha256": sha256(input_path),
        "selected_structure_context": str(context_path),
        "selected_structure_context_sha256": sha256(context_path),
        "selected_structure_context_mask": str(context_mask_path),
        "selected_rectification": {
            "crop_box_y0_y1_x0_x1": selected_info.get("inner_crop_box_y0_y1_x0_x1"),
            "inner_crop_side": selected_info.get("inner_crop_side"),
            "native_chord_input_no_resize": bool(
                selected_info.get("native_chord_input_no_resize", False)
            ),
            "output_shape_hw": [int(selected_input.shape[0]), int(selected_input.shape[1])],
            "mask_fraction": float(np.mean(selected_mask)),
        },
    }
    return candidate, trace_log


def main() -> None:
    args = parse_args()
    identity = json.loads(args.identity_metadata.read_text(encoding="utf-8"))
    metadata_out = copy.deepcopy(identity)

    primary_trace = (
        load_generator_module(
            args.generator_script, "v3b_selected_chord_trace_generator"
        )
        if args.generator_script is not None
        else chord_trace
    )
    trace_stages = [("strict_original_v3b", primary_trace)]
    if args.geometry_fallback_generator_script is not None:
        fallback_trace = load_generator_module(
            args.geometry_fallback_generator_script,
            "v3b_geometry_fallback_chord_trace_generator",
        )
        if Path(fallback_trace.__file__).resolve() != Path(primary_trace.__file__).resolve():
            trace_stages.append(("geometry_fallback_after_strict_failure", fallback_trace))

    contexts = []
    for stage_name, trace_module in trace_stages:
        stage_params = namespace_from_metadata(identity, args.package_dir, trace_module)
        if args.strict_observed_support_only and hasattr(
            stage_params, "thin_territory_source_adaptation"
        ):
            stage_params.thin_territory_source_adaptation = False
        contexts.append(
            {
                "stage": stage_name,
                "module": trace_module,
                "params": stage_params,
            }
        )
    params = contexts[0]["params"]
    if args.strict_observed_support_only:
        metadata_out.setdefault("params", {})[
            "thin_territory_source_adaptation"
        ] = False
    if args.native_chord_input_size is not None:
        native_size = int(args.native_chord_input_size)
        if native_size < 64 or native_size % 8:
            raise ValueError("--native-chord-input-size must be >= 64 and divisible by 8")
        for context in contexts:
            stage_params = context["params"]
            stage_params.chord_input_size = native_size
            stage_params.rectified_inner_min_size = native_size
            stage_params.native_chord_input_no_resize = True
            # The native path uses an exact-size search and bypasses the
            # historical crop-then-resize helper entirely.
            stage_params.rectified_inner_max_side_frac = float(native_size) / float(
                stage_params.tile_size
            )
        metadata_out.setdefault("params", {})["chord_input_size"] = native_size
        metadata_out["params"]["rectified_inner_min_size"] = native_size
        metadata_out["params"]["rectified_inner_max_side_frac"] = float(
            params.rectified_inner_max_side_frac
        )
        metadata_out["params"]["native_chord_input_no_resize"] = True

    args.package_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "chord_inputs",
        "candidate_masks",
        "candidate_overlays",
        "rectified_contexts",
        "rectified_context_masks",
        "atlas_targets",
        "debug",
    ):
        (args.package_dir / name).mkdir(parents=True, exist_ok=True)

    for context in contexts:
        trace_module = context["module"]
        stage_params = context["params"]
        manifest, metas, _ = trace_module.manifest_and_faces(stage_params)
        poses, sim, raw_to_room_matrix = trace_module.proj.load_da3_hfalign_poses(
            stage_params.dataset_dir, stage_params.da3_dir
        )
        context.update(
            {
                "manifest": manifest,
                "metas": metas,
                "poses": poses,
                "sim": sim,
                "raw_to_room_matrix": raw_to_room_matrix,
                "da3_views": trace_module.proj.load_da3_views(
                    stage_params.da3_dir, poses
                ),
                "all_faces": trace_module.proj.face_names(manifest, None),
                "caches": {
                    "zbuffer": {},
                    "face_id": {},
                    "reject": {},
                    "depth_calib": {},
                },
            }
        )

    report_faces = []
    for face_record in metadata_out["stats"]:
        face = str(face_record["face"])
        if args.faces and face not in args.faces:
            continue
        primary_module = contexts[0]["module"]
        source_image = primary_module.load_rgb(
            primary_module.source_image_path(params.source_dir, face)
        )
        shape = source_image.shape[:2]
        face_contexts = []
        for context in contexts:
            trace_module = context["module"]
            stage_params = context["params"]
            stage_source_image = trace_module.load_rgb(
                trace_module.source_image_path(stage_params.source_dir, face)
            )
            if stage_source_image.shape[:2] != shape:
                raise RuntimeError(
                    f"{face}: trace generators disagree on atlas shape"
                )
            face_contexts.append(
                {
                    **context,
                    "source_image": stage_source_image,
                    "masks": trace_module.load_strict_masks(
                        stage_params, face, shape
                    ),
                }
            )
        support_by_region: dict[int, np.ndarray] = {}
        territory_by_region: dict[int, np.ndarray] = {}
        for region in face_record["regions"]:
            region_id = int(region["region"])
            support_path = (
                args.region_assets_dir
                / "debug"
                / f"{face}_region_{region_id:02d}_support.png"
            )
            support_by_region[region_id] = load_mask(support_path, shape)
            material_mask_path = (
                args.region_assets_dir
                / "debug"
                / f"{face}_region_{region_id:02d}_material_mask.png"
            )
            territory_by_region[region_id] = (
                load_mask(material_mask_path, shape)
                if material_mask_path.is_file()
                else support_by_region[region_id].copy()
            )
            shutil.copy2(
                support_path,
                args.package_dir / "debug" / support_path.name,
            )

        face_log = {"face": face, "materials": []}
        skipped_material_ids: set[int] = set()
        for material_id, regions in sorted(material_groups(face_record).items()):
            base_support = np.zeros(shape, dtype=bool)
            base_territory = np.zeros(shape, dtype=bool)
            for region in regions:
                base_support |= support_by_region[int(region["region"])]
                base_territory |= territory_by_region[int(region["region"])]

            generator_attempts = []
            selected_result = None
            selected_context = None
            last_error: RuntimeError | None = None
            window_attempts = []
            for face_context in face_contexts:
                trace_module = face_context["module"]
                stage_params = face_context["params"]
                stage_masks = face_context["masks"]
                merged_support = base_support & (
                    stage_masks["source"] | stage_masks["high"]
                )
                merged_territory = base_territory | merged_support
                if np.count_nonzero(merged_support) < 128:
                    last_error = RuntimeError(
                        f"{face} material {material_id}: merged strict support is too small"
                    )
                    generator_attempts.append(
                        {
                            "stage": face_context["stage"],
                            "status": "insufficient_strict_support",
                            "error": str(last_error),
                        }
                    )
                    continue

                projection_weight = stage_masks["weight_sum"]
                window_attempts = []
                stage_result = None
                for search_side in support_adaptive_retry_sides(
                    merged_support, int(stage_params.tile_size)
                ):
                    _, _, search_rows = highest_weight_window(
                        merged_support,
                        projection_weight,
                        int(search_side),
                        min(
                            int(stage_params.tile_stride),
                            max(1, int(search_side) // 4),
                        ),
                    )
                    for weight_rank, row in enumerate(
                        diverse_weight_windows(search_rows)
                    ):
                        box = (int(row["y"]), int(row["x"]), int(row["side"]))
                        y0, x0, side = box
                        selected_support = np.zeros_like(merged_support)
                        selected_support[
                            y0 : y0 + side, x0 : x0 + side
                        ] = merged_support[y0 : y0 + side, x0 : x0 + side]
                        selected_territory = np.zeros_like(merged_territory)
                        selected_territory[
                            y0 : y0 + side, x0 : x0 + side
                        ] = merged_territory[y0 : y0 + side, x0 : x0 + side]
                        try:
                            candidate, trace_log = trace_and_rectify_highest_contributor(
                                trace_module,
                                stage_params,
                                face,
                                material_id,
                                box,
                                selected_support,
                                selected_territory,
                                face_context["poses"],
                                face_context["sim"],
                                face_context["manifest"],
                                face_context["metas"],
                                face_context["all_faces"],
                                face_context["da3_views"],
                                face_context["raw_to_room_matrix"],
                                face_context["caches"],
                                args.package_dir,
                            )
                            window_attempts.append(
                                {
                                    "side": int(search_side),
                                    "rank_by_atlas_projection_weight": int(weight_rank),
                                    "selected_box_yx_size": [
                                        int(value) for value in box
                                    ],
                                    "status": (
                                        "selected_first_geometrically_valid_window_after_"
                                        "projection_mass_plateau_order"
                                    ),
                                }
                            )
                            stage_result = (
                                merged_support,
                                merged_territory,
                                box,
                                selected_support,
                                selected_territory,
                                search_rows,
                                row,
                                weight_rank,
                                candidate,
                                trace_log,
                            )
                            break
                        except RuntimeError as error:
                            last_error = error
                            window_attempts.append(
                                {
                                    "side": int(search_side),
                                    "rank_by_atlas_projection_weight": int(weight_rank),
                                    "selected_box_yx_size": [
                                        int(value) for value in box
                                    ],
                                    "status": "no_valid_original_view_rectification",
                                    "error": str(error),
                                }
                            )
                    if stage_result is not None:
                        break
                generator_attempts.append(
                    {
                        "stage": face_context["stage"],
                        "status": "selected" if stage_result is not None else "exhausted",
                        "window_attempts": window_attempts,
                    }
                )
                if stage_result is not None:
                    selected_result = stage_result
                    selected_context = face_context
                    break
            if selected_result is None:
                assert last_error is not None
                if args.skip_untraceable_materials:
                    skipped_material_ids.add(int(material_id))
                    face_log["materials"].append(
                        {
                            "material_id": int(material_id),
                            "member_regions": sorted(
                                int(region["region"]) for region in regions
                            ),
                            "status": "rejected_after_all_original_view_traceback_attempts_failed",
                            "generator_attempts": generator_attempts,
                            "error": str(last_error),
                        }
                    )
                    continue
                raise last_error
            (
                merged_support,
                merged_territory,
                box,
                selected_support,
                selected_territory,
                search_rows,
                selected_weight_row,
                selected_weight_rank,
                candidate,
                trace_log,
            ) = selected_result
            assert selected_context is not None

            merged_path = (
                args.package_dir / "debug" / f"{face}_material_{material_id:02d}_merged_support.png"
            )
            territory_path = (
                args.package_dir
                / "debug"
                / f"{face}_material_{material_id:02d}_merged_territory.png"
            )
            selected_path = (
                args.package_dir / "debug" / f"{face}_material_{material_id:02d}_selected_weight_window.png"
            )
            save_mask(merged_path, merged_support)
            save_mask(territory_path, merged_territory)
            save_mask(selected_path, selected_support)

            y, x, side = box
            selected_module = selected_context["module"]
            selected_params = selected_context["params"]
            target_tile, target_mask = selected_module.target_tile_from_material_cluster(
                face,
                selected_context["source_image"],
                selected_context["masks"],
                box,
                merged_territory,
            )
            target_path = (
                args.package_dir / "atlas_targets" / f"{face}_material_{material_id:02d}_target.png"
            )
            target_mask_path = (
                args.package_dir / "atlas_targets" / f"{face}_material_{material_id:02d}_target_mask.png"
            )
            save_rgb(target_path, target_tile)
            save_mask(target_mask_path, target_mask)

            for region in regions:
                region["view_candidates"] = [copy.deepcopy(candidate)]
                region["material_level_traceback"] = {
                    "material_id": int(material_id),
                    "selected_atlas_box_yx_size": [int(y), int(x), int(side)],
                    "generated_stem": candidate["stem"],
                    "trace_log": str(args.trace_log),
                }

            face_log["materials"].append(
                {
                    "material_id": int(material_id),
                    "member_regions": sorted(int(region["region"]) for region in regions),
                    "trace_generator_stage": selected_context["stage"],
                    "generator_attempts": generator_attempts,
                    "atlas_search": {
                        "criterion": (
                            "first_geometrically_traceable_window_by_projected_atlas_weight;_"
                            "near_equal_mass_windows_use_mean_projection_confidence_tiebreak"
                        ),
                        "tile_size": int(selected_params.tile_size),
                        "tile_stride": int(selected_params.tile_stride),
                        "selected_box_yx_size": [int(y), int(x), int(side)],
                        "selected_window_rank_by_projection_weight": int(
                            selected_weight_rank
                        ),
                        "selected_window": selected_weight_row,
                        "top_windows": search_rows[:20],
                        "window_attempts": window_attempts,
                        "merged_support_mask": str(merged_path),
                        "merged_material_territory_mask": str(territory_path),
                        "selected_support_mask": str(selected_path),
                        "lab_or_appearance_used": False,
                    },
                    "traceback": trace_log,
                }
            )
        if skipped_material_ids:
            kept_regions = [
                region
                for region in face_record["regions"]
                if int(region["material_id"]) not in skipped_material_ids
            ]
            if not kept_regions:
                # A wholly unobserved face has no honest source patch for
                # MatSeg or CHORD. Preserve the face itself, but expose zero
                # sourced materials so downstream continuous-field completion
                # cannot mistake an empty placeholder for a traced material.
                face_record["regions"] = []
                face_record["material_count"] = 0
                face_log["all_materials_untraceable"] = True
                face_log["completion_policy"] = (
                    "preserve_face_for_later_continuous_field_completion"
                )
                for material_log in face_log["materials"]:
                    if material_log.get("status", "").startswith("rejected_"):
                        material_log["status"] = "unobserved_face_passthrough"
            else:
                remap: dict[int, int] = {}
                for region in kept_regions:
                    old_material_id = int(region["material_id"])
                    remap.setdefault(old_material_id, len(remap))
                    region["material_id"] = remap[old_material_id]
                face_record["regions"] = kept_regions
                face_record["material_count"] = len(remap)
                face_log["skipped_untraceable_material_ids"] = sorted(
                    skipped_material_ids
                )
                face_log["material_id_remap_after_skip"] = {
                    str(old): int(new) for old, new in remap.items()
                }
        report_faces.append(face_log)

    native_exact = bool(getattr(params, "native_chord_input_no_resize", False))
    metadata_out["method"] = (
        "unified_matseg_identity_then_v3b_traceback_native_exact"
        if native_exact
        else "unified_matseg_identity_then_v3b_traceback"
    )
    metadata_out["material_level_traceback"] = {
        "trace_log": str(args.trace_log),
        "note": (
            "All listed CHORD inputs were regenerated after MatSeg grouping by the original "
            "v3b geometric contributor trace and rectification functions. The selected crop is "
            "an exact native-size rectified crop with no spatial resize. Old CHORD candidate "
            "stems were not selected or reused."
            if native_exact
            else "All listed CHORD inputs were regenerated after MatSeg grouping by the original "
            "v3b geometric contributor trace and rectification functions; old CHORD candidate "
            "stems were not selected or reused."
        ),
    }
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(
        json.dumps(metadata_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = {
        "status": (
            "material_level_v3b_traceback_native_exact_regenerated"
            if native_exact
            else "material_level_v3b_traceback_regenerated"
        ),
        "identity_metadata": str(args.identity_metadata),
        "output_metadata": str(args.output_metadata),
        "old_chord_candidates_reused": False,
        "old_chord_outputs_reused": False,
        "skip_untraceable_materials": bool(args.skip_untraceable_materials),
        "trace_generator_stages": [
            {
                "stage": context["stage"],
                "script": str(Path(context["module"].__file__).resolve()),
            }
            for context in contexts
        ],
        "algorithm": [
            "merge strict atlas supports by MatSeg material_id",
            (
                "search merged atlas weight_sum windows by descending projection mass; "
                "inside a 1% mass plateau only, promote a window only when its mean "
                "projection confidence is at least 20% higher"
            ),
            "call original v3b trace_region_contributors on that newly selected support",
            "try contributors in descending trace-back weight order",
            (
                "call original v3b rectified_tile_from_view, then select one exact native-size "
                "rectified crop without any spatial resize"
                if native_exact
                else "call original v3b rectified_tile_from_view and "
                "crop_rectified_chord_input"
            ),
            "write a newly generated CHORD input",
            "also preserve the unresized full rectification from that same selected contributor as structure-only evidence",
        ],
        "faces": report_faces,
    }
    args.trace_log.parent.mkdir(parents=True, exist_ok=True)
    args.trace_log.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
