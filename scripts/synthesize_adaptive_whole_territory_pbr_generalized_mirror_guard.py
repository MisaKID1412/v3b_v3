#!/usr/bin/env python3
"""Adaptive whole-territory PBR synthesis from the accepted v3b trace-back.

The upstream contract is deliberately immutable here:
  * material labels and chosen_stem come from the supplied v3b layout;
  * the trace-back candidate metadata supplies the physical crop footprint;
  * full 512x512 CHORD outputs are the only PBR source.

Only the extension from one traced material exemplar to a complete territory is
changed.  Routing uses exemplar statistics, never a face name or material name.

Directional single-axis materials are deliberately separated from two-axis
motifs.  Reflecting a wood-grain exemplar in both axes creates false bilateral
ornaments even when trace-back and CHORD are correct.  The directional route
therefore synthesizes one native-scale whole-territory spectral realization and
never reflects, tiles, quilts, or pastes the CHORD result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generate_scale_locked_whole_material_fields import (
    LABEL_COLORS,
    candidate_map,
    detail_lock_sigma,
    face_record_map,
    load_json,
    multiscale_continuous_noise,
    repetition_audit,
    resize_mask,
    scale_fidelity,
)
from generate_scale_locked_whole_material_pbr import (
    load_rgb,
    load_scalar,
    normal_audit,
    normal_rgb_to_vector,
    normal_vector_to_rgb,
    normalize_vectors,
    resize_exact,
    save_rgb,
    save_scalar,
    smooth_joint_field,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chord-input-metadata", type=Path, required=True)
    parser.add_argument("--pbr-chord-output-dir", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--faces", nargs="*", default=None)
    parser.add_argument("--atlas-resolution-scale", type=float, default=2.0)
    parser.add_argument(
        "--chord-output-support-mode",
        choices=("auto", "full_normalized", "safe_inner"),
        default="auto",
        help=(
            "auto respects the candidate construction contract: material-level atlas "
            "tracebacks use their safe inner support; normalized view-contributor crops "
            "use the complete 512 CHORD output"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--structured-periodic-min-corr", type=float, default=0.18)
    parser.add_argument("--structured-axis-ratio-max", type=float, default=0.55)
    parser.add_argument("--structured-visible-highpass-min", type=float, default=0.012)
    parser.add_argument("--structured-visible-edge-p95-min", type=float, default=0.050)
    parser.add_argument("--structured-lowdetail-total-std-min", type=float, default=0.100)
    parser.add_argument("--mirror-risk-macro-ratio-min", type=float, default=0.970)
    parser.add_argument("--mirror-risk-total-std-max", type=float, default=0.100)
    parser.add_argument("--mirror-risk-lowcontrast-std-max", type=float, default=0.075)
    parser.add_argument("--mirror-risk-edge-p95-min", type=float, default=0.040)
    parser.add_argument("--smooth-highpass-max", type=float, default=0.012)
    parser.add_argument("--smooth-edge-p95-max", type=float, default=0.040)
    parser.add_argument("--phase-warp-frac", type=float, default=0.035)
    parser.add_argument("--structured-color-modulation", type=float, default=0.018)
    parser.add_argument("--directional-coherence-min", type=float, default=0.62)
    parser.add_argument("--directional-spectral-second-moment-min", type=float, default=0.55)
    parser.add_argument("--detail-lock-min-sigma", type=float, default=12.0)
    parser.add_argument("--detail-lock-patch-frac", type=float, default=0.18)
    parser.add_argument("--smooth-lowfreq-max-std", type=float, default=0.022)
    parser.add_argument("--smooth-lowfreq-covariance-scale", type=float, default=0.65)
    parser.add_argument("--lowfreq-boundary-sigma-frac", type=float, default=0.002)
    parser.add_argument("--preview-thumb-width", type=int, default=340)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_and_scale_sources(
    source_dir: Path,
    candidate: dict[str, Any],
    atlas_scale: float,
    requested_support_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], float, int, str]:
    paths = {key: source_dir / f"{key}.png" for key in ("basecolor", "normal", "roughness", "metallic")}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    full_base = load_rgb(paths["basecolor"])
    full_normal = load_rgb(paths["normal"])
    full_rough = load_scalar(paths["roughness"])
    full_metal = load_scalar(paths["metallic"])
    if full_base.shape[:2] != (512, 512):
        raise ValueError(
            f"{source_dir}: expected the accepted full 512x512 CHORD output, got {full_base.shape[:2]}"
        )
    # There are two historical candidate contracts.  A material-level MatSeg
    # traceback stores a 512 atlas canvas and an explicitly safe inner support;
    # a legacy view-contributor candidate stores a normalized material crop in
    # the complete 512 canvas.  Treating either convention as the other is the
    # main reason v1 and Structure3D could not previously share one script.
    values = candidate.get("inner_crop_box_y0_y1_x0_x1") or candidate.get("crop_box_y0_y1_x0_x1")
    box = [int(round(float(value))) for value in values] if values else [0, 512, 0, 512]
    if requested_support_mode == "auto":
        candidate_type = str(candidate.get("type", ""))
        support_mode = "safe_inner" if "material_level" in candidate_type and values else "full_normalized"
    else:
        support_mode = requested_support_mode
    if support_mode == "safe_inner":
        y0, y1, x0, x1 = box
        y0, y1 = int(np.clip(y0, 0, 511)), int(np.clip(y1, y0 + 1, 512))
        x0, x1 = int(np.clip(x0, 0, 511)), int(np.clip(x1, x0 + 1, 512))
        box = [y0, y1, x0, x1]
        base = full_base[y0:y1, x0:x1]
        normal = full_normal[y0:y1, x0:x1]
        rough = full_rough[y0:y1, x0:x1]
        metal = full_metal[y0:y1, x0:x1]
    else:
        base = full_base
        normal = full_normal
        rough = full_rough
        metal = full_metal
    source_side = float(candidate.get("inner_crop_side") or min(full_base.shape[:2]))
    period = max(16, int(round(source_side * atlas_scale)))
    base = resize_exact(base, period)
    normal = normal_vector_to_rgb(normal_rgb_to_vector(resize_exact(normal, period)))
    rough = np.clip(resize_exact(rough, period), 0.0, 1.0)
    metal = np.clip(resize_exact(metal, period), 0.0, 1.0)
    if rough.ndim == 2:
        rough = rough[..., None]
    if metal.ndim == 2:
        metal = metal[..., None]
    return base, normal, rough, metal, box, source_side, period, support_mode


def triangular_coordinate(value: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Continuous mirror coordinate and tangent sign; it has no hard seam."""
    edge = max(1.0, float(length - 1))
    cycle = 2.0 * edge
    phase = np.mod(value, cycle)
    forward = phase <= edge
    coordinate = np.where(forward, phase, cycle - phase).astype(np.float32)
    sign = np.where(forward, 1.0, -1.0).astype(np.float32)
    return coordinate, sign


def irrational_phase_coordinates(
    shape: tuple[int, int],
    period: int,
    periodic_corr: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """One global coordinate field: scale-locked, continuous and non-lattice.

    The source period is never resized after this point.  Incommensurate waves
    perturb phase slowly, so equal local phases in neighbouring repetitions no
    longer sample an identical coordinate.  This is coordinate synthesis, not
    cutting/pasting image patches.
    """
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    corr_lock = float(np.clip((periodic_corr - 0.18) / 0.70, 0.0, 1.0))
    warp_frac = float(args.phase_warp_frac) * (1.0 - 0.45 * corr_lock)
    amplitude = max(1.0, warp_frac * float(period))
    phi = rng.uniform(0.0, 2.0 * math.pi, size=8)
    root2 = math.sqrt(2.0)
    root3 = math.sqrt(3.0)
    root5 = math.sqrt(5.0)
    dx = amplitude * (
        0.52 * np.sin(2.0 * math.pi * yy / (period * (2.5 + root2)) + phi[0])
        + 0.31 * np.sin(2.0 * math.pi * (xx + 0.37 * yy) / (period * (3.0 + root5)) + phi[1])
        + 0.17 * np.sin(2.0 * math.pi * yy / (period * (5.0 + root3)) + phi[2])
    )
    dy = amplitude * (
        0.52 * np.sin(2.0 * math.pi * xx / (period * (2.5 + root3)) + phi[3])
        + 0.31 * np.sin(2.0 * math.pi * (yy + 0.29 * xx) / (period * (3.0 + root2)) + phi[4])
        + 0.17 * np.sin(2.0 * math.pi * xx / (period * (5.0 + root5)) + phi[5])
    )
    map_x, sign_x = triangular_coordinate(xx + dx, period)
    map_y, sign_y = triangular_coordinate(yy + dy, period)
    del yy, xx, dx, dy
    return map_x, map_y, sign_x, sign_y, {
        "phase_warp_fraction_of_physical_period": warp_frac,
        "phase_warp_amplitude_px": amplitude,
    }


def structured_joint_field(
    base_patch: np.ndarray,
    normal_patch: np.ndarray,
    rough_patch: np.ndarray,
    metal_patch: np.ndarray,
    shape: tuple[int, int],
    periodic_corr: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    period = int(base_patch.shape[0])
    map_x, map_y, sign_x, sign_y, phase_stats = irrational_phase_coordinates(
        shape, period, periodic_corr, rng, args
    )
    base = cv2.remap(base_patch, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    normal_rgb = cv2.remap(normal_patch, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    rough = cv2.remap(rough_patch, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    metal = cv2.remap(metal_patch, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

    # A single face-wide *material-axis* modulation prevents identical-looking
    # motif instances.  Independent RGB noise would invent coloured clouds, so
    # variation is constrained to the dominant covariance axis of the source.
    corr_lock = float(np.clip((periodic_corr - 0.18) / 0.70, 0.0, 1.0))
    modulation_strength = float(args.structured_color_modulation) * (1.0 - 0.50 * corr_lock)
    # OpenCV drops the last axis for a singleton channel during resize, so ask
    # the shared helper for three channels and retain one scalar realization.
    global_noise = multiscale_continuous_noise(shape, 3, rng)[..., :1]
    values = base_patch.reshape(-1, 3).astype(np.float64)
    covariance = np.cov(values, rowvar=False) + np.eye(3) * 1e-10
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    material_axis = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    if float(np.sum(material_axis)) < 0.0:
        material_axis *= -1.0
    # Absolute displacement is intentionally capped well below one 8-bit step
    # per channel for strongly periodic materials and below two steps otherwise.
    source_axis_std = math.sqrt(max(float(np.max(eigenvalues)), 0.0))
    modulation_amplitude = min(0.006, source_axis_std * 0.10) * (modulation_strength / 0.018)
    base = np.clip(
        base + global_noise * (modulation_amplitude * material_axis).reshape(1, 1, 3),
        0.0,
        1.0,
    )

    normal = normal_rgb_to_vector(normal_rgb)
    normal[..., 0] *= sign_x
    normal[..., 1] *= sign_y
    normal = normalize_vectors(normal)
    if rough.ndim == 2:
        rough = rough[..., None]
    if metal.ndim == 2:
        metal = metal[..., None]
    del map_x, map_y, sign_x, sign_y, normal_rgb, global_noise
    return base, normal, np.clip(rough, 0.0, 1.0), np.clip(metal, 0.0, 1.0), {
        "pbr_spatial_mapping": "shared_scale_locked_irrational_continuous_phase_field",
        "source_coordinate_extension": "continuous_mirror_phase_with_incommensurate_global_warp",
        "contains_image_patch_cut_and_paste": False,
        "contains_fixed_tile_grid": False,
        "structured_material_axis_modulation_strength": modulation_strength,
        "structured_material_axis_modulation_amplitude": modulation_amplitude,
        **phase_stats,
    }


def histogram_lut_match(field: np.ndarray, source: np.ndarray, bins: int = 1024) -> np.ndarray:
    """Restore non-Gaussian material statistics without spatial patch reuse."""
    result = np.empty_like(field)
    quantiles = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    for channel in range(field.shape[2]):
        source_q = np.quantile(source[..., channel], quantiles).astype(np.float32)
        field_q = np.quantile(field[..., channel], quantiles).astype(np.float32)
        result[..., channel] = np.interp(
            field[..., channel].reshape(-1), field_q, source_q
        ).reshape(field.shape[:2])
    return np.clip(result, 0.0, 1.0)


def directional_structure_metrics(patch: np.ndarray) -> dict[str, float]:
    """Measure whether spatial evidence is dominated by one unoriented axis.

    Structure-tensor coherence alone can confuse a decorative border with wood
    grain.  The second angular moment of the native-scale power spectrum adds a
    separate test: it is large only when most measured energy shares one axis.
    Both tests are rotation invariant and use no semantic or face identity.
    """
    luminance = (
        0.2126 * patch[..., 0]
        + 0.7152 * patch[..., 1]
        + 0.0722 * patch[..., 2]
    ).astype(np.float32)
    low = cv2.GaussianBlur(
        luminance, (0, 0), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REFLECT_101
    )
    gradient_x = cv2.Sobel(low, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gradient_y = cv2.Sobel(low, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    tensor_xx = float(np.mean(cv2.GaussianBlur(gradient_x * gradient_x, (0, 0), 6.0)))
    tensor_yy = float(np.mean(cv2.GaussianBlur(gradient_y * gradient_y, (0, 0), 6.0)))
    tensor_xy = float(np.mean(cv2.GaussianBlur(gradient_x * gradient_y, (0, 0), 6.0)))
    tensor_energy = max(tensor_xx + tensor_yy, 1e-12)
    coherence = math.sqrt((tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy**2) / tensor_energy
    gradient_axis_radians = 0.5 * math.atan2(2.0 * tensor_xy, tensor_xx - tensor_yy)

    height, width = luminance.shape
    highpass_sigma = max(2.0, min(height, width) / 18.0)
    highpass = luminance - cv2.GaussianBlur(
        luminance,
        (0, 0),
        sigmaX=highpass_sigma,
        sigmaY=highpass_sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    window = np.sqrt(
        np.maximum(
            np.hanning(height).astype(np.float32)[:, None]
            * np.hanning(width).astype(np.float32)[None, :],
            0.0,
        )
    )
    power = np.abs(np.fft.fftshift(np.fft.fft2(highpass * window))) ** 2
    frequency_y = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    frequency_x = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    radius = np.sqrt(frequency_x * frequency_x + frequency_y * frequency_y)
    angle = np.arctan2(frequency_y, frequency_x)
    valid = (radius >= 2.0 / max(1, min(height, width))) & (radius <= 0.35)
    valid_power = power[valid]
    valid_angle = angle[valid]
    total_power = max(float(np.sum(valid_power)), 1e-12)
    second_moment = abs(np.sum(valid_power * np.exp(2j * valid_angle)) / total_power)
    fourth_moment = abs(np.sum(valid_power * np.exp(4j * valid_angle)) / total_power)
    return {
        "directional_structure_tensor_coherence": float(coherence),
        "directional_gradient_axis_radians": float(gradient_axis_radians),
        "directional_grain_axis_radians": float(gradient_axis_radians + math.pi / 2.0),
        "directional_spectral_second_moment": float(second_moment),
        "directional_spectral_fourth_moment": float(fourth_moment),
    }


def stochastic_joint_field_no_patch(
    base_patch: np.ndarray,
    normal_patch: np.ndarray,
    rough_patch: np.ndarray,
    metal_patch: np.ndarray,
    shape: tuple[int, int],
    sigma: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    base, normal, rough, metal, stats = smooth_joint_field(
        base_patch, normal_patch, rough_patch, metal_patch, shape, sigma, rng, args
    )
    matched = histogram_lut_match(base, base_patch)
    base = np.clip(0.20 * base + 0.80 * matched, 0.0, 1.0)
    stats.update(
        {
            "pbr_spatial_mapping": "shared_random_phase_native_scale_wholefield",
            "basecolor_distribution_mapping": "wholefield_quantile_projection",
            "contains_image_patch_cut_and_paste": False,
            "contains_fixed_tile_grid": False,
        }
    )
    return base, normal, rough, metal, stats


def macro_nonstationarity(patch: np.ndarray) -> float:
    sigma = max(3.0, 0.08 * min(patch.shape[:2]))
    macro = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    total = float(np.std(patch))
    return float(np.std(macro) / max(total, 1e-6))


def choose_route(
    metrics: dict[str, Any],
    directional_metrics: dict[str, float],
    macro_ratio: float,
    total_std: float,
    args: argparse.Namespace,
) -> str:
    highpass = float(metrics.get("structured_highpass_std", 1.0))
    edge = float(metrics.get("structured_edge_p95", 1.0))
    periodic = float(metrics.get("structured_periodic_max_corr", 0.0))
    axis = float(metrics.get("structured_axis_energy_min_ratio", 1.0))
    directional_coherence = float(
        directional_metrics.get("directional_structure_tensor_coherence", 0.0)
    )
    directional_second_moment = float(
        directional_metrics.get("directional_spectral_second_moment", 0.0)
    )
    # A single-axis native-scale field (wood grain, brushed material, long
    # veins) must never enter the two-axis mirror mapping.  The two independent
    # rotation-invariant measurements make this a material-evidence decision,
    # not a room/face/material-name exception.
    if (
        directional_coherence >= args.directional_coherence_min
        and directional_second_moment >= args.directional_spectral_second_moment_min
    ):
        return "directional_spectral"
    # Weak autocorrelation plus broad colour variation is not evidence of a
    # repeatable motif.  In particular, smooth wood/paint exemplars can have a
    # large low-frequency standard deviation while containing no spatial mark
    # that may safely be reflected.  The structured route therefore requires
    # visible native-scale edge/high-pass evidence, not total variance alone.
    visible_structure = bool(
        highpass >= args.structured_visible_highpass_min
        or edge >= args.structured_visible_edge_p95_min
    )
    structured = bool(
        (
            periodic >= args.structured_periodic_min_corr
            and (
                visible_structure
                or total_std >= args.structured_lowdetail_total_std_min
            )
        )
        or (
            axis <= args.structured_axis_ratio_max
            and (total_std >= 0.035 and (highpass >= 0.006 or edge >= 0.030))
        )
    )
    if structured:
        return "structured"
    # If weak periodic evidence exists but its visible structure is too weak to
    # justify reflection, preserve its measured native-scale spectrum in one
    # whole-domain realization.  This catches mirror-risk materials without a
    # material/face name and without flattening them into a constant colour.
    mirror_risk_without_visible_motif = bool(
        periodic >= args.structured_periodic_min_corr
        and not visible_structure
        and (
            (
                macro_ratio >= args.mirror_risk_macro_ratio_min
                and total_std < args.mirror_risk_total_std_max
            )
            or (
                total_std < args.mirror_risk_lowcontrast_std_max
                and edge >= args.mirror_risk_edge_p95_min
            )
        )
    )
    if mirror_risk_without_visible_motif:
        return "nonstationary_spectral"
    if highpass < args.smooth_highpass_max and edge < args.smooth_edge_p95_max:
        return "smooth"
    if macro_ratio >= 0.68:
        return "nonstationary_spectral"
    return "stochastic"


def save_overview(out_dir: Path, faces: list[str], width: int) -> None:
    gap = 14
    row_h = 230
    folders = ("basecolor", "normal", "roughness", "metallic")
    canvas = Image.new("RGB", ((len(folders) + 1) * width + 6 * gap, 54 + len(faces) * row_h), (244, 244, 244))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (gap, 16),
        "v3b_v2 adaptive whole-territory PBR generalized mirror guard",
        fill=(20, 20, 20),
        font=font,
    )
    for row, face in enumerate(faces):
        y = 54 + row * row_h
        draw.text((gap, y + 8), face, fill=(20, 20, 20), font=font)
        for col, folder in enumerate(folders):
            path = out_dir / "pbr_textures" / folder / f"{face}.png"
            image = Image.open(path).convert("RGB")
            image.thumbnail((width, row_h - 34), Image.Resampling.LANCZOS)
            x = gap + (col + 1) * (width + gap)
            draw.text((x, y + 5), folder, fill=(20, 20, 20), font=font)
            canvas.paste(image, (x, y + 25))
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / "previews" / "adaptive_whole_territory_pbr_overview.jpg", quality=94)


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    layout_metadata_path = args.layout_dir / "metadata_material_placement.json"
    layout_metadata = load_json(layout_metadata_path)
    input_metadata = load_json(args.chord_input_metadata)
    candidates = candidate_map(input_metadata)
    faces_by_name = face_record_map(layout_metadata)
    faces = args.faces or list(faces_by_name)
    face_records: list[dict[str, Any]] = []

    for face_index, face in enumerate(faces):
        labels_base = np.load(args.layout_dir / "labels_npy" / f"{face}.npy").astype(np.int16)
        shape = (
            int(round(labels_base.shape[0] * args.atlas_resolution_scale)),
            int(round(labels_base.shape[1] * args.atlas_resolution_scale)),
        )
        labels = resize_mask(labels_base, shape, linear=False).round().astype(np.int16)
        weight_path = args.layout_dir / "labels_npy" / f"{face}_lowfreq_weights.npy"
        if weight_path.exists():
            weights_base = np.load(weight_path).astype(np.float32)
            lowfreq_weights = np.stack([resize_mask(item, shape, linear=True) for item in weights_base], axis=0)
            lowfreq_weights /= np.maximum(np.sum(lowfreq_weights, axis=0, keepdims=True), 1e-6)
        else:
            count = int(labels.max()) + 1
            lowfreq_weights = np.stack([(labels == index).astype(np.float32) for index in range(count)], axis=0)

        base_fields: list[np.ndarray] = []
        normal_fields: list[np.ndarray] = []
        rough_fields: list[np.ndarray] = []
        metal_fields: list[np.ndarray] = []
        material_records: list[dict[str, Any]] = []
        materials = sorted(faces_by_name[face]["materials"], key=lambda item: int(item["material_index"]))
        for material in materials:
            index = int(material["material_index"])
            material_id = int(material.get("material_id", index))
            stem = str(material["chosen_stem"])
            candidate = candidates[stem]
            source_dir = args.pbr_chord_output_dir / stem
            base_patch, normal_patch, rough_patch, metal_patch, inner_box, source_side, period, support_mode = crop_and_scale_sources(
                source_dir, candidate, args.atlas_resolution_scale, args.chord_output_support_mode
            )
            local_rng = np.random.default_rng(args.seed + face_index * 1009 + index * 9173)
            lock_sigma, metrics = detail_lock_sigma(base_patch, args)
            directional_metrics = directional_structure_metrics(base_patch)
            macro_ratio = macro_nonstationarity(base_patch)
            total_std = float(np.std(base_patch))
            route = choose_route(metrics, directional_metrics, macro_ratio, total_std, args)
            periodic = float(metrics.get("structured_periodic_max_corr", 0.0))

            if route == "structured":
                base, normal, rough, metal, mapping = structured_joint_field(
                    base_patch, normal_patch, rough_patch, metal_patch, shape, periodic, local_rng, args
                )
            elif route == "smooth":
                sigma = max(6.0, min(18.0, 0.06 * period))
                base, normal, rough, metal, mapping = smooth_joint_field(
                    base_patch, normal_patch, rough_patch, metal_patch, shape, sigma, local_rng, args
                )
                mapping.update({"contains_image_patch_cut_and_paste": False, "contains_fixed_tile_grid": False})
                lock_sigma = sigma
            else:
                sigma = max(10.0, min(32.0, 0.055 * period))
                base, normal, rough, metal, mapping = stochastic_joint_field_no_patch(
                    base_patch, normal_patch, rough_patch, metal_patch, shape, sigma, local_rng, args
                )
                lock_sigma = sigma
                if route == "directional_spectral":
                    mapping.update(
                        {
                            "pbr_spatial_mapping": "shared_directional_native_scale_random_phase_wholefield",
                            "source_coordinate_extension": "none",
                            "contains_source_reflection": False,
                            "directional_route_reason": "single_axis_structure_tensor_and_spectral_agreement",
                        }
                    )

            fidelity = scale_fidelity(base_patch, base, local_rng)
            repeat = repetition_audit(base, base_patch.shape[:2])
            material_records.append(
                {
                    "material_index": index,
                    "material_id": material_id,
                    "chosen_stem": stem,
                    "selection_score_unchanged": material.get("selection_score"),
                    "traceback_source_view": candidate.get("view_name"),
                    "traceback_source_rank_by_weight": candidate.get("source_rank_by_weight"),
                    "traceback_source_weight": candidate.get("weight"),
                    "traceback_source_weight_fraction": candidate.get("weight_frac"),
                    "traceback_chord_input": candidate.get("chord_input"),
                    "traceback_selection_recomputed": False,
                    "source_chord_dir": str(source_dir),
                    "source_chord_basecolor_sha256": sha256(source_dir / "basecolor.png"),
                    "source_chord_shape_hw": [512, 512],
                    "inner_crop_box_y0_y1_x0_x1": inner_box,
                    "chord_output_support_mode": support_mode,
                    "support_mode_selected_from_candidate_contract": args.chord_output_support_mode == "auto",
                    "source_patch_side_atlas_px": source_side,
                    "physical_period_hw": [period, period],
                    "atlas_resolution_scale": float(args.atlas_resolution_scale),
                    "material_route": route,
                    "route_uses_face_or_material_name": False,
                    "macro_nonstationarity_ratio": macro_ratio,
                    "source_total_std": total_std,
                    "detail_lock_sigma_hr_px": float(lock_sigma),
                    "same_scale_dog_spectrum_cosine": fidelity["same_scale_dog_spectrum_cosine"],
                    **metrics,
                    **directional_metrics,
                    **mapping,
                    **repeat,
                    **normal_audit(normal),
                }
            )
            base_fields.append(base)
            normal_fields.append(normal)
            rough_fields.append(rough)
            metal_fields.append(metal)
            save_rgb(args.out_dir / "source_patches_scale_locked" / "basecolor" / face / f"material_{index:02d}_{stem}.png", base_patch)
            print(
                f"[generalized-mirrorguard] {face} m{index}: route={route} stem={stem} physical={period}px "
                f"scale={fidelity['same_scale_dog_spectrum_cosine']:.3f} "
                f"repeat={repeat['highpass_max_patch_period_repeat_correlation']:.3f}",
                flush=True,
            )
            del base_patch, normal_patch, rough_patch, metal_patch
            gc.collect()

        base_low_atlas = np.zeros((*shape, 3), np.float32)
        base_high_atlas = np.zeros((*shape, 3), np.float32)
        normal_atlas = np.zeros((*shape, 3), np.float32)
        rough_atlas = np.zeros((*shape, 1), np.float32)
        metal_atlas = np.zeros((*shape, 1), np.float32)
        boundary_sigma = max(1.0, args.lowfreq_boundary_sigma_frac * min(shape))
        for index, (base, normal, rough, metal) in enumerate(zip(base_fields, normal_fields, rough_fields, metal_fields)):
            hard = (labels == index).astype(np.float32)[..., None]
            low = cv2.GaussianBlur(base, (0, 0), sigmaX=boundary_sigma, sigmaY=boundary_sigma)
            base_high_atlas += hard * (base - low)
            base_low_atlas += lowfreq_weights[index, ..., None] * low
            normal_atlas += hard * normal
            rough_atlas += hard * rough
            metal_atlas += hard * metal
        base_atlas = np.clip(base_low_atlas + base_high_atlas, 0.0, 1.0)
        normal_atlas = normalize_vectors(normal_atlas)
        save_rgb(args.out_dir / "pbr_textures" / "basecolor" / f"{face}.png", base_atlas)
        save_rgb(args.out_dir / "pbr_textures" / "normal" / f"{face}.png", normal_vector_to_rgb(normal_atlas))
        save_scalar(args.out_dir / "pbr_textures" / "roughness" / f"{face}.png", np.clip(rough_atlas, 0.0, 1.0))
        save_scalar(args.out_dir / "pbr_textures" / "metallic" / f"{face}.png", np.clip(metal_atlas, 0.0, 1.0))
        save_rgb(args.out_dir / "textures_base" / f"{face}.png", base_atlas)
        save_rgb(args.out_dir / "labels" / f"{face}.png", LABEL_COLORS[np.maximum(labels, 0) % len(LABEL_COLORS)].astype(np.float32) / 255.0)
        (args.out_dir / "labels_npy").mkdir(parents=True, exist_ok=True)
        np.save(args.out_dir / "labels_npy" / f"{face}.npy", labels)
        face_records.append(
            {
                "face": face,
                "shape_hw": [shape[0], shape[1]],
                "materials": material_records,
                "hard_pbr_material_boundary": True,
                "basecolor_low_frequency_boundary_sigma_px": boundary_sigma,
            }
        )
        del base_fields, normal_fields, rough_fields, metal_fields
        del base_low_atlas, base_high_atlas, base_atlas, normal_atlas, rough_atlas, metal_atlas
        gc.collect()

    save_overview(args.out_dir, faces, args.preview_thumb_width)
    metadata = {
        "method": "v3b_v2_adaptive_whole_territory_scale_locked_pbr_generalized_mirror_guard",
        "upstream_contract": {
            "layout_metadata": str(layout_metadata_path),
            "layout_metadata_sha256": sha256(layout_metadata_path),
            "candidate_metadata": str(args.chord_input_metadata),
            "candidate_metadata_sha256": sha256(args.chord_input_metadata),
            "chosen_stems_and_labels_modified": False,
            "chord_model_rerun": False,
            "full_512_chord_outputs_required": True,
            "chord_output_support_mode": args.chord_output_support_mode,
        },
        "generalization_contract": {
            "face_type_used_for_routing": False,
            "material_name_used_for_routing": False,
            "routes": [
                "smooth",
                "stochastic",
                "directional_spectral",
                "structured",
                "nonstationary_spectral",
            ],
            "directional_extension": "single_native_scale_wholefield_no_reflection_no_patch_placement",
            "structured_extension": "scale_locked_continuous_phase_field_not_patch_placement",
            "stochastic_extension": "native_scale_random_phase_wholefield_not_patch_placement",
            "smooth_extension": "continuous_global_lowfield_plus_native_scale_microdetail",
            "low_detail_false_periodicity_guard": {
                "purpose": "prevent weak autocorrelation in low-detail materials from entering mirror-based structured synthesis",
                "uses_face_or_material_name": False,
                "structured_visible_highpass_min": args.structured_visible_highpass_min,
                "structured_visible_edge_p95_min": args.structured_visible_edge_p95_min,
                "structured_lowdetail_total_std_min": args.structured_lowdetail_total_std_min,
                "mirror_risk_macro_ratio_min": args.mirror_risk_macro_ratio_min,
                "mirror_risk_total_std_max": args.mirror_risk_total_std_max,
                "mirror_risk_lowcontrast_std_max": args.mirror_risk_lowcontrast_std_max,
                "mirror_risk_edge_p95_min": args.mirror_risk_edge_p95_min,
            },
        },
        "atlas_resolution_scale": float(args.atlas_resolution_scale),
        "faces": face_records,
    }
    (args.out_dir / "metadata_adaptive_whole_territory_pbr.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
