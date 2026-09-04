#!/usr/bin/env python3
"""Generate aligned whole-face PBR maps from full trace-back CHORD outputs.

BaseColor decides every exemplar placement. Normal, roughness, and metallic are
transferred through the exact same graph-cut masks or spectral phase. They are
never synthesized with independent spatial randomness.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compose_inversecrop_nontile_atlas_v1 import quilt_texture
from generate_scale_locked_whole_material_fields import (
    LABEL_COLORS,
    SDXLGlobalLowBand,
    candidate_map,
    detail_lock_sigma,
    face_record_map,
    large_gaussian_lowpass,
    load_json,
    load_rgb,
    merge_global_lowband,
    multiscale_continuous_noise,
    observed_condition,
    repetition_audit,
    resize_mask,
    scale_fidelity,
    spectral_residual_field,
    wholefield_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chord-input-metadata", type=Path, required=True)
    parser.add_argument("--pbr-chord-output-dir", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--observed-layout-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--faces", nargs="*", default=None)
    parser.add_argument("--atlas-resolution-scale", type=float, default=2.0)
    parser.add_argument("--atlas-ppm-da3-units", type=float, default=900.1076468379747)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--quilt-block-frac", type=float, default=0.48)
    parser.add_argument("--quilt-overlap-frac", type=float, default=0.42)
    parser.add_argument("--quilt-min-block", type=int, default=112)
    parser.add_argument("--quilt-max-block", type=int, default=360)
    parser.add_argument("--neural-backend", choices=("none", "sdxl_global_lowband"), default="sdxl_global_lowband")
    parser.add_argument(
        "--sdxl-model",
        type=Path,
        default=Path("models/stable-diffusion-xl-base-1.0"),
    )
    parser.add_argument(
        "--texture-lora",
        type=Path,
        default=Path("models/texture-synthesis-topdown-base-condensed.safetensors"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-max-side", type=int, default=1024)
    parser.add_argument("--generation-min-side", type=int, default=384)
    parser.add_argument("--inference-steps", type=int, default=18)
    parser.add_argument("--guidance-scale", type=float, default=5.5)
    parser.add_argument("--img2img-strength", type=float, default=0.28)
    parser.add_argument("--lora-scale", type=float, default=0.65)
    parser.add_argument("--observed-anchor-strength", type=float, default=0.0)
    parser.add_argument("--adaptive-smooth-surface-lock", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continuous-stochastic-floor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--material-routing",
        choices=("v1_face", "generalized"),
        default="v1_face",
        help=(
            "v1_face preserves the accepted v1 face-specific behavior; generalized selects "
            "smooth, stochastic, or structured synthesis only from exemplar statistics"
        ),
    )
    parser.add_argument("--generalized-structured-periodic-min-corr", type=float, default=0.25)
    parser.add_argument("--generalized-structured-axis-ratio-max", type=float, default=0.55)
    parser.add_argument("--detail-lock-min-sigma", type=float, default=12.0)
    parser.add_argument("--detail-lock-patch-frac", type=float, default=0.18)
    parser.add_argument("--neural-lowband-min-face-frac", type=float, default=0.08)
    parser.add_argument("--smooth-lowfreq-max-std", type=float, default=0.022)
    parser.add_argument("--smooth-lowfreq-covariance-scale", type=float, default=0.65)
    parser.add_argument("--lowfreq-boundary-sigma-frac", type=float, default=0.002)
    parser.add_argument("--preview-thumb-width", type=int, default=300)
    return parser.parse_args()


def generalized_wholefield_prompt(
    patch: np.ndarray,
    structure_stats: dict[str, Any],
) -> tuple[str, str]:
    """Choose a structured-material prompt without using face type or orientation."""
    mean = np.mean(patch.reshape(-1, 3), axis=0)
    chroma = float(np.max(mean) - np.min(mean))
    brown = bool(mean[0] > 1.06 * mean[2] and mean[0] > mean[1] and chroma > 0.08)
    periodic = float(structure_stats.get("structured_periodic_max_corr", 0.0))
    axis_ratio = float(structure_stats.get("structured_axis_energy_min_ratio", 1.0))
    if brown and (periodic >= 0.25 or axis_ratio <= 0.55):
        material = "continuous natural wood plank material surface"
        rule = "structured_warm_material_wood_likelihood"
    else:
        material = (
            "the same continuous directional architectural material as the input, preserving "
            "its motif family, orientation, spacing, and color distribution"
        )
        rule = "structured_material_appearance_only"
    return (
        f"8k high resolution flat orthographic albedo map of {material}, exact same material "
        "and pattern scale as the input, realistic subtle non-repeating variation, one "
        "continuous surface, no room context, no objects, no patch seams, photoscan, colormap",
        rule,
    )


def load_scalar(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)[..., None] / 255.0


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB").save(path)


def save_scalar(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = image[..., 0] if image.ndim == 3 else image
    Image.fromarray(np.clip(data * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L").save(path)


def crop_by_box(image: np.ndarray, box: list[int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    return image[y0:y1, x0:x1].copy()


def crop_inner_patch(image: np.ndarray, candidate: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    height, width = image.shape[:2]
    values = candidate.get("inner_crop_box_y0_y1_x0_x1") or candidate.get("crop_box_y0_y1_x0_x1")
    if not values:
        return image.copy(), [0, height, 0, width]
    y0, y1, x0, x1 = [int(round(float(value))) for value in values]
    y0 = int(np.clip(y0, 0, height - 1))
    y1 = int(np.clip(y1, y0 + 1, height))
    x0 = int(np.clip(x0, 0, width - 1))
    x1 = int(np.clip(x1, x0 + 1, width))
    return image[y0:y1, x0:x1].copy(), [y0, y1, x0, x1]


def resize_exact(image: np.ndarray, side: int) -> np.ndarray:
    interpolation = cv2.INTER_CUBIC if side > min(image.shape[:2]) else cv2.INTER_AREA
    return cv2.resize(image, (side, side), interpolation=interpolation).astype(np.float32)


def normal_rgb_to_vector(image: np.ndarray) -> np.ndarray:
    return normalize_vectors(image * 2.0 - 1.0)


def normal_vector_to_rgb(vector: np.ndarray) -> np.ndarray:
    return np.clip(normalize_vectors(vector) * 0.5 + 0.5, 0.0, 1.0)


def normalize_vectors(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=2, keepdims=True)
    fallback = np.zeros_like(vector)
    fallback[..., 2] = 1.0
    return np.where(norm > 1e-6, vector / np.maximum(norm, 1e-6), fallback).astype(np.float32)


def pack_auxiliary(normal_vector: np.ndarray, roughness: np.ndarray, metallic: np.ndarray) -> np.ndarray:
    return np.concatenate([normal_vector, roughness, metallic], axis=2).astype(np.float32)


def unpack_auxiliary(auxiliary: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = normalize_vectors(auxiliary[..., :3])
    roughness = np.clip(auxiliary[..., 3:4], 0.0, 1.0)
    metallic = np.clip(auxiliary[..., 4:5], 0.0, 1.0)
    return normal, roughness, metallic


def lowfield_from_patch(
    low_patch: np.ndarray,
    shape: tuple[int, int],
    global_noise: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if low_patch.ndim == 2:
        low_patch = low_patch[..., None]
    values = low_patch.reshape(-1, low_patch.shape[2]).astype(np.float64)
    mean = np.mean(values, axis=0).astype(np.float32)
    if low_patch.shape[2] == 1:
        std = min(float(np.std(values)), float(args.smooth_lowfreq_max_std))
        return (
            mean.reshape(1, 1, 1)
            + float(args.smooth_lowfreq_covariance_scale) * std * global_noise[..., :1]
        ).astype(np.float32)
    covariance = np.cov(values, rowvar=False).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance + np.eye(low_patch.shape[2]) * 1e-10
    )
    eigenvalues = np.clip(eigenvalues, 0.0, float(args.smooth_lowfreq_max_std) ** 2)
    root = (eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T).astype(np.float32)
    noise = global_noise[..., : low_patch.shape[2]]
    variation = noise.reshape(-1, low_patch.shape[2]) @ root.T
    return (
        mean.reshape(1, 1, -1)
        + float(args.smooth_lowfreq_covariance_scale) * variation.reshape(*shape, -1)
    ).astype(np.float32)


def smooth_joint_field(
    base_patch: np.ndarray,
    normal_patch: np.ndarray,
    roughness_patch: np.ndarray,
    metallic_patch: np.ndarray,
    shape: tuple[int, int],
    sigma: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    base_low_patch = cv2.GaussianBlur(base_patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    base_residual = base_patch - base_low_patch
    global_noise = multiscale_continuous_noise(shape, 3, rng)
    base_low = lowfield_from_patch(base_low_patch, shape, global_noise, args)
    phase = np.exp(
        1j * rng.uniform(-math.pi, math.pi, size=(shape[0], shape[1] // 2 + 1)).astype(np.float32)
    ).astype(np.complex64)
    base_micro = spectral_residual_field(base_residual, shape, rng, shared_phase=phase)

    normal_vector = normal_rgb_to_vector(normal_patch)
    normal_low_patch = cv2.GaussianBlur(normal_vector, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normal_residual = normal_vector - normal_low_patch
    rough_low_patch = cv2.GaussianBlur(roughness_patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    metal_low_patch = cv2.GaussianBlur(metallic_patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if rough_low_patch.ndim == 2:
        rough_low_patch = rough_low_patch[..., None]
    if metal_low_patch.ndim == 2:
        metal_low_patch = metal_low_patch[..., None]
    auxiliary_residual = pack_auxiliary(
        normal_residual,
        roughness_patch - rough_low_patch,
        metallic_patch - metal_low_patch,
    )
    auxiliary_micro = spectral_residual_field(
        auxiliary_residual, shape, rng, shared_phase=phase
    )
    normal_mean = np.mean(normal_low_patch.reshape(-1, 3), axis=0)
    normal_field = normalize_vectors(normal_mean.reshape(1, 1, 3) + auxiliary_micro[..., :3])
    roughness_field = np.clip(
        lowfield_from_patch(rough_low_patch, shape, global_noise, args) + auxiliary_micro[..., 3:4],
        0.0,
        1.0,
    )
    metallic_field = np.clip(
        lowfield_from_patch(metal_low_patch, shape, global_noise, args) + auxiliary_micro[..., 4:5],
        0.0,
        1.0,
    )
    return (
        np.clip(base_low + base_micro, 0.0, 1.0),
        normal_field,
        roughness_field,
        metallic_field,
        {
            "pbr_spatial_mapping": "shared_random_fourier_phase_at_native_patch_pixel_scale",
            "pbr_low_frequency_mapping": "shared_continuous_global_noise",
        },
    )


def stochastic_joint_field(
    base_patch: np.ndarray,
    normal_patch: np.ndarray,
    roughness_patch: np.ndarray,
    metallic_patch: np.ndarray,
    shape: tuple[int, int],
    sigma: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    base_low_patch = cv2.GaussianBlur(base_patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    base_residual = base_patch - base_low_patch
    absolute_scale = max(float(np.percentile(np.abs(base_residual), 99.7)), 1e-4)
    encoded_base = np.clip(0.5 + 0.46 * base_residual / absolute_scale, 0.0, 1.0)

    normal_vector = normal_rgb_to_vector(normal_patch)
    normal_low_patch = cv2.GaussianBlur(normal_vector, (0, 0), sigmaX=sigma, sigmaY=sigma)
    rough_low_patch = cv2.GaussianBlur(roughness_patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    metal_low_patch = cv2.GaussianBlur(metallic_patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if rough_low_patch.ndim == 2:
        rough_low_patch = rough_low_patch[..., None]
    if metal_low_patch.ndim == 2:
        metal_low_patch = metal_low_patch[..., None]
    auxiliary_residual = pack_auxiliary(
        normal_vector - normal_low_patch,
        roughness_patch - rough_low_patch,
        metallic_patch - metal_low_patch,
    )
    encoded_field, auxiliary_field = quilt_texture(
        encoded_base,
        shape,
        rng,
        args.quilt_block_frac,
        args.quilt_overlap_frac,
        args.quilt_min_block,
        args.quilt_max_block,
        auxiliary_source=auxiliary_residual,
    )
    base_micro = (encoded_field - 0.5) * (absolute_scale / 0.46)
    global_noise = multiscale_continuous_noise(shape, 3, rng)
    base_low = lowfield_from_patch(base_low_patch, shape, global_noise, args)
    normal_mean = np.mean(normal_low_patch.reshape(-1, 3), axis=0)
    normal_field = normalize_vectors(normal_mean.reshape(1, 1, 3) + auxiliary_field[..., :3])
    roughness_field = np.clip(
        lowfield_from_patch(rough_low_patch, shape, global_noise, args) + auxiliary_field[..., 3:4],
        0.0,
        1.0,
    )
    metallic_field = np.clip(
        lowfield_from_patch(metal_low_patch, shape, global_noise, args) + auxiliary_field[..., 4:5],
        0.0,
        1.0,
    )
    return (
        np.clip(base_low + base_micro, 0.0, 1.0),
        normal_field,
        roughness_field,
        metallic_field,
        {
            "pbr_spatial_mapping": "shared_graphcut_and_alpha_for_native_scale_particle_residual",
            "pbr_low_frequency_mapping": "shared_continuous_global_noise",
            "stochastic_residual_encoding_abs_scale": absolute_scale,
        },
    )


def structured_joint_seed(
    base_patch: np.ndarray,
    normal_patch: np.ndarray,
    roughness_patch: np.ndarray,
    metallic_patch: np.ndarray,
    shape: tuple[int, int],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    auxiliary = pack_auxiliary(normal_rgb_to_vector(normal_patch), roughness_patch, metallic_patch)
    base_seed, auxiliary_field = quilt_texture(
        base_patch,
        shape,
        rng,
        args.quilt_block_frac,
        args.quilt_overlap_frac,
        args.quilt_min_block,
        args.quilt_max_block,
        auxiliary_source=auxiliary,
    )
    normal, roughness, metallic = unpack_auxiliary(auxiliary_field)
    return base_seed, normal, roughness, metallic, {
        "pbr_spatial_mapping": "shared_graphcut_candidate_crop_and_alpha_at_native_patch_scale",
        "pbr_low_frequency_mapping": "CHORD_PBR_exemplar_field",
    }


def normal_audit(normal_vector: np.ndarray) -> dict[str, float]:
    length = np.linalg.norm(normal_vector, axis=2)
    return {
        "normal_length_mean": float(np.mean(length)),
        "normal_length_p01": float(np.percentile(length, 1)),
        "normal_length_p99": float(np.percentile(length, 99)),
        "normal_invalid_fraction": float(np.mean(~np.isfinite(length))),
    }


def save_overview(out_dir: Path, faces: list[str], width: int) -> None:
    folders = ("basecolor", "normal", "roughness", "metallic")
    gap = 14
    row_height = 230
    canvas = Image.new(
        "RGB", ((len(folders) + 1) * width + (len(folders) + 2) * gap, 48 + len(faces) * row_height), (244, 244, 244)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 14), "v3b_newStart aligned scale-locked PBR", fill=(20, 20, 20), font=font)
    for row, face in enumerate(faces):
        y = 48 + row * row_height
        draw.text((gap, y + 8), face, fill=(20, 20, 20), font=font)
        for column, folder in enumerate(folders):
            path = out_dir / "pbr_textures" / folder / f"{face}.png"
            image = Image.open(path).convert("RGB")
            image.thumbnail((width, row_height - 34), Image.Resampling.LANCZOS)
            x = gap + (column + 1) * (width + gap)
            draw.text((x, y + 6), folder, fill=(20, 20, 20), font=font)
            canvas.paste(image, (x, y + 26))
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / "previews" / "aligned_scale_locked_pbr_overview.jpg", quality=94)


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    layout_metadata = load_json(args.layout_dir / "metadata_material_placement.json")
    input_metadata = load_json(args.chord_input_metadata)
    candidates = candidate_map(input_metadata)
    faces_by_name = face_record_map(layout_metadata)
    faces = args.faces or list(faces_by_name)
    rng = np.random.default_rng(args.seed)
    neural = SDXLGlobalLowBand(args) if args.neural_backend == "sdxl_global_lowband" else None
    face_records: list[dict[str, Any]] = []
    try:
        for face_index, face in enumerate(faces):
            labels_base = np.load(args.layout_dir / "labels_npy" / f"{face}.npy").astype(np.int16)
            shape = (
                int(round(labels_base.shape[0] * args.atlas_resolution_scale)),
                int(round(labels_base.shape[1] * args.atlas_resolution_scale)),
            )
            labels = resize_mask(labels_base, shape, linear=False).round().astype(np.int16)
            lowfreq_path = args.layout_dir / "labels_npy" / f"{face}_lowfreq_weights.npy"
            if lowfreq_path.exists():
                weights_base = np.load(lowfreq_path).astype(np.float32)
                lowfreq = np.stack([resize_mask(item, shape, linear=True) for item in weights_base], axis=0)
                lowfreq /= np.maximum(np.sum(lowfreq, axis=0, keepdims=True), 1e-6)
            else:
                count = int(labels.max()) + 1
                lowfreq = np.stack([(labels == index).astype(np.float32) for index in range(count)], axis=0)
            observed_path = args.observed_layout_dir / "observed_reference" / f"{face}.png"
            observed = load_rgb(observed_path) if observed_path.exists() else np.zeros((*labels_base.shape, 3), np.float32)
            observed = cv2.resize(observed, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

            base_fields: list[np.ndarray] = []
            normal_fields: list[np.ndarray] = []
            roughness_fields: list[np.ndarray] = []
            metallic_fields: list[np.ndarray] = []
            material_records: list[dict[str, Any]] = []
            materials = sorted(faces_by_name[face]["materials"], key=lambda item: int(item["material_index"]))
            for material in materials:
                index = int(material["material_index"])
                material_id = int(material.get("material_id", index))
                stem = str(material["chosen_stem"])
                candidate = candidates[stem]
                source_dir = args.pbr_chord_output_dir / stem
                required = {name: source_dir / f"{name}.png" for name in ("basecolor", "normal", "roughness", "metallic")}
                for path in required.values():
                    if not path.exists():
                        raise FileNotFoundError(path)
                source_base = load_rgb(required["basecolor"])
                source_normal = load_rgb(required["normal"])
                source_roughness = load_scalar(required["roughness"])
                source_metallic = load_scalar(required["metallic"])
                patch_base, inner_box = crop_inner_patch(source_base, candidate)
                patch_normal = crop_by_box(source_normal, inner_box)
                patch_roughness = crop_by_box(source_roughness, inner_box)
                patch_metallic = crop_by_box(source_metallic, inner_box)
                source_side = float(candidate.get("inner_crop_side") or min(patch_base.shape[:2]))
                target_side = max(16, int(round(source_side * args.atlas_resolution_scale)))
                patch_base = resize_exact(patch_base, target_side)
                patch_normal = normal_vector_to_rgb(normal_rgb_to_vector(resize_exact(patch_normal, target_side)))
                patch_roughness = np.clip(resize_exact(patch_roughness, target_side), 0.0, 1.0)
                if patch_roughness.ndim == 2:
                    patch_roughness = patch_roughness[..., None]
                patch_metallic = np.clip(resize_exact(patch_metallic, target_side), 0.0, 1.0)
                if patch_metallic.ndim == 2:
                    patch_metallic = patch_metallic[..., None]

                lock_sigma, structure_stats = detail_lock_sigma(patch_base, args)
                highpass_std = float(structure_stats.get("structured_highpass_std", 1.0))
                edge_p95 = float(structure_stats.get("structured_edge_p95", 1.0))
                periodic_corr = float(structure_stats.get("structured_periodic_max_corr", 1.0))
                axis_ratio = float(structure_stats.get("structured_axis_energy_min_ratio", 1.0))
                if args.material_routing == "generalized":
                    smooth = bool(
                        args.adaptive_smooth_surface_lock
                        and highpass_std < 0.012
                        and edge_p95 < 0.040
                        and periodic_corr < args.generalized_structured_periodic_min_corr
                    )
                    structured = bool(
                        not smooth
                        and (
                            periodic_corr >= args.generalized_structured_periodic_min_corr
                            or axis_ratio <= args.generalized_structured_axis_ratio_max
                        )
                    )
                    stochastic = bool(not smooth and not structured)
                else:
                    smooth = bool(
                        args.adaptive_smooth_surface_lock
                        and face != "floor"
                        and highpass_std < 0.012
                        and edge_p95 < 0.040
                        and periodic_corr < 0.25
                    )
                    stochastic = bool(
                        args.continuous_stochastic_floor
                        and face == "floor"
                        and periodic_corr < 0.25
                    )
                    structured = bool(not smooth and not stochastic)
                material_route = "smooth" if smooth else "stochastic" if stochastic else "structured"
                if smooth:
                    lock_sigma = max(6.0, min(18.0, 0.06 * target_side))
                    base_field, normal_field, roughness_field, metallic_field, mapping_stats = smooth_joint_field(
                        patch_base, patch_normal, patch_roughness, patch_metallic, shape, lock_sigma, rng, args
                    )
                    neural_stats: dict[str, Any] = {"neural_backend": "not_called_for_continuous_smooth_surface"}
                    seed_strategy = "continuous_smooth_joint_pbr"
                elif stochastic:
                    lock_sigma = max(16.0, min(36.0, 0.065 * target_side))
                    base_field, normal_field, roughness_field, metallic_field, mapping_stats = stochastic_joint_field(
                        patch_base, patch_normal, patch_roughness, patch_metallic, shape, lock_sigma, rng, args
                    )
                    neural_stats = {"neural_backend": "not_called_for_continuous_stochastic_floor"}
                    seed_strategy = "continuous_stochastic_joint_pbr"
                else:
                    base_seed, normal_field, roughness_field, metallic_field, mapping_stats = structured_joint_seed(
                        patch_base, patch_normal, patch_roughness, patch_metallic, shape, rng, args
                    )
                    territory = labels == index
                    condition, anchor_stats = observed_condition(
                        base_seed, observed, territory, args.observed_anchor_strength
                    )
                    injection_sigma = max(lock_sigma, float(args.neural_lowband_min_face_frac) * min(shape))
                    if args.material_routing == "generalized":
                        prompt, prompt_rule = generalized_wholefield_prompt(patch_base, structure_stats)
                    else:
                        prompt, prompt_rule = wholefield_prompt(face, patch_base)
                    if neural is None:
                        base_field = base_seed
                        neural_stats = {"neural_backend": "none"}
                    else:
                        generated, neural_stats = neural(
                            condition, args.seed + 1000 * face_index + index, prompt
                        )
                        base_field = merge_global_lowband(base_seed, generated, injection_sigma)
                        del generated
                    mapping_stats.update(anchor_stats)
                    mapping_stats["wholefield_prompt_rule"] = prompt_rule
                    seed_strategy = "structured_shared_graphcut_joint_pbr"

                prefix = f"material_{index:02d}_id{material_id}_{stem}"
                patch_outputs = {
                    "basecolor": args.out_dir / "materials_patches_scale_locked" / "basecolor" / face / f"{prefix}.png",
                    "normal": args.out_dir / "materials_patches_scale_locked" / "normal" / face / f"{prefix}.png",
                    "roughness": args.out_dir / "materials_patches_scale_locked" / "roughness" / face / f"{prefix}.png",
                    "metallic": args.out_dir / "materials_patches_scale_locked" / "metallic" / face / f"{prefix}.png",
                }
                field_outputs = {
                    "basecolor": args.out_dir / "material_fields" / "basecolor" / face / f"{prefix}.png",
                    "normal": args.out_dir / "material_fields" / "normal" / face / f"{prefix}.png",
                    "roughness": args.out_dir / "material_fields" / "roughness" / face / f"{prefix}.png",
                    "metallic": args.out_dir / "material_fields" / "metallic" / face / f"{prefix}.png",
                }
                save_rgb(patch_outputs["basecolor"], patch_base)
                save_rgb(patch_outputs["normal"], patch_normal)
                save_scalar(patch_outputs["roughness"], patch_roughness)
                save_scalar(patch_outputs["metallic"], patch_metallic)
                save_rgb(field_outputs["basecolor"], base_field)
                save_rgb(field_outputs["normal"], normal_vector_to_rgb(normal_field))
                save_scalar(field_outputs["roughness"], roughness_field)
                save_scalar(field_outputs["metallic"], metallic_field)

                fidelity = scale_fidelity(patch_base, base_field, rng)
                repeat = repetition_audit(base_field, patch_base.shape[:2])
                material_record = {
                    "material_index": index,
                    "material_id": material_id,
                    "chosen_stem": stem,
                    "source_full_chord_dir": str(source_dir),
                    "inner_crop_box_y0_y1_x0_x1": inner_box,
                    "source_patch_side_atlas_px": source_side,
                    "scale_locked_patch_shape_hw": [target_side, target_side],
                    "atlas_resolution_scale": float(args.atlas_resolution_scale),
                    "locked_pbr_channels_resampled_after_scale_lock": False,
                    "field_seed_strategy": seed_strategy,
                    "material_routing": args.material_routing,
                    "material_route": material_route,
                    "material_route_uses_face_type": args.material_routing != "generalized",
                    "material_route_metrics": {
                        "highpass_std": highpass_std,
                        "edge_p95": edge_p95,
                        "periodic_corr": periodic_corr,
                        "axis_energy_min_ratio": axis_ratio,
                    },
                    "detail_lock_sigma_hr_px": float(lock_sigma),
                    "patch_outputs": {key: str(value) for key, value in patch_outputs.items()},
                    "field_outputs": {key: str(value) for key, value in field_outputs.items()},
                    "basecolor_same_scale_dog_spectrum_cosine": fidelity["same_scale_dog_spectrum_cosine"],
                    **structure_stats,
                    **mapping_stats,
                    **neural_stats,
                    **repeat,
                    **normal_audit(normal_field),
                    "roughness_min": float(np.min(roughness_field)),
                    "roughness_max": float(np.max(roughness_field)),
                    "metallic_min": float(np.min(metallic_field)),
                    "metallic_max": float(np.max(metallic_field)),
                }
                material_records.append(material_record)
                base_fields.append(base_field)
                normal_fields.append(normal_field)
                roughness_fields.append(roughness_field)
                metallic_fields.append(metallic_field)
                print(
                    f"[joint-pbr] {face} m{index}: route={material_route} {seed_strategy} "
                    f"scale={fidelity['same_scale_dog_spectrum_cosine']:.3f} "
                    f"repeat={repeat['highpass_max_patch_period_repeat_correlation']:.3f}",
                    flush=True,
                )
                del source_base, source_normal, source_roughness, source_metallic
                del patch_base, patch_normal, patch_roughness, patch_metallic
                gc.collect()

            base_atlas = np.zeros((*shape, 3), np.float32)
            base_low_atlas = np.zeros_like(base_atlas)
            base_high_atlas = np.zeros_like(base_atlas)
            normal_atlas = np.zeros((*shape, 3), np.float32)
            roughness_atlas = np.zeros((*shape, 1), np.float32)
            metallic_atlas = np.zeros((*shape, 1), np.float32)
            boundary_sigma = max(1.0, args.lowfreq_boundary_sigma_frac * min(shape))
            for index, (base, normal, roughness, metallic) in enumerate(
                zip(base_fields, normal_fields, roughness_fields, metallic_fields)
            ):
                hard = (labels == index).astype(np.float32)[..., None]
                low = cv2.GaussianBlur(base, (0, 0), sigmaX=boundary_sigma, sigmaY=boundary_sigma)
                base_high_atlas += hard * (base - low)
                base_low_atlas += lowfreq[index, ..., None] * low
                normal_atlas += hard * normal
                roughness_atlas += hard * roughness
                metallic_atlas += hard * metallic
            base_atlas = np.clip(base_high_atlas + base_low_atlas, 0.0, 1.0)
            normal_atlas = normalize_vectors(normal_atlas)
            roughness_atlas = np.clip(roughness_atlas, 0.0, 1.0)
            metallic_atlas = np.clip(metallic_atlas, 0.0, 1.0)
            save_rgb(args.out_dir / "pbr_textures" / "basecolor" / f"{face}.png", base_atlas)
            save_rgb(args.out_dir / "pbr_textures" / "normal" / f"{face}.png", normal_vector_to_rgb(normal_atlas))
            save_scalar(args.out_dir / "pbr_textures" / "roughness" / f"{face}.png", roughness_atlas)
            save_scalar(args.out_dir / "pbr_textures" / "metallic" / f"{face}.png", metallic_atlas)
            save_rgb(args.out_dir / "textures_base" / f"{face}.png", base_atlas)
            save_rgb(args.out_dir / "labels" / f"{face}.png", LABEL_COLORS[np.maximum(labels, 0) % len(LABEL_COLORS)].astype(np.float32) / 255.0)
            (args.out_dir / "labels_npy").mkdir(parents=True, exist_ok=True)
            np.save(args.out_dir / "labels_npy" / f"{face}.npy", labels)
            face_records.append(
                {
                    "face": face,
                    "shape_hw": [int(shape[0]), int(shape[1])],
                    "material_count": len(material_records),
                    "hard_pbr_material_boundary": True,
                    "basecolor_low_frequency_boundary_sigma_px": float(boundary_sigma),
                    "materials": material_records,
                    **normal_audit(normal_atlas),
                }
            )
            del base_fields, normal_fields, roughness_fields, metallic_fields
            del base_atlas, base_low_atlas, base_high_atlas, normal_atlas, roughness_atlas, metallic_atlas
            gc.collect()
    finally:
        if neural is not None:
            neural.close()

    save_overview(args.out_dir, faces, args.preview_thumb_width)
    metadata = {
        "method": (
            "matseg_traceback_full_chord_generalized_material_routing_scale_locked_pbr_v1"
            if args.material_routing == "generalized"
            else "matseg_traceback_full_chord_shared_mapping_scale_locked_pbr_v1"
        ),
        "material_routing": args.material_routing,
        "material_routing_contract": {
            "face_type_used": args.material_routing != "generalized",
            "routes": ["smooth", "stochastic", "structured"],
            "statistics": [
                "highpass_std",
                "edge_p95",
                "periodic_corr",
                "axis_energy_min_ratio",
            ],
            "structured_periodic_min_corr": float(args.generalized_structured_periodic_min_corr),
            "structured_axis_ratio_max": float(args.generalized_structured_axis_ratio_max),
            "structured_prompt_uses_face_type": args.material_routing != "generalized",
        },
        "source_layout": str(args.layout_dir),
        "source_full_chord_outputs": str(args.pbr_chord_output_dir),
        "source_chord_input_metadata": str(args.chord_input_metadata),
        "atlas_resolution_scale": float(args.atlas_resolution_scale),
        "channel_contract": {
            "channels": ["basecolor", "normal", "roughness", "metallic"],
            "shared_spatial_mapping": True,
            "independent_channel_quilting": False,
            "normal_vector_renormalized_after_every_blend": True,
            "material_boundary": "hard_for_normal_roughness_metallic_and_basecolor_high_frequency",
            "scale": "all CHORD patches mapped to target atlas scale exactly once before synthesis",
        },
        "faces": face_records,
    }
    (args.out_dir / "metadata_pbr_placement.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
