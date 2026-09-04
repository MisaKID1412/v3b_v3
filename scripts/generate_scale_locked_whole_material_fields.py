#!/usr/bin/env python3
"""Generate one non-periodic material field per face territory with scale lock.

The neural model is allowed to change only the low-frequency, whole-canvas
appearance. Mid/high-frequency material detail is synthesized from the traced
CHORD patch at the exact atlas texel scale and is restored after generation.
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
from scipy.fft import irfft2, rfft2

from compose_inversecrop_nontile_atlas_v1 import quilt_texture
from integrated_texture_fields import analyze_structure


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a unique whole-face field for each material while hard-locking "
            "the traced patch's atlas texel scale."
        )
    )
    parser.add_argument("--chord-input-metadata", type=Path, required=True)
    parser.add_argument("--chord-output-dir", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--observed-layout-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--faces", nargs="*", default=None)
    parser.add_argument("--atlas-resolution-scale", type=float, default=2.0)
    parser.add_argument("--atlas-ppm-da3-units", type=float, default=900.1076468379747)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument(
        "--seed-field-backend",
        choices=("quilt_exact_scale", "spectral_covariance"),
        default="quilt_exact_scale",
    )
    parser.add_argument("--quilt-block-frac", type=float, default=0.48)
    parser.add_argument("--quilt-overlap-frac", type=float, default=0.42)
    parser.add_argument("--quilt-min-block", type=int, default=112)
    parser.add_argument("--quilt-max-block", type=int, default=360)
    parser.add_argument(
        "--neural-backend",
        choices=("none", "sdxl_global_lowband"),
        default="sdxl_global_lowband",
    )
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
    parser.add_argument(
        "--smooth-surface-backend",
        choices=("continuous_spectral", "quilt_neural"),
        default="continuous_spectral",
        help=(
            "Smooth paint/plaster is synthesized as one continuous stochastic field by default; "
            "it never exposes a quilt grid to the neural model."
        ),
    )
    parser.add_argument("--smooth-lowfreq-max-std", type=float, default=0.022)
    parser.add_argument("--smooth-lowfreq-covariance-scale", type=float, default=0.65)
    parser.add_argument(
        "--continuous-stochastic-floor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a lattice-free continuous field for non-periodic stochastic floor materials "
            "such as terrazzo; periodic wood/tile structures stay on the structured route."
        ),
    )
    parser.add_argument("--detail-lock-min-sigma", type=float, default=12.0)
    parser.add_argument("--detail-lock-patch-frac", type=float, default=0.18)
    parser.add_argument("--neural-lowband-min-face-frac", type=float, default=0.08)
    parser.add_argument("--lowfreq-boundary-sigma-frac", type=float, default=0.002)
    parser.add_argument("--preview-thumb-width", type=int, default=340)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB").save(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_map(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for face_record in metadata.get("stats", []):
        for region in face_record.get("regions", []):
            for candidate in region.get("view_candidates", []):
                result[str(candidate["stem"])] = dict(candidate)
    return result


def face_record_map(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["face"]): record for record in metadata.get("faces", [])}


def crop_inner_patch(image: np.ndarray, candidate: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    h, w = image.shape[:2]
    box = candidate.get("inner_crop_box_y0_y1_x0_x1") or candidate.get("crop_box_y0_y1_x0_x1")
    if not box:
        return image.copy(), [0, h, 0, w]
    y0, y1, x0, x1 = [int(round(float(value))) for value in box]
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, y0 + 1, h))
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    return image[y0:y1, x0:x1].copy(), [y0, y1, x0, x1]


def resize_patch_exact_scale(patch: np.ndarray, source_side: float, scale: float) -> np.ndarray:
    target_side = max(16, int(round(float(source_side) * float(scale))))
    interpolation = cv2.INTER_CUBIC if target_side > min(patch.shape[:2]) else cv2.INTER_AREA
    return cv2.resize(patch, (target_side, target_side), interpolation=interpolation).astype(np.float32)


def spectral_covariance_field(
    patch: np.ndarray,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary non-periodic synthesis retaining the patch's pixel frequencies.

    A shared random Fourier phase is applied to all color channels, retaining
    their cross-channel phase relationships. No spatial coordinate or patch is
    resized inside this function.
    """
    h, w = int(shape[0]), int(shape[1])
    ph, pw = patch.shape[:2]
    padded_patch = np.pad(
        patch,
        ((max(1, ph // 2), max(1, ph // 2)), (max(1, pw // 2), max(1, pw // 2)), (0, 0)),
        mode="reflect",
    )
    window_y = np.hanning(padded_patch.shape[0]).astype(np.float32)
    window_x = np.hanning(padded_patch.shape[1]).astype(np.float32)
    window = np.sqrt(np.maximum(window_y[:, None] * window_x[None, :], 0.0)).astype(np.float32)
    random_phase = np.exp(
        1j * rng.uniform(-math.pi, math.pi, size=(h, w // 2 + 1)).astype(np.float32)
    ).astype(np.complex64)
    output = np.empty((h, w, 3), dtype=np.float32)
    source_mean = np.mean(patch.reshape(-1, 3), axis=0)
    source_std = np.std(patch.reshape(-1, 3), axis=0)
    for channel in range(3):
        residual = (padded_patch[..., channel] - float(source_mean[channel])) * window
        kernel = np.zeros((h, w), dtype=np.float32)
        kh = min(h, residual.shape[0])
        kw = min(w, residual.shape[1])
        kernel[:kh, :kw] = residual[:kh, :kw]
        kernel = np.roll(kernel, (-kh // 2, -kw // 2), axis=(0, 1))
        spectrum = rfft2(kernel, workers=1).astype(np.complex64, copy=False)
        synthesized = irfft2(spectrum * random_phase, s=(h, w), workers=1).astype(np.float32)
        synthesized -= float(np.mean(synthesized))
        std = float(np.std(synthesized))
        if std > 1e-7:
            synthesized *= float(source_std[channel]) / std
        output[..., channel] = synthesized + float(source_mean[channel])
        del kernel, spectrum, synthesized
    return np.clip(output, 0.0, 1.0)


def spectral_residual_field(
    residual_patch: np.ndarray,
    shape: tuple[int, int],
    rng: np.random.Generator,
    shared_phase: np.ndarray | None = None,
) -> np.ndarray:
    """Synthesize a zero-mean residual without changing its pixel frequency scale.

    The exemplar is embedded at native atlas resolution before the Fourier
    transform.  Only phase is randomized; no image or frequency axis is resized.
    This route is intentionally restricted to automatically detected smooth
    paint/plaster where the exemplar contains stochastic micro-detail rather
    than wood grain, aggregate, or a large decorative motif.
    """
    h, w = int(shape[0]), int(shape[1])
    ph, pw = residual_patch.shape[:2]
    window = np.sqrt(
        np.maximum(
            np.hanning(ph).astype(np.float32)[:, None]
            * np.hanning(pw).astype(np.float32)[None, :],
            0.0,
        )
    )
    if shared_phase is None:
        shared_phase = np.exp(
            1j * rng.uniform(-math.pi, math.pi, size=(h, w // 2 + 1)).astype(np.float32)
        ).astype(np.complex64)
    elif shared_phase.shape != (h, w // 2 + 1):
        raise ValueError(
            f"shared phase shape {shared_phase.shape} does not match {(h, w // 2 + 1)}"
        )
    channels = int(residual_patch.shape[2])
    output = np.empty((h, w, channels), dtype=np.float32)
    for channel in range(channels):
        source = residual_patch[..., channel].astype(np.float32)
        target_std = float(np.std(source))
        kernel = np.zeros((h, w), dtype=np.float32)
        kh, kw = min(h, ph), min(w, pw)
        kernel[:kh, :kw] = source[:kh, :kw] * window[:kh, :kw]
        kernel = np.roll(kernel, (-kh // 2, -kw // 2), axis=(0, 1))
        spectrum = rfft2(kernel, workers=1).astype(np.complex64, copy=False)
        synthesized = irfft2(spectrum * shared_phase, s=(h, w), workers=1).astype(np.float32)
        synthesized -= float(np.mean(synthesized))
        actual_std = float(np.std(synthesized))
        if actual_std > 1e-8:
            synthesized *= target_std / actual_std
        output[..., channel] = synthesized
        del kernel, spectrum, synthesized
    return output


def multiscale_continuous_noise(
    shape: tuple[int, int],
    channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a non-periodic whole-canvas field with no patch-sized lattice."""
    h, w = shape
    result = np.zeros((h, w, channels), dtype=np.float32)
    total_weight = 0.0
    for cells, weight in ((4, 1.0), (7, 0.55), (13, 0.28)):
        coarse_h = max(3, int(round(cells * h / max(h, w)))) + 2
        coarse_w = max(3, int(round(cells * w / max(h, w)))) + 2
        coarse = rng.normal(0.0, 1.0, size=(coarse_h, coarse_w, channels)).astype(np.float32)
        octave = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        result += float(weight) * octave
        total_weight += float(weight)
    result /= max(total_weight, 1e-8)
    flat = result.reshape(-1, channels)
    flat -= np.mean(flat, axis=0, keepdims=True)
    flat /= np.maximum(np.std(flat, axis=0, keepdims=True), 1e-6)
    return flat.reshape(h, w, channels)


def continuous_smooth_material_field(
    patch: np.ndarray,
    shape: tuple[int, int],
    detail_sigma: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one seam-free paint/plaster field at a locked texel scale.

    Low frequencies are a single global stochastic realization.  Micro-detail
    comes from the high-pass exemplar spectrum embedded at its existing pixel
    resolution.  Therefore neither component contains a repeated tile grid.
    """
    low_patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=detail_sigma, sigmaY=detail_sigma)
    micro_patch = patch - low_patch
    values = low_patch.reshape(-1, 3).astype(np.float64)
    mean = np.mean(values, axis=0).astype(np.float32)
    covariance = np.cov(values, rowvar=False).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance + np.eye(3) * 1e-10)
    max_variance = float(args.smooth_lowfreq_max_std) ** 2
    eigenvalues = np.clip(eigenvalues, 0.0, max_variance)
    covariance_root = (
        eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    ).astype(np.float32)
    global_noise = multiscale_continuous_noise(shape, 3, rng)
    low_variation = global_noise.reshape(-1, 3) @ covariance_root.T
    low_variation = low_variation.reshape(shape[0], shape[1], 3)
    low_field = mean.reshape(1, 1, 3) + float(args.smooth_lowfreq_covariance_scale) * low_variation
    micro_field = spectral_residual_field(micro_patch, shape, rng)
    field = np.clip(low_field + micro_field, 0.0, 1.0)
    return field, np.clip(low_field, 0.0, 1.0), {
        "smooth_surface_backend": "continuous_global_lowfreq_plus_native_scale_spectral_microdetail",
        "smooth_lowfreq_source_std_rgb": [float(v) for v in np.std(values, axis=0)],
        "smooth_microdetail_source_std_rgb": [
            float(v) for v in np.std(micro_patch.reshape(-1, 3), axis=0)
        ],
        "smooth_lowfreq_max_std": float(args.smooth_lowfreq_max_std),
        "smooth_lowfreq_covariance_scale": float(args.smooth_lowfreq_covariance_scale),
        "smooth_field_resampled_after_scale_lock": False,
        "smooth_field_contains_patch_lattice": False,
    }


def continuous_stochastic_material_field(
    patch: np.ndarray,
    shape: tuple[int, int],
    detail_sigma: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Preserve non-Gaussian grains while removing low-frequency quilt blocks.

    Terrazzo and aggregate have localized particles that random-phase synthesis
    turns into Gaussian noise.  We instead quilt only the zero-mean high-pass
    residual.  Since every source block has had its low-frequency color field
    removed, a rectangular placement cannot introduce rectangular shading.
    The global low-frequency component is synthesized once for the whole face.
    """
    low_patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=detail_sigma, sigmaY=detail_sigma)
    micro_patch = patch - low_patch
    absolute_scale = float(np.percentile(np.abs(micro_patch), 99.7))
    absolute_scale = max(absolute_scale, 1e-4)
    encoded = np.clip(0.5 + 0.46 * micro_patch / absolute_scale, 0.0, 1.0)
    encoded_field = quilt_texture(
        encoded,
        shape,
        rng,
        args.quilt_block_frac,
        args.quilt_overlap_frac,
        args.quilt_min_block,
        args.quilt_max_block,
    )
    micro_field = (encoded_field - 0.5) * (absolute_scale / 0.46)

    values = low_patch.reshape(-1, 3).astype(np.float64)
    mean = np.mean(values, axis=0).astype(np.float32)
    covariance = np.cov(values, rowvar=False).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance + np.eye(3) * 1e-10)
    max_variance = float(args.smooth_lowfreq_max_std) ** 2
    eigenvalues = np.clip(eigenvalues, 0.0, max_variance)
    covariance_root = (
        eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    ).astype(np.float32)
    global_noise = multiscale_continuous_noise(shape, 3, rng)
    low_variation = global_noise.reshape(-1, 3) @ covariance_root.T
    low_variation = low_variation.reshape(shape[0], shape[1], 3)
    low_field = mean.reshape(1, 1, 3) + float(args.smooth_lowfreq_covariance_scale) * low_variation
    field = np.clip(low_field + micro_field, 0.0, 1.0)
    return field, np.clip(low_field, 0.0, 1.0), {
        "smooth_surface_backend": "continuous_global_lowfreq_plus_native_scale_quilted_residual",
        "stochastic_residual_encoding_abs_scale": absolute_scale,
        "smooth_lowfreq_source_std_rgb": [float(v) for v in np.std(values, axis=0)],
        "smooth_microdetail_source_std_rgb": [
            float(v) for v in np.std(micro_patch.reshape(-1, 3), axis=0)
        ],
        "smooth_lowfreq_max_std": float(args.smooth_lowfreq_max_std),
        "smooth_lowfreq_covariance_scale": float(args.smooth_lowfreq_covariance_scale),
        "smooth_field_resampled_after_scale_lock": False,
        "smooth_field_contains_low_frequency_patch_lattice": False,
        "microdetail_phase_randomized": False,
    }


def resize_mask(mask: np.ndarray, shape: tuple[int, int], linear: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST
    return cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=interpolation)


def observed_condition(
    seed_field: np.ndarray,
    observed: np.ndarray,
    territory: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if observed.shape[:2] != seed_field.shape[:2]:
        observed = cv2.resize(observed, (seed_field.shape[1], seed_field.shape[0]), interpolation=cv2.INTER_LINEAR)
    observed_mask = np.any(observed > (2.0 / 255.0), axis=2) & territory
    if np.count_nonzero(observed_mask) < 64 or strength <= 0:
        return seed_field, {"observed_anchor_fraction": 0.0, "observed_anchor_strength": 0.0}
    source_values = observed[observed_mask]
    seed_values = seed_field[observed_mask]
    source_median = np.median(source_values, axis=0)
    seed_median = np.median(seed_values, axis=0)
    normalized = np.clip(observed - source_median.reshape(1, 1, 3) + seed_median.reshape(1, 1, 3), 0.0, 1.0)
    alpha = cv2.GaussianBlur(observed_mask.astype(np.float32), (0, 0), sigmaX=5.0)
    alpha = np.clip(alpha * float(strength), 0.0, float(strength))[..., None]
    conditioned = (1.0 - alpha) * seed_field + alpha * normalized
    return np.clip(conditioned, 0.0, 1.0), {
        "observed_anchor_fraction": float(np.mean(observed_mask)),
        "observed_anchor_strength": float(strength),
    }


def generation_shape(shape: tuple[int, int], max_side: int, min_side: int) -> tuple[int, int]:
    h, w = shape
    scale = min(float(max_side) / max(h, w), 1.0)
    gen_h = max(64, int(round(h * scale / 64.0)) * 64)
    gen_w = max(64, int(round(w * scale / 64.0)) * 64)
    if min(gen_h, gen_w) < min_side:
        grow = float(min_side) / max(min(gen_h, gen_w), 1)
        if max(gen_h, gen_w) * grow <= max_side * 1.35:
            gen_h = int(round(gen_h * grow / 64.0)) * 64
            gen_w = int(round(gen_w * grow / 64.0)) * 64
    return max(64, gen_h), max(64, gen_w)


class SDXLGlobalLowBand:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline

        self.torch = torch
        self.args = args
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            str(args.sdxl_model),
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
            use_safetensors=True,
        )
        self.pipe.load_lora_weights(
            str(args.texture_lora.parent),
            weight_name=args.texture_lora.name,
            adapter_name="texture_wholefield",
        )
        self.pipe.set_adapters("texture_wholefield", adapter_weights=float(args.lora_scale))
        self.pipe.enable_vae_tiling()
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        if args.model_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(args.device)
        self.negative = (
            "room, interior, furniture, object, horizon, perspective, camera view, directional "
            "lighting, shadow, text, logo, repeated grid, checkerboard, mosaic, stretched pattern"
        )

    def __call__(self, image: np.ndarray, seed: int, prompt: str) -> tuple[np.ndarray, dict[str, Any]]:
        gen_h, gen_w = generation_shape(
            image.shape[:2], self.args.generation_max_side, self.args.generation_min_side
        )
        init = cv2.resize(image, (gen_w, gen_h), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(np.clip(init * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB")
        generator = self.torch.Generator(device="cpu").manual_seed(int(seed))
        result = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative,
            image=pil,
            strength=float(self.args.img2img_strength),
            guidance_scale=float(self.args.guidance_scale),
            num_inference_steps=int(self.args.inference_steps),
            generator=generator,
            height=gen_h,
            width=gen_w,
        ).images[0]
        generated = np.asarray(result.convert("RGB"), dtype=np.float32) / 255.0
        return generated, {
            "neural_backend": "sdxl_global_lowband",
            "generation_shape_hw": [int(gen_h), int(gen_w)],
            "inference_steps": int(self.args.inference_steps),
            "guidance_scale": float(self.args.guidance_scale),
            "img2img_strength": float(self.args.img2img_strength),
            "lora_scale": float(self.args.lora_scale),
            "prompt": prompt,
            "negative_prompt": self.negative,
        }

    def close(self) -> None:
        del self.pipe
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def color_match(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    ref_lab = cv2.cvtColor(np.clip(reference * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    img_lab = cv2.cvtColor(np.clip(image * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_mean = np.mean(ref_lab.reshape(-1, 3), axis=0)
    ref_std = np.std(ref_lab.reshape(-1, 3), axis=0)
    img_mean = np.mean(img_lab.reshape(-1, 3), axis=0)
    img_std = np.std(img_lab.reshape(-1, 3), axis=0)
    matched = (img_lab - img_mean.reshape(1, 1, 3)) * (
        ref_std / np.maximum(img_std, 1e-4)
    ).reshape(1, 1, 3) + ref_mean.reshape(1, 1, 3)
    return cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def detail_lock_sigma(patch: np.ndarray, args: argparse.Namespace) -> tuple[float, dict[str, Any]]:
    metrics = analyze_structure(patch)
    periods = []
    if float(metrics.get("structured_period_corr_x", 0.0)) >= 0.12:
        periods.append(int(metrics.get("structured_period_x", 0)))
    if float(metrics.get("structured_period_corr_y", 0.0)) >= 0.12:
        periods.append(int(metrics.get("structured_period_y", 0)))
    period_sigma = 0.55 * max(periods) if periods else 0.0
    sigma = max(
        float(args.detail_lock_min_sigma),
        float(args.detail_lock_patch_frac) * min(patch.shape[:2]),
        period_sigma,
    )
    return float(sigma), {**metrics, "detected_periods_used_for_lock": periods}


def wholefield_prompt(face: str, patch: np.ndarray) -> tuple[str, str]:
    mean = np.mean(patch.reshape(-1, 3), axis=0)
    chroma = float(np.max(mean) - np.min(mean))
    brown = bool(mean[0] > 1.06 * mean[2] and mean[0] > mean[1] and chroma > 0.08)
    if face == "floor" and brown:
        material = "natural wood plank flooring"
        rule = "wood_floor_color_context"
    elif face == "floor":
        material = "light terrazzo stone flooring with fine mineral aggregate"
        rule = "terrazzo_floor_context"
    elif face == "ceiling":
        material = "continuous painted plaster ceiling surface"
        rule = "painted_ceiling_context"
    else:
        material = "continuous painted architectural wall surface"
        rule = "painted_wall_context"
    return (
        f"8k high resolution top down albedo map of {material}, exact same material and pattern "
        "scale as the input, flat orthographic, realistic subtle non-repeating variation, "
        "single continuous surface, no patch seams, photoscan, colormap",
        rule,
    )


def large_gaussian_lowpass(image: np.ndarray, sigma: float) -> np.ndarray:
    """Equivalent large-radius low pass without a huge full-resolution kernel."""
    if sigma <= 36.0:
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    scale = max(0.035, min(1.0, 28.0 / float(sigma)))
    small_w = max(32, int(round(image.shape[1] * scale)))
    small_h = max(32, int(round(image.shape[0] * scale)))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(
        small,
        (0, 0),
        sigmaX=max(1.0, sigma * scale),
        sigmaY=max(1.0, sigma * scale),
    )
    return cv2.resize(blurred, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)


def merge_global_lowband(seed_field: np.ndarray, generated_small: np.ndarray, sigma: float) -> np.ndarray:
    generated = cv2.resize(
        generated_small, (seed_field.shape[1], seed_field.shape[0]), interpolation=cv2.INTER_CUBIC
    )
    generated = color_match(seed_field, generated)
    generated_low = large_gaussian_lowpass(generated, sigma)
    seed_low = large_gaussian_lowpass(seed_field, sigma)
    seed_detail = seed_field - seed_low
    return np.clip(generated_low + seed_detail, 0.0, 1.0)


def dog_signature(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sigmas = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    values = []
    for sigma in sigmas:
        if 4.0 * sigma >= min(gray.shape):
            values.append(0.0)
            continue
        first = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        second = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0 * sigma)
        values.append(float(np.std(first - second)))
    array = np.asarray(values, dtype=np.float32)
    return array / max(float(np.linalg.norm(array)), 1e-8)


def scale_fidelity(patch: np.ndarray, field: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    ph, pw = patch.shape[:2]
    if field.shape[0] < ph or field.shape[1] < pw:
        sample = cv2.resize(field, (pw, ph), interpolation=cv2.INTER_AREA)
        samples = [sample]
    else:
        samples = []
        for _ in range(5):
            y0 = int(rng.integers(0, max(1, field.shape[0] - ph + 1)))
            x0 = int(rng.integers(0, max(1, field.shape[1] - pw + 1)))
            samples.append(field[y0 : y0 + ph, x0 : x0 + pw])
    source_signature = dog_signature(patch)
    field_signature = np.mean(np.stack([dog_signature(sample) for sample in samples]), axis=0)
    field_signature /= max(float(np.linalg.norm(field_signature)), 1e-8)
    cosine = float(np.dot(source_signature, field_signature))
    return {
        "same_scale_dog_spectrum_cosine": cosine,
        "source_dog_signature": [float(value) for value in source_signature],
        "field_dog_signature": [float(value) for value in field_signature],
        "sample_count": len(samples),
    }


def correlation_at_lag(gray: np.ndarray, dy: int, dx: int) -> float:
    y0a, y1a = max(0, dy), min(gray.shape[0], gray.shape[0] + dy)
    x0a, x1a = max(0, dx), min(gray.shape[1], gray.shape[1] + dx)
    y0b, y1b = max(0, -dy), min(gray.shape[0], gray.shape[0] - dy)
    x0b, x1b = max(0, -dx), min(gray.shape[1], gray.shape[1] - dx)
    if y1a <= y0a or x1a <= x0a:
        return 0.0
    first = gray[y0a:y1a, x0a:x1a].reshape(-1).astype(np.float32)
    second = gray[y0b:y1b, x0b:x1b].reshape(-1).astype(np.float32)
    first -= float(np.mean(first))
    second -= float(np.mean(second))
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denom) if denom > 1e-8 else 0.0


def repetition_audit(field: np.ndarray, patch_shape: tuple[int, int]) -> dict[str, Any]:
    small_scale = min(1.0, 640.0 / max(field.shape[:2]))
    gray = cv2.cvtColor(np.clip(field * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if small_scale < 1.0:
        gray = cv2.resize(gray, None, fx=small_scale, fy=small_scale, interpolation=cv2.INTER_AREA)
    # Remove illumination/global color drift before measuring patch-period
    # repetition.  Smooth walls otherwise report a false high correlation
    # simply because both compared regions have nearly the same mean color.
    highpass_sigma = max(1.0, 0.035 * min(patch_shape) * small_scale)
    gray = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=highpass_sigma, sigmaY=highpass_sigma)
    ph = max(1, int(round(patch_shape[0] * small_scale)))
    pw = max(1, int(round(patch_shape[1] * small_scale)))
    lags = [(0, pw), (ph, 0), (0, 2 * pw), (2 * ph, 0)]
    correlations = [correlation_at_lag(gray, dy, dx) for dy, dx in lags]
    return {
        "highpass_patch_period_lag_correlations": [float(value) for value in correlations],
        "highpass_max_patch_period_repeat_correlation": float(max(correlations)) if correlations else 0.0,
        "repeat_audit_highpass_sigma_px": float(highpass_sigma),
    }


def label_image(labels: np.ndarray) -> np.ndarray:
    return LABEL_COLORS[np.maximum(labels, 0) % len(LABEL_COLORS)].astype(np.float32) / 255.0


def save_overview(out_dir: Path, faces: list[str], thumb_width: int) -> None:
    gap = 16
    row_height = 270
    width = 3 * thumb_width + 4 * gap
    height = 54 + len(faces) * row_height + gap
    canvas = Image.new("RGB", (width, height), (244, 244, 244))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 16), "v3b_newStart: structured territory + scale-locked whole material field", fill=(20, 20, 20), font=font)
    for row, face in enumerate(faces):
        y = 54 + row * row_height
        paths = [
            out_dir / "labels" / f"{face}.png",
            out_dir / "textures_base" / f"{face}.png",
        ]
        draw.text((gap, y + 4), face, fill=(20, 20, 20), font=font)
        for col, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_width, row_height - 30), Image.Resampling.LANCZOS)
            x = gap + (col + 1) * (thumb_width + gap)
            draw.text((x, y + 4), path.parent.name, fill=(20, 20, 20), font=font)
            canvas.paste(image, (x, y + 24))
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / "previews" / "scale_locked_wholefield_overview.jpg", quality=95)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layout_metadata = load_json(args.layout_dir / "metadata_material_placement.json")
    inputs = load_json(args.chord_input_metadata)
    candidates = candidate_map(inputs)
    faces_by_name = face_record_map(layout_metadata)
    faces = args.faces or list(faces_by_name)
    rng = np.random.default_rng(args.seed)
    neural = SDXLGlobalLowBand(args) if args.neural_backend == "sdxl_global_lowband" else None
    records: list[dict[str, Any]] = []
    try:
        for face_index, face in enumerate(faces):
            labels_base = np.load(args.layout_dir / "labels_npy" / f"{face}.npy").astype(np.int16)
            hr_shape = (
                max(1, int(round(labels_base.shape[0] * args.atlas_resolution_scale))),
                max(1, int(round(labels_base.shape[1] * args.atlas_resolution_scale))),
            )
            labels = resize_mask(labels_base, hr_shape, linear=False).round().astype(np.int16)
            lowfreq_path = args.layout_dir / "labels_npy" / f"{face}_lowfreq_weights.npy"
            if lowfreq_path.exists():
                lowfreq_base = np.load(lowfreq_path).astype(np.float32)
                lowfreq_weights = np.stack(
                    [resize_mask(channel, hr_shape, linear=True) for channel in lowfreq_base], axis=0
                )
                lowfreq_weights /= np.maximum(np.sum(lowfreq_weights, axis=0, keepdims=True), 1e-6)
            else:
                count = int(labels.max()) + 1
                lowfreq_weights = np.stack([(labels == index).astype(np.float32) for index in range(count)], axis=0)
            observed_path = args.observed_layout_dir / "observed_reference" / f"{face}.png"
            observed = load_rgb(observed_path) if observed_path.exists() else np.zeros((*labels_base.shape, 3), dtype=np.float32)
            observed = cv2.resize(observed, (hr_shape[1], hr_shape[0]), interpolation=cv2.INTER_LINEAR)
            fields: list[np.ndarray] = []
            material_records: list[dict[str, Any]] = []
            materials = sorted(
                faces_by_name[face].get("materials", []), key=lambda item: int(item["material_index"])
            )
            for material in materials:
                index = int(material["material_index"])
                material_id = int(material.get("material_id", index))
                stem = str(material["chosen_stem"])
                candidate = candidates[stem]
                source_path = args.chord_output_dir / stem / "basecolor.png"
                source = load_rgb(source_path)
                patch, inner_box = crop_inner_patch(source, candidate)
                source_side = float(candidate.get("inner_crop_side") or min(patch.shape[:2]))
                patch_scaled = resize_patch_exact_scale(patch, source_side, args.atlas_resolution_scale)
                lock_sigma, structure_stats = detail_lock_sigma(patch_scaled, args)
                smooth_surface = bool(
                    args.adaptive_smooth_surface_lock
                    and face != "floor"
                    and float(structure_stats.get("structured_highpass_std", 1.0)) < 0.012
                    and float(structure_stats.get("structured_edge_p95", 1.0)) < 0.040
                    and float(structure_stats.get("structured_periodic_max_corr", 1.0)) < 0.25
                )
                stochastic_floor = bool(
                    args.continuous_stochastic_floor
                    and face == "floor"
                    and float(structure_stats.get("structured_periodic_max_corr", 1.0)) < 0.25
                )
                if smooth_surface:
                    lock_sigma = max(
                        6.0,
                        min(18.0, 0.06 * min(patch_scaled.shape[:2])),
                    )
                    neural_injection_sigma = lock_sigma
                elif stochastic_floor:
                    # Terrazzo/aggregate is stochastic, so its grains can be
                    # randomized in phase without changing their pixel scale.
                    # Keep the cutoff below the large visual structures used by
                    # the periodic wood/tile route.
                    lock_sigma = max(
                        16.0,
                        min(36.0, 0.065 * min(patch_scaled.shape[:2])),
                    )
                    neural_injection_sigma = lock_sigma
                else:
                    neural_injection_sigma = max(
                        lock_sigma,
                        float(args.neural_lowband_min_face_frac) * min(hr_shape),
                    )
                structure_stats["adaptive_smooth_surface_lock_applied"] = smooth_surface
                structure_stats["continuous_stochastic_floor_applied"] = stochastic_floor
                territory = labels == index
                continuous_surface = bool(
                    (smooth_surface and args.smooth_surface_backend == "continuous_spectral")
                    or stochastic_floor
                )
                if continuous_surface:
                    if stochastic_floor:
                        final_field, generated_small, smooth_stats = continuous_stochastic_material_field(
                            patch_scaled, hr_shape, lock_sigma, rng, args
                        )
                    else:
                        final_field, generated_small, smooth_stats = continuous_smooth_material_field(
                            patch_scaled, hr_shape, lock_sigma, rng, args
                        )
                    field_seed = final_field.copy()
                    neural_condition = generated_small
                    anchor_stats = {
                        "observed_anchor_fraction": 0.0,
                        "observed_anchor_strength": 0.0,
                    }
                    continuous_reason = "stochastic_floor" if stochastic_floor else "smooth_surface"
                    neural_stats: dict[str, Any] = {
                        "neural_backend": f"not_called_for_continuous_{continuous_reason}",
                        "neural_generation_skipped": True,
                        "continuous_field_reason": continuous_reason,
                        **smooth_stats,
                    }
                    prompt_rule = f"not_used_for_continuous_{continuous_reason}"
                    seed_strategy = "single_continuous_field_without_patch_lattice"
                else:
                    if args.seed_field_backend == "quilt_exact_scale":
                        field_seed = quilt_texture(
                            patch_scaled,
                            hr_shape,
                            rng,
                            args.quilt_block_frac,
                            args.quilt_overlap_frac,
                            args.quilt_min_block,
                            args.quilt_max_block,
                        )
                        seed_strategy = "large_block_mincut_quilt_same_texel_scale"
                    else:
                        field_seed = spectral_covariance_field(patch_scaled, hr_shape, rng)
                        seed_strategy = "global_random_phase_covariance_same_texel_scale"
                    neural_condition, anchor_stats = observed_condition(
                        field_seed, observed, territory, args.observed_anchor_strength
                    )
                    prompt, prompt_rule = wholefield_prompt(face, patch_scaled)
                    if neural is None:
                        final_field = field_seed
                        generated_small = cv2.resize(
                            neural_condition,
                            (min(1024, hr_shape[1]), min(1024, hr_shape[0])),
                            interpolation=cv2.INTER_AREA,
                        )
                        neural_stats = {"neural_backend": "none", "neural_generation_skipped": True}
                    else:
                        generated_small, neural_stats = neural(
                            neural_condition, args.seed + 1000 * face_index + index, prompt
                        )
                        # The generated canvas contributes only below the lock
                        # cutoff; the exact-scale material detail is restored.
                        final_field = merge_global_lowband(
                            field_seed, generated_small, neural_injection_sigma
                        )
                source_out = args.out_dir / "materials_source" / face / f"material_{index:02d}_id{material_id}_{stem}_chord.png"
                patch_out = args.out_dir / "materials_patches_scale_locked" / face / f"material_{index:02d}_id{material_id}_{stem}_patch.png"
                seed_out = args.out_dir / "material_fields_seed" / face / f"material_{index:02d}_id{material_id}_{stem}_field.png"
                condition_out = args.out_dir / "material_fields_condition" / face / f"material_{index:02d}_id{material_id}_{stem}_condition.png"
                lowband_out = args.out_dir / "material_fields_lowband" / face / f"material_{index:02d}_id{material_id}_{stem}_lowband.png"
                field_out = args.out_dir / "material_fields" / face / f"material_{index:02d}_id{material_id}_{stem}_field.png"
                save_rgb(source_out, source)
                save_rgb(patch_out, patch_scaled)
                save_rgb(seed_out, field_seed)
                save_rgb(condition_out, neural_condition)
                save_rgb(lowband_out, generated_small)
                save_rgb(field_out, final_field)
                fidelity = scale_fidelity(patch_scaled, final_field, rng)
                repeat = repetition_audit(final_field, patch_scaled.shape[:2])
                record = {
                    "material_index": index,
                    "material_id": material_id,
                    "chosen_stem": stem,
                    "source_chord_basecolor": str(source_path),
                    "inner_crop_box_y0_y1_x0_x1": inner_box,
                    "source_patch_side_atlas_px": source_side,
                    "atlas_resolution_scale": float(args.atlas_resolution_scale),
                    "scale_locked_patch_shape_hw": [int(v) for v in patch_scaled.shape[:2]],
                    "physical_patch_extent_da3_units": float(source_side / max(args.atlas_ppm_da3_units, 1e-8)),
                    "locked_mid_high_detail_resampled_after_scale_lock": False,
                    "neural_low_frequency_canvas_resampled": (
                        neural_stats.get("neural_backend") == "sdxl_global_lowband"
                    ),
                    "detail_lock_sigma_hr_px": float(lock_sigma),
                    "neural_injection_lowpass_sigma_hr_px": float(neural_injection_sigma),
                    "wholefield_prompt_rule": prompt_rule,
                    "field_seed_strategy": seed_strategy,
                    "scale_locked_patch": str(patch_out),
                    "material_field_seed": str(seed_out),
                    "wholefield_condition_preview": str(condition_out),
                    "wholefield_lowband_preview": str(lowband_out),
                    "material_field": str(field_out),
                    **anchor_stats,
                    **structure_stats,
                    **neural_stats,
                    **fidelity,
                    **repeat,
                }
                material_records.append(record)
                fields.append(final_field)
                print(
                    f"[scale-locked-wholefield] {face} m{index}: {stem} "
                    f"patch={patch_scaled.shape[1]}px field={hr_shape[1]}x{hr_shape[0]} "
                    f"spectrum={fidelity['same_scale_dog_spectrum_cosine']:.3f} "
                    f"highpass_repeat={repeat['highpass_max_patch_period_repeat_correlation']:.3f}",
                    flush=True,
                )
                del source, patch, patch_scaled, field_seed, neural_condition, final_field, generated_small
                gc.collect()
            hard_atlas = np.zeros((*hr_shape, 3), dtype=np.float32)
            low_atlas = np.zeros_like(hard_atlas)
            high_atlas = np.zeros_like(hard_atlas)
            boundary_sigma = max(1.0, args.lowfreq_boundary_sigma_frac * min(hr_shape))
            for index, field in enumerate(fields):
                low = cv2.GaussianBlur(field, (0, 0), sigmaX=boundary_sigma, sigmaY=boundary_sigma)
                high = field - low
                hard = (labels == index).astype(np.float32)[..., None]
                hard_atlas += hard * field
                high_atlas += hard * high
                low_atlas += lowfreq_weights[index, ..., None] * low
            atlas = np.clip(high_atlas + low_atlas, 0.0, 1.0)
            save_rgb(args.out_dir / "textures_base_hard_debug" / f"{face}.png", hard_atlas)
            save_rgb(args.out_dir / "textures_base" / f"{face}.png", atlas)
            save_rgb(args.out_dir / "labels" / f"{face}.png", label_image(labels))
            (args.out_dir / "labels_npy").mkdir(parents=True, exist_ok=True)
            np.save(args.out_dir / "labels_npy" / f"{face}.npy", labels.astype(np.int16))
            np.save(
                args.out_dir / "labels_npy" / f"{face}_soft_weights.npy",
                np.stack([(labels == index).astype(np.float32) for index in range(len(fields))], axis=0),
            )
            records.append(
                {
                    "face": face,
                    "source_shape_hw": [int(v) for v in labels_base.shape],
                    "wholefield_shape_hw": [int(v) for v in hr_shape],
                    "material_count": len(fields),
                    "hard_high_frequency_boundary": True,
                    "low_frequency_boundary_sigma_hr_px": float(boundary_sigma),
                    "materials": material_records,
                }
            )
            del fields, hard_atlas, low_atlas, high_atlas, atlas
            gc.collect()
    finally:
        if neural is not None:
            neural.close()

    save_overview(args.out_dir, faces, args.preview_thumb_width)
    output_metadata = dict(layout_metadata)
    output_metadata["method"] = "structured_territory_scale_locked_whole_material_field_v2"
    output_metadata["scale_locked_wholefield"] = {
        "source_layout": str(args.layout_dir),
        "observed_condition_source": str(args.observed_layout_dir),
        "chord_output_dir": str(args.chord_output_dir),
        "neural_backend": args.neural_backend,
        "scale_contract": (
            "The traced CHORD patch is first mapped to atlas_resolution_scale exactly once. "
            "For structured materials, the whole-canvas neural result contributes only frequencies "
            "below the recorded detail-lock cutoff and exact-scale mid/high detail is restored. "
            "For automatically detected smooth paint/plaster, a single continuous low-frequency "
            "field is combined with native-pixel spectral micro-detail and no quilt lattice or neural "
            "resizing is used. High-frequency material territory boundaries are hard."
        ),
        "records": records,
    }
    (args.out_dir / "metadata_material_placement.json").write_text(
        json.dumps(output_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
