#!/usr/bin/env python3
"""Audit texel-scale fidelity and expected quilt-lattice boundary energy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--max-audit-side", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def float_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)


def band_signature(gray: np.ndarray, detail_cutoff: float) -> np.ndarray:
    cutoff = max(2.0, float(detail_cutoff))
    micro = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=cutoff, sigmaY=cutoff)
    values: list[float] = []
    for sigma in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        if 4.0 * sigma >= min(micro.shape):
            values.append(0.0)
            continue
        first = cv2.GaussianBlur(micro, (0, 0), sigmaX=sigma, sigmaY=sigma)
        second = cv2.GaussianBlur(micro, (0, 0), sigmaX=2.0 * sigma, sigmaY=2.0 * sigma)
        values.append(float(np.std(first - second)))
    vector = np.asarray(values, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def micro_scale_cosine(
    patch: np.ndarray,
    field: np.ndarray,
    detail_cutoff: float,
    rng: np.random.Generator,
) -> float:
    ph, pw = patch.shape[:2]
    # Never resize either image for a scale audit. A thin architectural
    # territory (for example a skirting strip) can be smaller than its source
    # exemplar in one dimension; compare equal-size native-pixel crops instead.
    sample_h = min(ph, field.shape[0])
    sample_w = min(pw, field.shape[1])
    py = (ph - sample_h) // 2
    px = (pw - sample_w) // 2
    source = patch[py : py + sample_h, px : px + sample_w]
    samples = []
    for _ in range(7):
        y = int(rng.integers(0, field.shape[0] - sample_h + 1))
        x = int(rng.integers(0, field.shape[1] - sample_w + 1))
        samples.append(field[y : y + sample_h, x : x + sample_w])
    source_signature = band_signature(float_gray(source), detail_cutoff)
    signatures = [band_signature(float_gray(sample), detail_cutoff) for sample in samples]
    field_signature = np.mean(np.stack(signatures), axis=0)
    field_signature /= max(float(np.linalg.norm(field_signature)), 1e-8)
    return float(np.dot(source_signature, field_signature))


def adaptive_material_field(run_dir: Path, face: str, material_index: int) -> np.ndarray:
    """Extract one native-scale material territory from a composed PBR face.

    The accepted adaptive backend intentionally does not retain a full-face
    scratch field for every material. Use the published label map to crop the
    final BaseColor territory and extend only pixels outside an irregular mask
    from their nearest valid material pixel. No image resampling is performed.
    """

    face_path = run_dir / "pbr_textures" / "basecolor" / f"{face}.png"
    labels_path = run_dir / "labels_npy" / f"{face}.npy"
    if not face_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"{face} material {material_index}: missing final face or labels")
    image = load_rgb(face_path)
    labels = np.load(labels_path)
    if labels.shape != image.shape[:2]:
        raise ValueError(f"{face}: labels {labels.shape} do not match BaseColor {image.shape[:2]}")
    mask = labels == material_index
    if not np.any(mask):
        raise ValueError(f"{face} material {material_index}: empty final territory")
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = image[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    if np.all(crop_mask):
        return crop
    nearest = ndimage.distance_transform_edt(
        ~crop_mask, return_distances=False, return_indices=True
    )
    filled = crop.copy()
    filled[~crop_mask] = crop[nearest[0][~crop_mask], nearest[1][~crop_mask]]
    return filled


def lattice_boundary_excess(
    image: np.ndarray,
    patch_side: int,
    block_frac: float,
    overlap_frac: float,
    max_side: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    scale = min(1.0, float(max_side) / max(image.shape[:2]))
    gray = float_gray(image)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    block = max(8, int(round(patch_side * block_frac * scale)))
    overlap = max(2, min(block - 1, int(round(block * overlap_frac))))
    step = max(1, block - overlap)
    low = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, 0.025 * step))
    gx = np.abs(np.diff(low, axis=1))
    gy = np.abs(np.diff(low, axis=0))

    def axis_score(gradient: np.ndarray, length: int, transpose: bool) -> float:
        starts = list(range(step, max(step + 1, length - 1), step))
        if not starts:
            return 1.0
        band = max(2, overlap)
        expected: list[float] = []
        random_values: list[float] = []
        for start in starts:
            lo, hi = max(0, start - 1), min(length - 1, start + band)
            section = gradient[:, lo:hi] if not transpose else gradient[lo:hi, :]
            if section.size:
                expected.append(float(np.mean(np.max(section, axis=1 if not transpose else 0))))
            random_start = int(rng.integers(1, max(2, length - band)))
            rlo, rhi = random_start - 1, min(length - 1, random_start + band)
            section = gradient[:, rlo:rhi] if not transpose else gradient[rlo:rhi, :]
            if section.size:
                random_values.append(float(np.mean(np.max(section, axis=1 if not transpose else 0))))
        return float(np.mean(expected) / max(np.mean(random_values), 1e-8))

    return {
        "expected_quilt_lattice_gradient_excess_x": axis_score(gx, gray.shape[1], False),
        "expected_quilt_lattice_gradient_excess_y": axis_score(gy, gray.shape[0], True),
        "audit_block_px": float(block),
        "audit_overlap_px": float(overlap),
        "audit_step_px": float(step),
    }


def main() -> int:
    args = parse_args()
    pbr_metadata_path = args.run_dir / "metadata_pbr_placement.json"
    if pbr_metadata_path.exists():
        metadata = json.loads(pbr_metadata_path.read_text(encoding="utf-8"))
        records = metadata["faces"]
        patch_root = args.run_dir / "source_patches_scale_locked" / "basecolor"
        field_root = None
        scale_key = "same_scale_dog_spectrum_cosine"
        adaptive_pbr = True
    else:
        metadata = json.loads((args.run_dir / "metadata_material_placement.json").read_text(encoding="utf-8"))
        records = metadata["scale_locked_wholefield"]["records"]
        patch_root = args.run_dir / "materials_patches_scale_locked"
        field_root = args.run_dir / "material_fields"
        scale_key = "same_scale_dog_spectrum_cosine"
        adaptive_pbr = False
    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []
    for face_record in records:
        face = str(face_record["face"])
        for material in face_record["materials"]:
            index = int(material["material_index"])
            patch_paths = sorted((patch_root / face).glob(f"material_{index:02d}_*.png"))
            field_paths = (
                []
                if adaptive_pbr
                else sorted((field_root / face).glob(f"material_{index:02d}_*.png"))
            )
            if len(patch_paths) != 1 or (not adaptive_pbr and len(field_paths) != 1):
                raise FileNotFoundError(f"{face} material {index}: patch={patch_paths}, field={field_paths}")
            patch = load_rgb(patch_paths[0])
            field = (
                adaptive_material_field(args.run_dir, face, index)
                if adaptive_pbr
                else load_rgb(field_paths[0])
            )
            cutoff = float(material["detail_lock_sigma_hr_px"])
            strategy = str(material.get("field_seed_strategy", material.get("material_route", "unknown")))
            backend = material.get("smooth_surface_backend", material.get("pbr_spatial_mapping"))
            result: dict[str, Any] = {
                "face": face,
                "material_index": index,
                "chosen_stem": material["chosen_stem"],
                "field_seed_strategy": strategy,
                "field_backend": backend,
                "microdetail_same_scale_band_cosine": micro_scale_cosine(
                    patch, field, cutoff, rng
                ),
                "reported_same_scale_dog_spectrum_cosine": float(
                    material[scale_key]
                ),
                "highpass_repeat_correlation": float(
                    material["highpass_max_patch_period_repeat_correlation"]
                ),
            }
            if "quilt" in (strategy + " " + str(backend or "")):
                result.update(
                    lattice_boundary_excess(
                        field,
                        min(patch.shape[:2]),
                        0.48,
                        0.42,
                        args.max_audit_side,
                        rng,
                    )
                )
            results.append(result)
            print(
                f"[audit] {face} m{index}: micro_scale={result['microdetail_same_scale_band_cosine']:.3f} "
                f"repeat={result['highpass_repeat_correlation']:.3f}",
                flush=True,
            )
    summary = {
        "run_dir": str(args.run_dir),
        "material_count": len(results),
        "minimum_microdetail_same_scale_band_cosine": min(
            result["microdetail_same_scale_band_cosine"] for result in results
        ),
        "maximum_highpass_repeat_correlation": max(
            result["highpass_repeat_correlation"] for result in results
        ),
        "materials": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "materials"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
