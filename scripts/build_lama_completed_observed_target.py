#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt


DEFAULT_FACES = ["floor", "ceiling"] + [f"wall_{index:02d}" for index in range(6)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use LaMa/IOPaint to inpaint holes in the strict real-observation atlas, "
            "then export it as completed_observed for material placement."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--faces", nargs="*", default=DEFAULT_FACES)
    parser.add_argument("--iopaint-bin", type=Path, default=Path("iopaint"))
    parser.add_argument("--model", default="lama")
    parser.add_argument("--model-dir", type=Path, default=Path("models/iopaint"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-dilate-px", type=int, default=2)
    parser.add_argument("--filled-weight", type=float, default=0.35)
    parser.add_argument("--far-filled-weight", type=float, default=0.04)
    parser.add_argument("--lama-local-distance-px", type=float, default=96.0)
    parser.add_argument("--max-lama-component-area-frac", type=float, default=0.012)
    parser.add_argument("--max-lama-component-distance-px", type=float, default=42.0)
    parser.add_argument("--smooth-fill-sigma-frac", type=float, default=0.018)
    parser.add_argument("--smooth-fill-iterations", type=int, default=4)
    parser.add_argument(
        "--completion-mode",
        choices=["legacy_raw_iopaint", "gated_local"],
        default="legacy_raw_iopaint",
        help=(
            "legacy_raw_iopaint matches the accepted v1 target: use the IOPaint/LaMa output "
            "directly and assign filled-weight confidence to holes. gated_local blends LaMa "
            "near observed pixels with smooth nearest-fill for large holes."
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-iopaint", action="store_true")
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if image.shape != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return image > 127


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def face_paths(source_dir: Path, face: str) -> tuple[Path, Path]:
    debug = source_dir / "debug"
    raw_path = debug / f"{face}_raw_projected.png"
    if not raw_path.exists():
        raw_path = source_dir / "textures" / f"{face}.png"
    mask_path = debug / f"{face}_final_keep_mask.png"
    if not mask_path.exists():
        mask_path = debug / f"{face}_observed_mask.png"
    return raw_path, mask_path


def valid_faces(source_dir: Path, requested: list[str]) -> list[str]:
    faces = []
    for face in requested:
        raw_path, mask_path = face_paths(source_dir, face)
        if raw_path.exists() and mask_path.exists():
            faces.append(face)
    return faces


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    size = 2 * pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy().astype(np.float32)
    red = np.array([245, 48, 58], dtype=np.float32)
    out[mask] = 0.45 * out[mask] + 0.55 * red
    return np.clip(out, 0, 255).astype(np.uint8)


def smooth_nearest_fill(image: np.ndarray, observed: np.ndarray, sigma_frac: float, iterations: int) -> np.ndarray:
    if np.all(observed):
        return image.copy()
    if not np.any(observed):
        return np.full_like(image, 128)
    _, nearest = distance_transform_edt(~observed, return_indices=True)
    filled = image.copy().astype(np.float32)
    unknown = ~observed
    filled[unknown] = image[nearest[0][unknown], nearest[1][unknown]].astype(np.float32)
    sigma = max(1.0, float(sigma_frac) * min(image.shape[:2]))
    for _ in range(max(1, int(iterations))):
        blurred = cv2.GaussianBlur(filled, (0, 0), sigma)
        filled[unknown] = blurred[unknown]
        filled[observed] = image[observed]
    return np.clip(filled, 0, 255).astype(np.uint8)


def gated_lama_completion(args: argparse.Namespace, raw: np.ndarray, observed: np.ndarray, lama: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lama.shape != raw.shape:
        lama = cv2.resize(lama, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
    unknown = ~observed
    if not np.any(unknown):
        confidence = np.ones(observed.shape, dtype=np.float32)
        return raw.copy(), confidence, np.zeros(observed.shape, dtype=np.float32)
    distance = distance_transform_edt(unknown).astype(np.float32)
    local_radius = max(float(args.lama_local_distance_px), 1.0)
    lama_alpha = np.exp(-((distance / local_radius) ** 2)).astype(np.float32)
    lama_alpha[observed] = 0.0
    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(unknown.astype(np.uint8), connectivity=8)
    max_area = float(args.max_lama_component_area_frac) * float(unknown.size)
    max_distance = float(args.max_lama_component_distance_px)
    allowed = np.zeros(unknown.shape, dtype=bool)
    for component in range(1, component_count):
        component_mask = component_labels == component
        area = float(stats[component, cv2.CC_STAT_AREA])
        component_max_distance = float(np.max(distance[component_mask])) if np.any(component_mask) else 0.0
        if area <= max_area or component_max_distance <= max_distance:
            allowed[component_mask] = True
    lama_alpha[~allowed] = 0.0
    smooth_fill = smooth_nearest_fill(
        raw,
        observed,
        args.smooth_fill_sigma_frac,
        args.smooth_fill_iterations,
    )
    completed = raw.copy().astype(np.float32)
    alpha = lama_alpha[..., None]
    completed[unknown] = alpha[unknown] * lama.astype(np.float32)[unknown] + (1.0 - alpha[unknown]) * smooth_fill.astype(np.float32)[unknown]
    confidence = np.ones(observed.shape, dtype=np.float32)
    confidence[unknown] = float(args.far_filled_weight) + (
        float(args.filled_weight) - float(args.far_filled_weight)
    ) * lama_alpha[unknown]
    return np.clip(completed, 0, 255).astype(np.uint8), np.clip(confidence, 0.0, 1.0), lama_alpha


def prepare_inputs(args: argparse.Namespace, faces: list[str]) -> list[dict]:
    image_dir = args.out_dir / "iopaint_input" / "images"
    mask_dir = args.out_dir / "iopaint_input" / "masks"
    overlay_dir = args.out_dir / "iopaint_input" / "overlays"
    observed_dir = args.out_dir / "observed"
    weight_dir = args.out_dir / "weights"
    for path in (image_dir, mask_dir, overlay_dir, observed_dir, weight_dir):
        path.mkdir(parents=True, exist_ok=True)

    records = []
    for face in faces:
        raw_path, mask_path = face_paths(args.source_dir, face)
        raw = load_rgb(raw_path)
        observed = load_mask(mask_path, raw.shape[:2])
        unknown = ~observed
        iopaint_mask = dilate_mask(unknown, args.mask_dilate_px)

        observed_image = raw.copy()
        observed_image[~observed] = 0
        save_rgb(observed_dir / f"{face}.png", observed_image)
        save_rgb(image_dir / f"{face}.png", observed_image)
        save_gray(mask_dir / f"{face}.png", iopaint_mask.astype(np.uint8) * 255)
        save_rgb(overlay_dir / f"{face}.png", overlay_mask(observed_image, iopaint_mask))

        initial_weight = np.where(observed, 255, int(round(255 * args.filled_weight))).astype(np.uint8)
        save_gray(weight_dir / f"{face}_initial.png", initial_weight)
        records.append(
            {
                "face": face,
                "raw_path": str(raw_path),
                "keep_mask_path": str(mask_path),
                "observed_fraction": float(np.mean(observed)),
                "inpaint_fraction": float(np.mean(iopaint_mask)),
                "shape_hw": [int(raw.shape[0]), int(raw.shape[1])],
            }
        )
        print(
            f"[lama-completed] prepared {face}: observed={records[-1]['observed_fraction']:.3f} "
            f"inpaint={records[-1]['inpaint_fraction']:.3f} shape={raw.shape[1]}x{raw.shape[0]}",
            flush=True,
        )
    return records


def run_iopaint(args: argparse.Namespace) -> None:
    output_dir = args.out_dir / "iopaint_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(str(args.iopaint_bin))
    if executable:
        cmd = [executable, "run"]
    else:
        cmd = [sys.executable, "-m", "iopaint", "run"]
    cmd.extend(
        [
            f"--model={args.model}",
            f"--device={args.device}",
            f"--image={args.out_dir / 'iopaint_input' / 'images'}",
            f"--mask={args.out_dir / 'iopaint_input' / 'masks'}",
            f"--output={output_dir}",
            f"--model-dir={args.model_dir}",
        ]
    )
    env = os.environ.copy()
    cache_root = str(Path.cwd() / ".cache")
    env.setdefault("XDG_CACHE_HOME", cache_root)
    env.setdefault("HF_HOME", str(Path(cache_root) / "huggingface"))
    env.setdefault("TORCH_HOME", str(Path(cache_root) / "torch"))
    env.setdefault("PIP_CACHE_DIR", str(Path(cache_root) / "pip"))
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    print("[lama-completed] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def collect_outputs(args: argparse.Namespace, faces: list[str]) -> None:
    completed_dir = args.out_dir / "completed_observed"
    raw_lama_dir = args.out_dir / "lama_raw"
    weight_dir = args.out_dir / "weights"
    alpha_dir = args.out_dir / "lama_local_alpha"
    completed_dir.mkdir(parents=True, exist_ok=True)
    raw_lama_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    for face in faces:
        candidates = [
            args.out_dir / "iopaint_output" / f"{face}.png",
            args.out_dir / "iopaint_output" / face / "output.png",
            args.out_dir / "iopaint_input" / "images" / f"{face}.png",
        ]
        source = next((path for path in candidates if path.exists()), None)
        if source is None:
            raise FileNotFoundError(f"Missing IOPaint output for {face}")
        raw_path, mask_path = face_paths(args.source_dir, face)
        raw = load_rgb(raw_path)
        observed = load_mask(mask_path, raw.shape[:2])
        unknown = ~observed
        raw_observed = raw.copy()
        raw_observed[~observed] = 0
        lama = load_rgb(source)
        if args.completion_mode == "legacy_raw_iopaint":
            if lama.shape != raw.shape:
                lama = cv2.resize(lama, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
            completed = lama.copy()
            confidence = np.where(observed, 1.0, float(args.filled_weight)).astype(np.float32)
            lama_alpha = unknown.astype(np.float32)
        else:
            completed, confidence, lama_alpha = gated_lama_completion(args, raw_observed, observed, lama)
        save_rgb(raw_lama_dir / f"{face}.png", lama)
        save_rgb(completed_dir / f"{face}.png", completed)
        save_gray(weight_dir / f"{face}.png", np.rint(confidence * 255.0).astype(np.uint8))
        save_gray(alpha_dir / f"{face}.png", np.rint(lama_alpha * 255.0).astype(np.uint8))
        print(
            f"[lama-completed] completed {face}: {source} "
            f"mean_weight={float(np.mean(confidence)):.3f} mean_lama_alpha={float(np.mean(lama_alpha)):.3f}",
            flush=True,
        )


def thumb(path: Path, max_size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def make_overview(args: argparse.Namespace, faces: list[str]) -> None:
    columns = [
        ("strict observed", args.out_dir / "observed"),
        ("LaMa hole mask", args.out_dir / "iopaint_input" / "masks"),
        ("raw LaMa", args.out_dir / "lama_raw"),
        ("completed_observed", args.out_dir / "completed_observed"),
        ("target weight", args.out_dir / "weights"),
    ]
    panel_w, panel_h = 340, 210
    gap = 14
    header = 74
    row_h = 248
    width = gap + len(columns) * (panel_w + gap)
    height = header + len(faces) * row_h + gap
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 14), "LaMa completed_observed target from strict projected atlas", fill=(20, 20, 20), font=font)
    for col, (title, _) in enumerate(columns):
        draw.text((gap + col * (panel_w + gap), 46), title, fill=(20, 20, 20), font=font)

    for row, face in enumerate(faces):
        y0 = header + row * row_h
        draw.text((gap, y0), face, fill=(20, 20, 20), font=font)
        for col, (_, directory) in enumerate(columns):
            path = directory / f"{face}.png"
            image = thumb(path, (panel_w, panel_h))
            x = gap + col * (panel_w + gap) + (panel_w - image.width) // 2
            y = y0 + 24 + (panel_h - image.height) // 2
            canvas.paste(image, (x, y))
    canvas.save(args.out_dir / "completed_observed_lama_overview.jpg", quality=94)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    faces = valid_faces(args.source_dir, args.faces)
    if not faces:
        raise RuntimeError(f"No valid faces found in {args.source_dir}")

    records = prepare_inputs(args, faces)
    if not args.prepare_only and not args.skip_iopaint:
        run_iopaint(args)
    collect_outputs(args, faces)
    make_overview(args, faces)
    metadata = {
        "method": "lama_completed_observed_from_strict_projection_v1",
        "source_dir": str(args.source_dir),
        "faces": records,
        "mask_dilate_px": int(args.mask_dilate_px),
        "completion_mode": args.completion_mode,
        "filled_weight": float(args.filled_weight),
        "far_filled_weight": float(args.far_filled_weight),
        "lama_local_distance_px": float(args.lama_local_distance_px),
        "max_lama_component_area_frac": float(args.max_lama_component_area_frac),
        "max_lama_component_distance_px": float(args.max_lama_component_distance_px),
        "smooth_fill_sigma_frac": float(args.smooth_fill_sigma_frac),
        "smooth_fill_iterations": int(args.smooth_fill_iterations),
        "iopaint": {
            "bin": str(args.iopaint_bin),
            "model": args.model,
            "model_dir": str(args.model_dir),
            "device": args.device,
            "skipped": bool(args.skip_iopaint or args.prepare_only),
        },
    }
    (args.out_dir / "metadata_completed_observed_lama.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[lama-completed] wrote {args.out_dir / 'completed_observed_lama_overview.jpg'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
