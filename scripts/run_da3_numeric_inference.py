#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 and save full numeric depth/conf/camera outputs."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--ref-view-strategy", default="saddle_balanced")
    parser.add_argument("--export-format", default=None)
    parser.add_argument("--conf-thresh-percentile", type=float, default=40.0)
    parser.add_argument("--num-max-points", type=int, default=1000000)
    parser.add_argument("--hide-cameras", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260626,
        help="Seed numpy/torch/python before DA3 inference and GLB downsampling. Use -1 to disable.",
    )
    return parser.parse_args()


def find_images(image_dir: Path) -> list[str]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(str(image_dir / pattern)))
    paths = sorted(set(paths))
    return paths


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    images = find_images(args.image_dir)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise FileNotFoundError(f"No input images found under {args.image_dir}")

    # Optional autocast override. DA3 picks bfloat16 whenever
    # torch.cuda.is_bf16_supported() is True, which on Turing (RTX 2080 Ti,
    # sm75) is emulated: the memory-efficient SDPA kernel then refuses bf16 and
    # PyTorch falls back to the math backend that materializes every attention
    # matrix, so >100-view scenes OOM on 11GB. DA3_AUTOCAST_DTYPE=float16
    # selects DA3's own fp16 fallback path (the one it uses on non-bf16 GPUs).
    autocast_override = os.environ.get("DA3_AUTOCAST_DTYPE", "").strip().lower()
    if not autocast_override and torch.cuda.is_available():
        # PyTorch builds on Turing cards can report bf16 support even though
        # attention then falls back to the quadratic math kernel.  Select the
        # DA3 fp16 path from hardware capability, not from a room/dataset rule.
        capability_major, _ = torch.cuda.get_device_capability()
        if capability_major < 8:
            autocast_override = "float16"
            print(
                "[da3] automatic Turing/older GPU guard: use float16 attention",
                flush=True,
            )
    if autocast_override in {"float16", "fp16", "half"}:
        torch.cuda.is_bf16_supported = lambda *a, **k: False  # type: ignore[assignment]
        print("[da3] autocast override: float16 (bf16 reported unsupported)", flush=True)
    elif autocast_override:
        raise ValueError(f"Unsupported DA3_AUTOCAST_DTYPE={autocast_override!r}; use float16 or unset")

    from depth_anything_3.api import DepthAnything3

    # Optional: offload retained backbone features to pinned host memory and
    # re-upload DPT head chunks (see scripts/da3_lowmem_patch.py). Needed on
    # 11GB GPUs for >~70 views at process_res 504; math is unchanged.
    lowmem_setting = os.environ.get("DA3_LOWMEM_OFFLOAD", "auto").strip().lower()
    if lowmem_setting not in {"0", "1", "auto"}:
        raise ValueError("DA3_LOWMEM_OFFLOAD must be 0, 1, or auto")
    if lowmem_setting == "auto" and torch.cuda.is_available():
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        lowmem_offload = bool(len(images) >= 32 and total_gib <= 12.5)
        if lowmem_offload:
            print(
                f"[da3] automatic low-memory guard: {len(images)} views on {total_gib:.1f} GiB",
                flush=True,
            )
    else:
        lowmem_offload = lowmem_setting == "1"
    if lowmem_offload:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import da3_lowmem_patch

        da3_lowmem_patch.apply()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[da3] device={device}", flush=True)
    print(f"[da3] model={args.model_dir}", flush=True)
    print(f"[da3] images={len(images)}", flush=True)

    model = DepthAnything3.from_pretrained(
        args.model_dir,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        local_files_only=args.local_files_only,
        force_download=args.force_download,
    )
    model = model.to(device=device)
    model.eval()

    inference_kwargs = {
        "image": images,
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "ref_view_strategy": args.ref_view_strategy,
    }
    if args.export_format:
        inference_kwargs["export_dir"] = str(args.out_dir)
        inference_kwargs["export_format"] = args.export_format
        inference_kwargs["conf_thresh_percentile"] = args.conf_thresh_percentile
        inference_kwargs["num_max_points"] = args.num_max_points
        inference_kwargs["show_cameras"] = not args.hide_cameras

    with torch.no_grad():
        prediction = model.inference(**inference_kwargs)

    np.save(args.out_dir / "depth.npy", prediction.depth)
    np.save(args.out_dir / "conf.npy", prediction.conf)
    np.save(args.out_dir / "extrinsics.npy", prediction.extrinsics)
    np.save(args.out_dir / "intrinsics.npy", prediction.intrinsics)
    if getattr(prediction, "processed_images", None) is not None:
        np.save(args.out_dir / "processed_images.npy", prediction.processed_images)
        preview_dir = args.out_dir / "processed_images"
        preview_dir.mkdir(exist_ok=True)
        for idx, arr in enumerate(prediction.processed_images):
            Image.fromarray(arr.astype(np.uint8)).save(preview_dir / f"{idx:06d}_{Path(images[idx]).stem}.png")

    meta = {
        "model_dir": args.model_dir,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "local_files_only": bool(args.local_files_only),
        "force_download": bool(args.force_download),
        "image_dir": str(args.image_dir),
        "images": images,
        "image_names": [Path(p).name for p in images],
        "num_images": len(images),
        "device": str(device),
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "ref_view_strategy": args.ref_view_strategy,
        "export_format": args.export_format,
        "conf_thresh_percentile": args.conf_thresh_percentile,
        "num_max_points": args.num_max_points,
        "show_cameras": not args.hide_cameras,
        "seed": args.seed,
        "autocast_dtype_override": autocast_override or None,
        "lowmem_offload_patch": bool(lowmem_offload),
        "lowmem_offload_setting": lowmem_setting,
        "extrinsics_convention": "w2c",
        "intrinsics_coordinate_space": "processed_images",
        "depth_shape": list(prediction.depth.shape),
        "conf_shape": list(prediction.conf.shape) if prediction.conf is not None else None,
        "extrinsics_shape": list(prediction.extrinsics.shape) if prediction.extrinsics is not None else None,
        "intrinsics_shape": list(prediction.intrinsics.shape) if prediction.intrinsics is not None else None,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "depth": str(args.out_dir / "depth.npy")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
