#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
from omegaconf import OmegaConf
from torchvision.transforms import v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chord inference from a local checkpoint.")
    parser.add_argument("--chord-repo", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Square inference size. Use 0 to preserve native input scale and pad only.",
    )
    parser.add_argument("--pad-multiple", type=int, default=32)
    return parser.parse_args()


def image_files(indir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(indir.rglob(ext))
    return sorted(files)


def main() -> int:
    args = parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)
    if not args.chord_repo.exists():
        raise FileNotFoundError(args.chord_repo)

    sys.path.insert(0, str(args.chord_repo))
    from chord import ChordModel  # noqa: WPS433
    from chord.io import load_torch_file, read_image, save_maps  # noqa: WPS433

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    config = OmegaConf.load(args.config_path)
    model = ChordModel(config)
    print(f"[INFO] Loading Chord model from local ckpt: {args.ckpt}")
    state_dict = load_torch_file(str(args.ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = image_files(args.input_dir)
    print(f"[INFO] found {len(files)} images in {args.input_dir}")
    print(f"[INFO] saving results to {args.output_dir}")

    for image_file in tqdm.tqdm(files, desc="[INFO] processing images"):
        image = read_image(str(image_file)).to(device)
        ori_h, ori_w = image.shape[-2:]
        if args.size > 0:
            x = v2.Resize(size=(args.size, args.size), antialias=True)(image).unsqueeze(0)
            pad_h = pad_w = 0
        else:
            multiple = max(1, int(args.pad_multiple))
            pad_h = (-ori_h) % multiple
            pad_w = (-ori_w) % multiple
            x = image.unsqueeze(0)
            if pad_h or pad_w:
                pad_mode = "reflect" if ori_h > pad_h and ori_w > pad_w else "replicate"
                x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)
        amp_ctx = torch.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
        with torch.no_grad(), amp_ctx:
            output = model(x)
        for key in list(output.keys()):
            if args.size > 0:
                output[key] = v2.Resize(size=(ori_h, ori_w), antialias=True)(output[key])
            else:
                output[key] = output[key][..., :ori_h, :ori_w]
        output["input"] = image
        if "metalness" in output and "metallic" not in output:
            output["metallic"] = output["metalness"]
        save_maps(str(args.output_dir / image_file.stem), output)

    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
