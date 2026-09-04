#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
from omegaconf import OmegaConf
from torchvision.transforms import v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chord inference with tiled VAE support.")
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
    parser.add_argument(
        "--disable-tiled-vae",
        action="store_true",
        help="Keep the original direct VAE encode/decode path.",
    )
    parser.add_argument(
        "--basecolor-only",
        action="store_true",
        help="Run only the CHORD basecolor stage for very large atlas inputs.",
    )
    parser.add_argument(
        "--roughness-grid-chunk-size",
        type=int,
        default=25,
        help=(
            "Exact CHORD roughness/metallic grid-search chunk size. Values below 25 use a "
            "running-min implementation with identical candidates and lower peak VRAM."
        ),
    )
    return parser.parse_args()


def image_files(indir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(indir.rglob(ext))
    return sorted(files)


def enable_tiled_vae(model) -> None:
    sd = model.model.sd
    vae = sd.vae
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()

    def encode_imgs_deterministic_tiled(self, imgs):
        if imgs.shape[1] == 1:
            imgs = v2.functional.grayscale_to_rgb(imgs)
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        if hasattr(posterior, "mode"):
            latents = posterior.mode()
        else:
            latents = posterior.mean
        return latents * self.vae.config.scaling_factor

    sd.encode_imgs_deterministic = types.MethodType(encode_imgs_deterministic_tiled, sd)


def keep_basecolor_stage_only(model) -> None:
    chain = model.model.chain
    if "basecolor" not in chain:
        raise KeyError("CHORD chain does not contain a basecolor stage")
    model.model.chain = {"basecolor": chain["basecolor"]}
    print("[INFO] restricted CHORD chain to basecolor stage")


def enable_low_memory_exact_roughness_grid(model, chunk_size: int) -> None:
    """Reduce CHORD's grid-search memory without changing its candidate values."""
    if int(chunk_size) >= 25:
        return
    from chord.module.chord import find_light_dir
    from chord.util import get_positions, srgb_to_rgb

    def compute_approx_roughness_metallic(self, render, maps, seperate=False, light=None):
        render_linear = srgb_to_rgb(render)
        _, _, height, width = render_linear.shape
        active_light = find_light_dir(maps["approxIrr"], self.prior_light) if light is None else light
        positions = get_positions(height, width, 10).to(self.device)
        cameras = torch.tensor([0, 0, 10.0]).to(self.device)
        roughness_samples = torch.arange(
            25, 225 + self.roughness_step, self.roughness_step
        ) / 255
        metallic_samples = torch.arange(0.0, 1.0 + self.metallic_step, self.metallic_step)
        roughness_values = roughness_samples[:, None].repeat(1, len(metallic_samples)).reshape(-1)
        metallic_values = metallic_samples[None].repeat(len(roughness_samples), 1).reshape(-1)

        grid_maps = {
            "basecolor": maps["basecolor"][None].permute(0, 1, 3, 4, 2),
            "normal": maps["normal"][None].permute(0, 1, 3, 4, 2),
        }
        best_loss = None
        best_roughness = None
        best_metallic = None
        for start in range(0, len(roughness_values), int(chunk_size)):
            stop = min(len(roughness_values), start + int(chunk_size))
            current_roughness = roughness_values[start:stop].reshape(-1, 1, 1, 1, 1).to(render_linear)
            current_metallic = metallic_values[start:stop].reshape(-1, 1, 1, 1, 1).to(render_linear)
            grid_maps["roughness"] = current_roughness
            grid_maps["metallic"] = current_metallic
            rendered = self.compute_render(grid_maps, cameras, positions, active_light)
            loss = (render_linear[None].permute(0, 1, 3, 4, 2) - rendered).abs().sum(-1)
            chunk_loss, chunk_index = torch.min(loss, dim=0)
            chunk_roughness = roughness_values[start:stop].to(render_linear)[chunk_index]
            chunk_metallic = metallic_values[start:stop].to(render_linear)[chunk_index]
            if best_loss is None:
                best_loss = chunk_loss
                best_roughness = chunk_roughness
                best_metallic = chunk_metallic
            else:
                replace = chunk_loss < best_loss
                best_loss = torch.where(replace, chunk_loss, best_loss)
                best_roughness = torch.where(replace, chunk_roughness, best_roughness)
                best_metallic = torch.where(replace, chunk_metallic, best_metallic)
            del rendered, loss, chunk_loss, chunk_index, chunk_roughness, chunk_metallic
            torch.cuda.empty_cache()
        roughness = best_roughness.unsqueeze(1)
        metallic = best_metallic.unsqueeze(1)
        if seperate:
            return roughness, metallic
        return torch.cat([roughness, metallic, torch.zeros_like(roughness)], dim=1)

    model.model.compute_approxRouMet = types.MethodType(
        compute_approx_roughness_metallic, model.model
    )
    print(
        "[INFO] enabled exact low-memory roughness/metallic grid search "
        f"with chunk_size={int(chunk_size)}"
    )


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
    if not args.disable_tiled_vae:
        enable_tiled_vae(model)
        print("[INFO] enabled tiled VAE encode/decode path")
    if args.basecolor_only:
        keep_basecolor_stage_only(model)
    else:
        enable_low_memory_exact_roughness_grid(model, args.roughness_grid_chunk_size)

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
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    raise SystemExit(main())
