import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from complete_room_textures_full import (  # noqa: E402
    FACE_NAMES,
    apply_face_material_identity,
    build_material_prior,
    compute_quality_masks,
    edge_energy,
    find_best_tile,
    load_rgb,
    make_seamless_tile,
    match_lab_statistics,
    normalize_tile_lighting,
    save_mask,
    save_rgb,
    source_image_path,
    tile_image,
)


PROMPTS = {
    "floor": (
        "top down seamless PBR albedo colormap of realistic indoor light wood floor material, "
        "natural wood grain, subtle plank variation, clean empty room surface, no furniture, no shadows, colormap"
    ),
    "ceiling": (
        "top down seamless PBR albedo colormap of plain off white indoor ceiling paint material, "
        "subtle plaster texture, clean empty room surface, no lamps, no shadows, colormap"
    ),
    "wall_00": (
        "top down seamless PBR albedo colormap of plain painted indoor wall material, "
        "subtle plaster texture, clean empty room wall, no furniture, no poster, no window, no shadows, colormap"
    ),
    "wall_01": (
        "top down seamless PBR albedo colormap of plain painted indoor wall material, "
        "subtle plaster texture, clean empty room wall, no furniture, no poster, no window, no shadows, colormap"
    ),
    "wall_02": (
        "top down seamless PBR albedo colormap of plain painted indoor wall material, "
        "subtle plaster texture, clean empty room wall, no furniture, no poster, no window, no shadows, colormap"
    ),
    "wall_03": (
        "top down seamless PBR albedo colormap of plain painted indoor wall material, "
        "subtle plaster texture, clean empty room wall, no furniture, no poster, no window, no shadows, colormap"
    ),
}

NEGATIVE_PROMPT = (
    "furniture, chair, table, cabinet, bookshelf, curtain, bed, person, object, clutter, "
    "poster, picture frame, text, logo, watermark, strong shadow, lighting gradient, perspective view"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-face material priors from the highest-confidence atlas regions. "
            "The generated prior is used only as a texture/material candidate for low-confidence "
            "and unseen texels; it is not allowed to overwrite high-confidence observations."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed-texture", choices=["raw", "texture"], default="raw")
    parser.add_argument("--backend", choices=["patch", "sdxl_lora", "material_anything"], default="patch")
    parser.add_argument("--faces", nargs="*", default=FACE_NAMES)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-stride", type=int, default=16)
    parser.add_argument("--floor-min-source-fraction", type=float, default=0.88)
    parser.add_argument("--ceiling-min-source-fraction", type=float, default=0.78)
    parser.add_argument("--wall-min-source-fraction", type=float, default=0.70)
    parser.add_argument("--surface-contaminant-mask-dir", type=Path, default=None)
    parser.add_argument("--surface-contaminant-dilate-px", type=int, default=8)
    parser.add_argument("--keep-valid-views", type=int, default=3)
    parser.add_argument("--candidate-valid-views", type=int, default=2)
    parser.add_argument("--min-valid-views", type=int, default=1)
    parser.add_argument("--keep-clean-thresh", type=float, default=0.58)
    parser.add_argument("--source-clean-thresh", type=float, default=0.43)
    parser.add_argument("--fill-clean-thresh", type=float, default=0.16)
    parser.add_argument("--keep-object-risk-thresh", type=float, default=0.48)
    parser.add_argument("--source-object-risk-thresh", type=float, default=0.66)
    parser.add_argument("--fill-object-risk-thresh", type=float, default=0.82)
    parser.add_argument("--mask-boundary-keep-thresh", type=float, default=0.46)
    parser.add_argument("--footprint-keep-min", type=float, default=0.12)
    parser.add_argument("--sdxl-model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--sdxl-lora-id", default="dog-god/texture-synthesis-sdxl-lora")
    parser.add_argument("--sdxl-lora-weight", default="texture-synthesis-topdown-base-condensed.safetensors")
    parser.add_argument("--sdxl-lora-scale", type=float, default=0.78)
    parser.add_argument("--material-anything-dir", type=Path, default=Path("third_party/MaterialAnything"))
    parser.add_argument(
        "--material-anything-estimator",
        type=Path,
        default=Path("models/materialanything/material_estimator"),
    )
    parser.add_argument("--material-anything-size", type=int, default=512)
    parser.add_argument("--material-anything-steps", type=int, default=32)
    parser.add_argument("--material-anything-keep-thresh", type=float, default=0.62)
    parser.add_argument("--material-prior-quality-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--material-prior-max-lab-delta", type=float, default=12.0)
    parser.add_argument("--material-prior-max-edge-ratio", type=float, default=1.55)
    parser.add_argument("--material-prior-max-sat-ratio", type=float, default=1.12)
    parser.add_argument("--face-material-outlier-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--face-material-source-reliability", type=float, default=0.44)
    parser.add_argument("--face-material-inlier-percentile", type=float, default=90.0)
    parser.add_argument("--face-material-outlier-dilate-px", type=int, default=2)
    parser.add_argument("--face-material-outlier-max-ratio", type=float, default=0.34)
    parser.add_argument(
        "--material-anything-wall-mode",
        choices=["stat_prior", "observed", "projected_mask"],
        default="stat_prior",
        help=(
            "For walls, floors, and ceilings, stat_prior uses the selected high-confidence material tile "
            "as the condition and does not preserve scattered observed texels directly; observed "
            "preserves robust observed texels directly; projected_mask uses the whole projected structure "
            "mask after object/contaminant rejection rather than only high-confidence material texels."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=22)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=0.36)
    parser.add_argument("--floor-strength", type=float, default=None)
    parser.add_argument("--ceiling-strength", type=float, default=None)
    parser.add_argument("--wall-strength", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--allow-fallback", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def inpaint_tile_holes(tile, source_mask):
    if tile.size == 0 or np.all(source_mask):
        return tile
    mask = (~source_mask).astype(np.uint8) * 255
    if np.count_nonzero(mask) == 0:
        return tile
    tile_u8 = np.clip(tile * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(tile_u8, cv2.COLOR_RGB2BGR)
    fixed = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def prepare_source_tile(face, image, masks, args):
    if face == "floor":
        min_source = args.floor_min_source_fraction
    elif face == "ceiling":
        min_source = args.ceiling_min_source_fraction
    else:
        min_source = args.wall_min_source_fraction
    tile, tile_box = find_best_tile(args, face, image, masks, args.tile_size, min_source)
    y, x, size = tile_box
    source_mask = masks["source"][y : y + size, x : x + size]
    tile = inpaint_tile_holes(tile, source_mask)
    if face in {"floor", "ceiling"}:
        tile = normalize_tile_lighting(tile)
    else:
        # Wall observations often contain missed furniture, curtains, posters, or trim.
        # Use the reliable region only for chroma/low-frequency guidance; the material
        # model should not learn high-frequency object patterns as wall texture.
        edge = cv2.Sobel(
            cv2.cvtColor(np.clip(tile * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )
        edge = np.abs(edge) / 255.0
        reliable = source_mask & (edge <= max(0.035, float(np.percentile(edge[source_mask], 55.0)) if np.any(source_mask) else 0.035))
        if np.count_nonzero(reliable) < 64:
            reliable = source_mask
        if np.count_nonzero(reliable) >= 64:
            color = np.median(tile[reliable], axis=0).reshape(1, 1, 3)
        else:
            color = np.median(tile.reshape(-1, 3), axis=0).reshape(1, 1, 3)
        smooth = np.ones_like(tile) * color
        subtle = cv2.GaussianBlur(tile - cv2.GaussianBlur(tile, (0, 0), max(3.0, size / 18.0)), (0, 0), 2.0)
        tile = np.clip(smooth + 0.035 * subtle, 0.0, 1.0)
    return make_seamless_tile(tile), tile_box


def load_sdxl_pipe(args):
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.sdxl_model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.load_lora_weights(
        args.sdxl_lora_id,
        weight_name=args.sdxl_lora_weight,
        adapter_name="material_texture",
    )
    if hasattr(pipe, "set_adapters"):
        pipe.set_adapters(["material_texture"], adapter_weights=[args.sdxl_lora_scale])
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe, torch


def load_material_anything_pipe(args):
    import torch
    import sys

    ma_dir = str(args.material_anything_dir)
    if ma_dir not in sys.path:
        sys.path.insert(0, ma_dir)
    from pipelines.pipeline_stable_diffusion_switcher import StableDiffusionPipeline as MaterialEstimator
    from models.scheduling_ddpm import DDPMScheduler

    pipe = MaterialEstimator.from_pretrained(
        str(args.material_anything_estimator),
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDPMScheduler.from_pretrained(str(args.material_anything_estimator), subfolder="scheduler")
    pipe = pipe.to(args.device)
    for method_name in ("enable_vae_slicing", "enable_vae_tiling", "enable_attention_slicing"):
        if hasattr(pipe, method_name):
            getattr(pipe, method_name)()
    return pipe, torch


def face_strength(face, args):
    if face == "floor" and args.floor_strength is not None:
        return args.floor_strength
    if face == "ceiling" and args.ceiling_strength is not None:
        return args.ceiling_strength
    if face.startswith("wall") and args.wall_strength is not None:
        return args.wall_strength
    return args.strength


def sdxl_material_tile(pipe, torch, tile, face, args, seed):
    init = Image.fromarray(np.clip(tile * 255.0, 0, 255).astype(np.uint8)).convert("RGB")
    generator = torch.Generator(device=args.device).manual_seed(seed)
    result = pipe(
        prompt=PROMPTS[face],
        negative_prompt=NEGATIVE_PROMPT,
        image=init,
        strength=face_strength(face, args),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    generated = np.asarray(result.convert("RGB"), dtype=np.float32) / 255.0
    if generated.shape[:2] != tile.shape[:2]:
        generated = cv2.resize(generated, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_AREA)
    return make_seamless_tile(generated)


def resize_rgb(image, size, interpolation=cv2.INTER_AREA):
    return cv2.resize(image, (size, size), interpolation=interpolation)


def robust_keep_mask(face, image, masks, args):
    edge = cv2.GaussianBlur(
        cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY),
        (0, 0),
        0.8,
    )
    gx = cv2.Sobel(edge, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(edge, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.sqrt(gx * gx + gy * gy) / 255.0
    src_edge = edge_energy[masks["source"]]
    edge_cap = float(np.percentile(src_edge, 72.0)) if src_edge.size else 0.10
    base = (
        masks["source"]
        & (masks["reliability"] >= args.material_anything_keep_thresh)
        & (masks["object_risk"] <= 0.30)
        & (masks["mask_boundary_trust"] >= 0.58)
        & (~masks["contaminant"])
    )
    if face.startswith("wall"):
        base &= edge_energy <= max(0.08, edge_cap)
    elif face == "floor":
        base = (
            (masks["source"] | masks["high"])
            & (masks["reliability"] >= max(0.40, args.material_anything_keep_thresh - 0.18))
            & (masks["object_risk"] <= 0.46)
            & (masks["mask_boundary_trust"] >= 0.42)
            & (~masks["contaminant"])
        )
    if np.count_nonzero(base) < 0.01 * base.size:
        base = masks["source"] & (~masks["contaminant"])
    return base


def material_anything_prior(pipe, torch, face, image, masks, patch_prior, args, seed):
    size = args.material_anything_size
    keep = robust_keep_mask(face, image, masks, args)
    if args.material_anything_wall_mode == "projected_mask":
        keep = (
            masks["observed"]
            & (masks["clean_score"] >= args.fill_clean_thresh)
            & (masks["object_risk"] <= args.fill_object_risk_thresh)
            & (masks["mask_boundary_trust"] >= 0.12)
            & (~masks["contaminant"])
        )
        keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    elif (
        face.startswith("wall") or face in {"floor", "ceiling"}
    ) and args.material_anything_wall_mode == "stat_prior":
        keep = np.zeros_like(keep, dtype=bool)
    keep_f = keep.astype(np.float32)
    keep_soft = cv2.GaussianBlur(keep_f, (0, 0), 2.0)
    keep_soft = np.clip(keep_soft, 0.0, 1.0)
    cond = image * keep_soft[..., None] + patch_prior * (1.0 - keep_soft[..., None])
    if face.startswith("wall"):
        # The estimator should see a clean wall context, not missed furniture texture.
        cond = cv2.GaussianBlur(cond, (0, 0), 1.1)
    cond_small = resize_rgb(cond, size)
    prior_small = resize_rgb(patch_prior, size)
    keep_small = cv2.resize(keep_f, (size, size), interpolation=cv2.INTER_NEAREST)
    if face == "floor":
        normal_color = (128, 128, 255)
    elif face == "ceiling":
        normal_color = (128, 128, 255)
    else:
        normal_color = (128, 128, 255)
    normal = Image.new("RGB", (size, size), normal_color)

    cond_pil = Image.fromarray(np.clip(cond_small * 255.0, 0, 255).astype(np.uint8)).convert("RGB")
    init_albedo = torch.from_numpy(prior_small[None].astype(np.float32))
    rm = np.ones_like(prior_small, dtype=np.float32)
    bump = np.ones_like(prior_small, dtype=np.float32) * np.array([0.5, 0.5, 1.0], dtype=np.float32)
    init_materials = {
        "albedo": init_albedo,
        "roughness_metallic": torch.from_numpy(rm[None].astype(np.float32)),
        "bump": torch.from_numpy(bump[None].astype(np.float32)),
    }
    keep_tensor = torch.from_numpy(keep_small[None].astype(np.float32)).to(args.device)
    generator = torch.Generator(device=args.device).manual_seed(seed)
    output = pipe(
        prompt=[""],
        cond_image=[cond_pil],
        normal_image=[normal],
        init_materials=init_materials,
        masks=keep_tensor,
        num_inference_steps=args.material_anything_steps,
        guidance_scale=1.0,
        generator=generator,
        height=size,
        width=size,
    ).images
    albedo = np.asarray(output[0].convert("RGB"), dtype=np.float32) / 255.0
    albedo = cv2.resize(albedo, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    if face == "floor":
        albedo = make_seamless_tile(albedo, max(12, min(albedo.shape[:2]) // 28))
    elif face.startswith("wall"):
        albedo = cv2.GaussianBlur(albedo, (0, 0), 0.65)
    albedo = match_lab_statistics(
        albedo,
        image,
        keep,
        max_std_ratio=1.80 if face == "floor" else 1.25,
    )
    return np.clip(albedo, 0.0, 1.0), keep


def prior_quality_metrics(generated, patch_prior):
    generated_u8 = np.clip(generated * 255.0, 0, 255).astype(np.uint8)
    patch_u8 = np.clip(patch_prior * 255.0, 0, 255).astype(np.uint8)
    gen_lab = cv2.cvtColor(generated_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    patch_lab = cv2.cvtColor(patch_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_delta = np.sqrt(np.sum((gen_lab - patch_lab) ** 2, axis=-1))
    gen_edge = edge_energy(generated)
    patch_edge = edge_energy(patch_prior)
    gen_hsv = cv2.cvtColor(generated_u8, cv2.COLOR_RGB2HSV)
    patch_hsv = cv2.cvtColor(patch_u8, cv2.COLOR_RGB2HSV)
    gen_sat = gen_hsv[..., 1].astype(np.float32) / 255.0
    patch_sat = patch_hsv[..., 1].astype(np.float32) / 255.0
    return {
        "lab_delta_mean": float(np.mean(lab_delta)),
        "lab_delta_p95": float(np.percentile(lab_delta, 95.0)),
        "edge_mean": float(np.mean(gen_edge)),
        "patch_edge_mean": float(np.mean(patch_edge)),
        "edge_ratio": float(np.mean(gen_edge) / (np.mean(patch_edge) + 1e-6)),
        "sat_mean": float(np.mean(gen_sat)),
        "patch_sat_mean": float(np.mean(patch_sat)),
        "sat_ratio": float(np.mean(gen_sat) / (np.mean(patch_sat) + 1e-6)),
    }


def reject_material_prior(face, generated, patch_prior, args):
    if not args.material_prior_quality_gate:
        return None, prior_quality_metrics(generated, patch_prior)
    metrics = prior_quality_metrics(generated, patch_prior)
    reasons = []
    lab_delta_limit = args.material_prior_max_lab_delta
    edge_ratio_limit = args.material_prior_max_edge_ratio
    sat_ratio_limit = args.material_prior_max_sat_ratio
    if face.startswith("wall"):
        lab_delta_limit += 2.0
        sat_ratio_limit += 0.08
    elif face == "ceiling":
        lab_delta_limit += 4.0
        sat_ratio_limit += 0.10
    if metrics["lab_delta_mean"] > lab_delta_limit:
        reasons.append(f"lab_delta_mean={metrics['lab_delta_mean']:.2f}>{lab_delta_limit:.2f}")
    if (
        metrics["edge_ratio"] > edge_ratio_limit
        and metrics["edge_mean"] > metrics["patch_edge_mean"] + 0.0012
    ):
        reasons.append(f"edge_ratio={metrics['edge_ratio']:.2f}>{edge_ratio_limit:.2f}")
    if (
        metrics["sat_ratio"] > sat_ratio_limit
        and metrics["sat_mean"] > metrics["patch_sat_mean"] + 0.030
    ):
        reasons.append(f"sat_ratio={metrics['sat_ratio']:.2f}>{sat_ratio_limit:.2f}")
    if reasons:
        return "; ".join(reasons), metrics
    return None, metrics


def build_full_prior(face, image, masks, material_tile, tile_box):
    y, x, _ = tile_box
    prior = tile_image(material_tile, image.shape[:2], offset_y=y, offset_x=x)
    target_mask = masks.get("material_ref", masks["high"] | masks["source"])
    if face.startswith("wall"):
        prior = match_lab_statistics(prior, image, target_mask, max_std_ratio=1.15)
    elif face == "ceiling":
        prior = match_lab_statistics(prior, image, target_mask, max_std_ratio=1.10)
    else:
        prior = match_lab_statistics(prior, image, target_mask, max_std_ratio=1.65)
    if face == "ceiling":
        smooth = cv2.GaussianBlur(prior, (0, 0), 8.0)
        prior = np.clip(0.75 * smooth + 0.25 * prior, 0.0, 1.0)
    return np.clip(prior, 0.0, 1.0)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tile_dir = args.out_dir / "tiles"
    prior_dir = args.out_dir / "priors"
    debug_dir = args.out_dir / "debug"
    tile_dir.mkdir(parents=True, exist_ok=True)
    prior_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    pipe = torch = None
    init_error = None
    if args.backend == "sdxl_lora":
        try:
            pipe, torch = load_sdxl_pipe(args)
        except Exception as exc:
            if not args.allow_fallback:
                raise
            init_error = repr(exc)
            print(f"[warn] SDXL material LoRA failed to initialize; using patch backend: {init_error}")
            args.backend = "patch"
    elif args.backend == "material_anything":
        try:
            pipe, torch = load_material_anything_pipe(args)
        except Exception as exc:
            if not args.allow_fallback:
                raise
            init_error = repr(exc)
            print(f"[warn] Material Anything failed to initialize; using patch backend: {init_error}")
            args.backend = "patch"

    stats = []
    for face_i, face in enumerate(args.faces):
        image = load_rgb(source_image_path(args.input_dir, face, args.seed_texture))
        masks = compute_quality_masks(args, face, image.shape[:2])
        masks = apply_face_material_identity(args, face, image, masks)
        ref_tile, tile_box = prepare_source_tile(face, image, masks, args)
        patch_prior = build_full_prior(face, image, masks, ref_tile, tile_box)
        generated_tile = ref_tile
        prior = patch_prior
        method = "patch"
        error = None
        quality = None
        if args.backend == "sdxl_lora":
            try:
                generated_tile = sdxl_material_tile(pipe, torch, ref_tile, face, args, args.seed + face_i * 137)
                method = "sdxl_lora"
            except Exception as exc:
                if not args.allow_fallback:
                    raise
                error = repr(exc)
                print(f"[warn] {face}: SDXL material generation failed; using patch prior: {error}")
                generated_tile = ref_tile
                method = "patch_fallback"
            prior = build_full_prior(face, image, masks, generated_tile, tile_box)
        elif args.backend == "material_anything":
            try:
                prior, keep_mask = material_anything_prior(
                    pipe, torch, face, image, masks, patch_prior, args, args.seed + face_i * 137
                )
                rejection, quality = reject_material_prior(face, prior, patch_prior, args)
                if rejection is not None:
                    print(f"[material-gate] {face}: rejecting Material Anything prior: {rejection}")
                    prior = patch_prior
                    generated_tile = ref_tile
                    method = "material_anything_rejected_to_patch"
                    error = rejection
                else:
                    generated_tile = cv2.resize(
                        prior, (ref_tile.shape[1], ref_tile.shape[0]), interpolation=cv2.INTER_AREA
                    )
                    method = "material_anything"
                save_mask(debug_dir / f"{face}_ma_keep_mask.png", keep_mask)
            except Exception as exc:
                if not args.allow_fallback:
                    raise
                error = repr(exc)
                print(f"[warn] {face}: Material Anything generation failed; using patch prior: {error}")
                generated_tile = ref_tile
                prior = patch_prior
                method = "patch_fallback"
        save_rgb(tile_dir / f"{face}.png", generated_tile)
        save_rgb(prior_dir / f"{face}.png", prior)
        save_mask(debug_dir / f"{face}_source_mask.png", masks["source"])
        save_mask(debug_dir / f"{face}_high_mask.png", masks["high"])
        save_mask(debug_dir / f"{face}_reliability.png", masks["reliability"])
        save_mask(debug_dir / f"{face}_face_material_outlier_mask.png", masks["face_material_outlier"])
        stats.append(
            {
                "face": face,
                "method": method,
                "tile_box_yx_size": [int(tile_box[0]), int(tile_box[1]), int(tile_box[2])],
                "high_texels": int(np.count_nonzero(masks["high"])),
                "source_texels": int(np.count_nonzero(masks["source"])),
                "face_material_outlier_texels": int(np.count_nonzero(masks["face_material_outlier"])),
                "mean_reliability_source": float(np.mean(masks["reliability"][masks["source"]]))
                if np.any(masks["source"])
                else 0.0,
                "material_prior_quality": quality,
                "error": error,
            }
        )
        print(f"[material] {face}: {method}, tile={stats[-1]['tile_box_yx_size']}")

    with open(args.out_dir / "metadata_material_priors.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_export_dir": str(args.input_dir),
                "backend": args.backend,
                "init_error": init_error,
                "parameters": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                    if key not in {"input_dir", "out_dir"}
                },
                "stats": stats,
            },
            f,
            indent=2,
        )
    print("[done] material priors:", prior_dir)


if __name__ == "__main__":
    main()
