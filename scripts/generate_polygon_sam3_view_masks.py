#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


OBJECT_PROMPTS = [
    "desk",
    "table",
    "chair",
    "cabinet",
    "filing cabinet",
    "drawer cabinet",
    "pedestal cabinet",
    "bookshelf",
    "bookcase",
    "shelf",
    "bag",
    "clothes",
    "computer monitor",
    "keyboard",
    "mouse",
    "tripod",
    "printer",
    "office printer",
    "photocopier",
    "copy machine",
    "multifunction printer",
    "scanner",
    "electronic equipment",
    "office equipment",
    "machine",
    "appliance",
    "air conditioner",
    "wall mounted air conditioner",
    "heater",
    "radiator",
    "speaker",
    "fan",
    "ceiling light",
    "ceiling lamp",
    "light fixture",
    "chandelier",
    "pendant light",
    "curtain rod",
    "curtain track",
    "curtain rail",
    "trash bin",
    "waste bin",
    "box",
    "storage box",
]

SURFACE_PROMPT_GROUPS = {
    "whiteboard": [
        "whiteboard",
        "white board",
        "large whiteboard",
        "wall-mounted whiteboard",
        "wall mounted whiteboard",
        "marker board",
        "writing board",
        "dry erase board",
        "board with writing",
        "whiteboard with writing",
        "marker tray",
        "blackboard",
        "chalkboard",
        "green board",
        "classroom board",
    ],
    "wall_art": [
        "painting",
        "framed painting",
        "canvas painting",
        "picture",
        "framed picture",
        "wall-mounted picture",
        "picture frame",
        "poster",
        "poster on wall",
        "wall art",
        "artwork",
        "framed artwork",
        "wall-mounted artwork",
        "wall mounted artwork",
        "wall decoration",
        "decorative wall picture",
    ],
    "wall_sign": [
        "notice",
        "notice board",
        "paper",
        "sheet of paper",
        "paper on wall",
        "sign",
        "label",
        "calendar",
        "sticker",
        "wall sticker",
        "tape on wall",
    ],
}

ARCHITECTURAL_CUTOUT_PROMPT_GROUPS = {
    "door_core": [
        "doorway",
        "door frame",
        "open door",
        "room door",
        "office door",
        "wooden door",
        "door panel",
        "door with sign",
        "door handle",
    ],
    "door_broad": [
        "door",
        "closed door",
        "entrance door",
    ],
    "window": [
        "window",
        "glass window",
        "window frame",
        "large window",
        "floor-to-ceiling window",
        "bay window",
    ],
}


def load_sam3(device: str, checkpoint_path: Path | None = None, confidence_threshold: float = 0.18):
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except Exception as exc:
        raise RuntimeError(
            "SAM3 is not installed. Install facebookresearch/sam3 in the active CUDA environment."
        ) from exc

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        load_from_HF=checkpoint_path is None,
    )
    model.eval()
    return Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def masks_to_numpy(masks, mask_thresh: float) -> np.ndarray:
    if masks is None:
        return np.zeros((0, 1, 1), dtype=bool)
    if isinstance(masks, torch.Tensor):
        arr = masks.detach().float().cpu().numpy()
    else:
        arr = np.asarray(masks)
    while arr.ndim > 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected SAM3 mask shape: {arr.shape}")
    if arr.dtype == np.bool_:
        return arr
    return arr > mask_thresh


def scores_to_numpy(scores, count: int) -> np.ndarray:
    if scores is None:
        return np.ones((count,), dtype=np.float32)
    if isinstance(scores, torch.Tensor):
        arr = scores.detach().float().cpu().numpy()
    else:
        arr = np.asarray(scores, dtype=np.float32)
    return arr.reshape(-1)[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SAM3 object/surface/cutout masks for polygon room views.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--colmap-model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--view-face-mask-dir", type=Path, default=None)
    parser.add_argument("--faces", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam3-checkpoint-path", type=Path, default=None)
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16", "none"], default="fp16")
    parser.add_argument("--score-thresh", type=float, default=0.12)
    parser.add_argument("--object-score-thresh", type=float, default=None)
    parser.add_argument("--surface-score-thresh", type=float, default=None)
    parser.add_argument("--cutout-score-thresh", type=float, default=None)
    parser.add_argument("--mask-thresh", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=36)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.50)
    parser.add_argument("--object-max-mask-area-ratio", type=float, default=None)
    parser.add_argument("--surface-max-mask-area-ratio", type=float, default=None)
    parser.add_argument("--cutout-max-mask-area-ratio", type=float, default=None)
    parser.add_argument("--min-structure-overlap", type=float, default=0.12)
    parser.add_argument("--surface-min-structure-overlap", type=float, default=None)
    parser.add_argument("--cutout-min-structure-overlap", type=float, default=None)
    parser.add_argument("--min-face-overlap", type=float, default=0.06)
    parser.add_argument("--surface-min-face-overlap", type=float, default=None)
    parser.add_argument("--cutout-min-face-overlap", type=float, default=None)
    parser.add_argument("--protect-dominant-structure-ratio", type=float, default=0.93)
    parser.add_argument("--protect-min-area-ratio", type=float, default=0.20)
    parser.add_argument(
        "--broad-covering-object-max-mask-area-ratio",
        type=float,
        default=0.98,
        help=(
            "Maximum image-area ratio for explicit broad removable surface coverings "
            "such as rugs and curtains. Ordinary object prompts still use the stricter "
            "object-max-mask-area-ratio."
        ),
    )
    parser.add_argument("--dilate-object-px", type=int, default=3)
    parser.add_argument("--view-face-stride", type=int, default=5)
    parser.add_argument("--view-limit", type=int, default=None)
    parser.add_argument("--view-ids", default=None)
    parser.add_argument("--view-start", type=int, default=0)
    parser.add_argument("--view-stride", type=int, default=1)
    parser.add_argument("--object-prompt", action="append", default=None)
    parser.add_argument("--surface-prompt", action="append", default=None)
    parser.add_argument("--cutout-prompt", action="append", default=None)
    parser.add_argument(
        "--disable-object-prompts",
        action="store_true",
        help="Disable movable-object prompting when the input views are already empty-room GT.",
    )
    parser.add_argument(
        "--disable-surface-prompts",
        action="store_true",
        help="Disable removable-surface prompting when the input views are already empty-room GT.",
    )
    parser.add_argument(
        "--disable-cutout-prompts",
        action="store_true",
        help="Disable architectural door/window cutout prompting for this run.",
    )
    parser.add_argument("--surface-color-mode", choices=["none", "dark_green_board"], default="none")
    parser.add_argument("--surface-color-close-px", type=int, default=7)
    parser.add_argument("--surface-color-dilate-px", type=int, default=2)
    parser.add_argument("--surface-priority-over-object", dest="surface_priority_over_object", action="store_true", default=True)
    parser.add_argument("--no-surface-priority-over-object", dest="surface_priority_over_object", action="store_false")
    parser.add_argument("--object-priority-over-cutout", dest="object_priority_over_cutout", action="store_true", default=True)
    parser.add_argument("--no-object-priority-over-cutout", dest="object_priority_over_cutout", action="store_false")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def image_keys(pose) -> list[str]:
    stem = Path(pose.name).stem
    return [f"view_{pose.image_id:03d}", f"view_{pose.image_id:06d}", stem]


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def load_face_mask_metadata(out_dir: Path, view_face_mask_dir: Path | None):
    face_dir = view_face_mask_dir or (out_dir / "view_face_masks")
    meta_path = face_dir.parent / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing precomputed polygon face-mask metadata: {meta_path}. "
            "Run generate_polygon_view_face_masks_from_colmap.py first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    poses = [
        SimpleNamespace(
            image_id=int(item["image_id"]),
            name=str(item["name"]),
            height=int(item["shape_hw"][0]),
            width=int(item["shape_hw"][1]),
        )
        for item in meta["views"]
    ]
    return face_dir, meta["faces"], meta.get("palette_rgb", {}), poses


def load_face_id(face_dir: Path, pose) -> np.ndarray:
    for key in image_keys(pose):
        path = face_dir / f"{key}_face_id.npy"
        if path.exists():
            return np.load(path).astype(np.uint16)
    raise FileNotFoundError(f"Missing face_id npy for {pose.name} in {face_dir}")


def jsonable_args(args: argparse.Namespace) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def autocast_context(device: str, amp_dtype: str):
    if device == "cuda" and amp_dtype == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    if device == "cuda" and amp_dtype == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def parse_faces(value: str | None, faces: list[str]) -> list[str]:
    if value is None:
        return [f for f in faces if f.startswith("wall_")]
    out = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in out if x not in faces]
    if unknown:
        raise ValueError(f"Unknown faces: {unknown}")
    return out


def parse_view_ids(value: str | None) -> set[int] | None:
    if value is None:
        return None
    ids = {int(item.strip()) for item in value.split(",") if item.strip()}
    return ids


def score_threshold(args: argparse.Namespace, kind: str) -> float:
    value = getattr(args, f"{kind}_score_thresh", None)
    return args.score_thresh if value is None else float(value)


def kind_threshold(args: argparse.Namespace, kind: str, name: str) -> float:
    value = getattr(args, f"{kind}_{name}", None)
    return float(getattr(args, name)) if value is None else float(value)


def structure_dominance(mask: np.ndarray, face_id: np.ndarray) -> tuple[float, int]:
    structure = face_id != 255
    if not np.any(mask):
        return 0.0, -1
    ids = face_id[mask & structure]
    if ids.size == 0:
        return 0.0, -1
    counts = np.bincount(ids.astype(np.int32), minlength=int(face_id[structure].max()) + 1)
    return float(counts.max() / max(1, int(mask.sum()))), int(counts.argmax())


def keep_surface_mask(
    mask: np.ndarray,
    face_id: np.ndarray,
    target_face_ids: np.ndarray,
    args: argparse.Namespace,
    kind: str,
):
    area = int(np.count_nonzero(mask))
    if area < args.min_mask_area:
        return False, 0.0, {}
    area_ratio = area / float(mask.size)
    max_mask_area_ratio = kind_threshold(args, kind, "max_mask_area_ratio")
    if area_ratio > max_mask_area_ratio:
        return False, 0.0, {}
    target = np.isin(face_id, target_face_ids)
    structure_pixels = mask & target
    structure_ratio = float(np.count_nonzero(structure_pixels) / max(area, 1))
    min_structure_overlap = kind_threshold(args, kind, "min_structure_overlap")
    if structure_ratio < min_structure_overlap:
        return False, structure_ratio, {}
    counts = {
        int(fid): int(np.count_nonzero(structure_pixels & (face_id == fid)))
        for fid in target_face_ids
    }
    dominant = max(counts.values()) / float(max(area, 1))
    min_face_overlap = kind_threshold(args, kind, "min_face_overlap")
    if dominant < min_face_overlap:
        return False, structure_ratio, counts
    return True, structure_ratio, counts


def resize_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape_hw:
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy()
    if np.any(mask):
        out[mask] = (0.45 * out[mask] + 0.55 * np.asarray(color, dtype=np.float32)).astype(np.uint8)
    return out


def dark_green_board_color_mask(rgb: np.ndarray, close_px: int, dilate_px: int) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.int32)
    s = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.int32)
    mask = (h >= 35) & (h <= 88) & (s >= 24) & (v <= 170)
    if close_px > 0 and np.any(mask):
        k = 2 * int(close_px) + 1
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1).astype(bool)
    if dilate_px > 0 and np.any(mask):
        k = 2 * int(dilate_px) + 1
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1).astype(bool)
    return mask


def save_per_view_masks(out_dir: Path, pose, masks: dict[str, np.ndarray]) -> None:
    for key in image_keys(pose):
        for suffix, mask in masks.items():
            save_mask(out_dir / f"{key}_{suffix}.png", mask)


def main() -> int:
    args = parse_args()
    if args.view_stride < 1:
        raise ValueError("--view-stride must be >= 1")
    if args.view_start < 0 or args.view_start >= args.view_stride:
        raise ValueError("--view-start must be in [0, view_stride)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out_dir / "overlays"
    face_dir = args.out_dir / "view_face_masks"
    overlay_dir.mkdir(exist_ok=True)
    face_dir.mkdir(exist_ok=True)

    face_dir, faces, palette, poses = load_face_mask_metadata(args.out_dir, args.view_face_mask_dir)
    view_ids = parse_view_ids(args.view_ids)
    if view_ids is not None:
        selected = [p for p in poses if p.image_id in view_ids]
    else:
        if args.view_limit is not None:
            poses = poses[: args.view_limit]
        selected = [p for i, p in enumerate(poses) if (i - args.view_start) % args.view_stride == 0]
    target_faces = parse_faces(None, faces)
    target_face_ids = np.asarray([faces.index(f) for f in target_faces], dtype=np.uint16)

    object_prompts = [] if args.disable_object_prompts else args.object_prompt or OBJECT_PROMPTS
    # Rugs and curtains are intentionally broad, single-structure-dominant
    # removable objects.  The generic structural-surface protection below
    # would otherwise reject their complete masks and keep only fragments,
    # contaminating the floor or wall material atlas.  Limit the exemption to
    # explicit covering prompts; ordinary objects retain the strict limits.
    broad_covering_object_prompts = {
        "area rug",
        "floor rug",
        "rug",
        "curtain",
        "drape",
    }
    surface_prompts = (
        []
        if args.disable_surface_prompts
        else args.surface_prompt
        or [p for prompts in SURFACE_PROMPT_GROUPS.values() for p in prompts]
    )
    cutout_prompts = (
        []
        if args.disable_cutout_prompts
        else args.cutout_prompt
        or [p for prompts in ARCHITECTURAL_CUTOUT_PROMPT_GROUPS.values() for p in prompts]
    )
    surface_prompt_to_group = {p: g for g, prompts in SURFACE_PROMPT_GROUPS.items() for p in prompts}
    cutout_prompt_to_group = {p: g for g, prompts in ARCHITECTURAL_CUTOUT_PROMPT_GROUPS.items() for p in prompts}

    print(f"[sam3] loading model on {args.device}", flush=True)
    processor = load_sam3(args.device, args.sam3_checkpoint_path, args.score_thresh)
    object_kernel = None
    if args.dilate_object_px > 0:
        object_kernel = np.ones((2 * args.dilate_object_px + 1, 2 * args.dilate_object_px + 1), dtype=np.uint8)

    all_stats = []
    for pose in selected:
        face_id = load_face_id(face_dir, pose)
        image_path = args.dataset_dir / "input_images" / pose.name
        if not image_path.exists():
            image_path = args.dataset_dir / "input_images" / Path(pose.name).name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing input image for registered COLMAP view: {pose.name}")
        image = Image.open(image_path).convert("RGB")
        if image.size != (pose.width, pose.height):
            image = image.resize((pose.width, pose.height), RESAMPLE_LANCZOS)
        rgb = np.asarray(image)
        object_mask = np.zeros((pose.height, pose.width), dtype=bool)
        surface_mask = np.zeros_like(object_mask)
        cutout_mask = np.zeros_like(object_mask)
        surface_groups = {g: np.zeros_like(object_mask) for g in SURFACE_PROMPT_GROUPS}
        cutout_groups = {g: np.zeros_like(object_mask) for g in ARCHITECTURAL_CUTOUT_PROMPT_GROUPS}
        prompt_stats = []

        with torch.inference_mode(), autocast_context(args.device, args.amp_dtype):
            state = processor.set_image(image)
            for kind, prompts in (("object", object_prompts), ("surface", surface_prompts), ("cutout", cutout_prompts)):
                for prompt in prompts:
                    output = processor.set_text_prompt(state=state, prompt=prompt)
                    masks = masks_to_numpy(output.get("masks"), args.mask_thresh)
                    scores = scores_to_numpy(output.get("scores"), masks.shape[0])
                    kept = 0
                    rejected = 0
                    for mask_i, mask in enumerate(masks):
                        mask = resize_mask(mask, (pose.height, pose.width))
                        area = int(np.count_nonzero(mask))
                        if area < args.min_mask_area:
                            rejected += 1
                            continue
                        score = float(scores[mask_i]) if mask_i < scores.size else 1.0
                        if score < score_threshold(args, kind):
                            rejected += 1
                            continue
                        if kind == "object":
                            area_ratio = area / float(mask.size)
                            max_object_area_ratio = kind_threshold(args, "object", "max_mask_area_ratio")
                            if prompt in broad_covering_object_prompts:
                                max_object_area_ratio = max(
                                    max_object_area_ratio,
                                    float(args.broad_covering_object_max_mask_area_ratio),
                                )
                            if area_ratio > max_object_area_ratio:
                                rejected += 1
                                continue
                            dominant_ratio, _ = structure_dominance(mask, face_id)
                            if (
                                prompt not in broad_covering_object_prompts
                                and
                                area_ratio >= args.protect_min_area_ratio
                                and dominant_ratio >= args.protect_dominant_structure_ratio
                            ):
                                rejected += 1
                                continue
                            object_mask |= mask
                        else:
                            keep, _, _ = keep_surface_mask(mask, face_id, target_face_ids, args, kind)
                            if not keep:
                                rejected += 1
                                continue
                            if kind == "surface":
                                group = surface_prompt_to_group.get(prompt, "custom_surface")
                                if group not in surface_groups:
                                    surface_groups[group] = np.zeros_like(object_mask)
                                surface_groups[group] |= mask
                                surface_mask |= mask
                            else:
                                group = cutout_prompt_to_group.get(prompt, "custom_cutout")
                                if group not in cutout_groups:
                                    cutout_groups[group] = np.zeros_like(object_mask)
                                cutout_groups[group] |= mask
                                cutout_mask |= mask
                        kept += 1
                    prompt_stats.append({"kind": kind, "prompt": prompt, "instances": int(masks.shape[0]), "kept": kept, "rejected": rejected})

        if args.surface_color_mode == "dark_green_board" and np.any(surface_mask):
            board_color_mask = dark_green_board_color_mask(rgb, args.surface_color_close_px, args.surface_color_dilate_px)
            surface_mask &= board_color_mask
            for group in list(surface_groups):
                surface_groups[group] &= board_color_mask

        if object_kernel is not None and np.any(object_mask):
            object_mask = cv2.dilate(object_mask.astype(np.uint8), object_kernel, iterations=1).astype(bool)

        if args.surface_priority_over_object and np.any(surface_mask):
            object_mask &= ~surface_mask
        if args.object_priority_over_cutout and np.any(object_mask):
            cutout_mask &= ~object_mask
            for group in list(cutout_groups):
                cutout_groups[group] &= ~object_mask

        masks_to_save = {
            "object_mask": object_mask,
            "surface": surface_mask,
            "cutout": cutout_mask,
        }
        masks_to_save.update({f"surface_{g}": m for g, m in surface_groups.items()})
        masks_to_save.update({f"cutout_{g}": m for g, m in cutout_groups.items()})
        save_per_view_masks(args.out_dir, pose, masks_to_save)

        overlay = overlay_mask(rgb, object_mask, (255, 60, 40))
        overlay = overlay_mask(overlay, surface_mask, (255, 200, 40))
        overlay = overlay_mask(overlay, cutout_mask, (40, 150, 255))
        for key in image_keys(pose)[:1]:
            Image.fromarray(overlay).save(overlay_dir / f"{key}_overlay.png")

        stat = {
            "image_id": pose.image_id,
            "name": pose.name,
            "object_pixels": int(np.count_nonzero(object_mask)),
            "surface_pixels": int(np.count_nonzero(surface_mask)),
            "cutout_pixels": int(np.count_nonzero(cutout_mask)),
            "prompts": prompt_stats,
        }
        all_stats.append(stat)
        print(
            f"[sam3-view] {pose.name}: object={stat['object_pixels']} surface={stat['surface_pixels']} cutout={stat['cutout_pixels']}",
            flush=True,
        )

    metadata_name = "metadata.json" if args.view_stride == 1 else f"metadata_worker_{args.view_start:02d}_of_{args.view_stride:02d}.json"
    (args.out_dir / metadata_name).write_text(
        json.dumps(
            {
                "method": "polygon_sam3_view_masks_for_v61_preprojection_reject",
                "params": jsonable_args(args),
                "faces": faces,
                "target_faces": target_faces,
                "palette_rgb": palette,
                "object_prompts": object_prompts,
                "surface_prompt_groups": SURFACE_PROMPT_GROUPS,
                "cutout_prompt_groups": ARCHITECTURAL_CUTOUT_PROMPT_GROUPS,
                "views": all_stats,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
