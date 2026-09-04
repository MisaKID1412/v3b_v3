#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
            "Build a non-tiled CHORD base atlas by inverting the atlas_rectified "
            "CHORD-input crop before material placement."
        )
    )
    parser.add_argument("--chord-input-metadata", type=Path, required=True)
    parser.add_argument("--chord-output-dir", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--strict-observed-dir", type=Path, default=None)
    parser.add_argument("--completed-observed-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--faces", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--use-soft-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quilt-block-frac", type=float, default=0.55)
    parser.add_argument("--quilt-overlap-frac", type=float, default=0.28)
    parser.add_argument("--max-quilt-block", type=int, default=220)
    parser.add_argument("--min-quilt-block", type=int, default=42)
    parser.add_argument("--high-texture-sobel-threshold", type=float, default=0.035)
    parser.add_argument("--anchor-feather-frac", type=float, default=0.10)
    parser.add_argument("--preview-thumb-width", type=int, default=340)
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_gray(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if shape is not None and image.shape != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def load_metadata(path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    regions: dict[tuple[str, int], dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    faces: list[str] = []
    for face_record in data["stats"]:
        face = face_record["face"]
        faces.append(face)
        shape_hw = face_record.get("shape_hw")
        for region in face_record.get("regions", []):
            region_record = dict(region)
            region_record["_face"] = face
            region_record["_shape_hw"] = shape_hw
            regions[(face, int(region["region"]))] = region_record
            for candidate in region.get("view_candidates", []):
                item = dict(candidate)
                item["_face"] = face
                item["_region"] = int(region["region"])
                item["_shape_hw"] = shape_hw
                item["_box_yx_size"] = region.get("box_yx_size")
                item["_material_id"] = int(region.get("material_id", region["region"]))
                candidates[item["stem"]] = item
    return regions, candidates, faces


def selected_chord_stems(chord_output_dir: Path) -> list[str]:
    stems = []
    for path in sorted(chord_output_dir.iterdir()):
        if path.is_dir() and (path / "basecolor.png").exists():
            stems.append(path.name)
    return stems


def material_map_by_face(candidates: dict[str, dict[str, Any]], chord_output_dir: Path) -> dict[str, dict[int, str]]:
    grouped: dict[str, dict[int, str]] = {}
    for stem in selected_chord_stems(chord_output_dir):
        candidate = candidates.get(stem)
        if candidate is None:
            continue
        face = candidate["_face"]
        material_id = int(candidate["_material_id"])
        grouped.setdefault(face, {})[material_id] = stem
    return grouped


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(weights.astype(np.float32), 0.0)
    denom = np.sum(weights, axis=0, keepdims=True)
    return weights / np.maximum(denom, 1e-8)


def load_layout_weights(layout_dir: Path, face: str) -> np.ndarray:
    npy = layout_dir / "labels_npy" / f"{face}_soft_weights.npy"
    if npy.exists():
        weights = np.load(npy)
        if weights.ndim == 2:
            weights = weights[None, ...]
        return normalize_weights(weights)
    label_path = layout_dir / "labels" / f"{face}.png"
    if not label_path.exists():
        raise FileNotFoundError(f"missing layout labels/weights for {face}")
    labels_rgb = np.asarray(Image.open(label_path).convert("RGB"), dtype=np.int16)
    distances = np.sum((labels_rgb[..., None, :] - LABEL_COLORS[None, None, :, :].astype(np.int16)) ** 2, axis=-1)
    labels = np.argmin(distances, axis=-1).astype(np.int16)
    material_count = int(labels.max()) + 1
    weights = np.stack([(labels == index).astype(np.float32) for index in range(material_count)], axis=0)
    return normalize_weights(weights)


def label_image(weights: np.ndarray) -> np.ndarray:
    labels = np.argmax(weights, axis=0)
    return LABEL_COLORS[labels % len(LABEL_COLORS)].astype(np.float32) / 255.0


def soft_weight_image(weights: np.ndarray) -> np.ndarray:
    colors = LABEL_COLORS[np.arange(weights.shape[0]) % len(LABEL_COLORS)].astype(np.float32) / 255.0
    return np.einsum("khw,kc->hwc", weights, colors).astype(np.float32)


def resize_area(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    interpolation = cv2.INTER_AREA if h < image.shape[0] or w < image.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(image, (w, h), interpolation=interpolation).astype(np.float32)


def texture_metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sob = np.sqrt(gx * gx + gy * gy)
    return {
        "laplacian_mean": float(np.mean(lap)),
        "sobel_mean": float(np.mean(sob)),
        "gray_std": float(np.std(gray)),
    }


def inverse_patch_from_chord(
    chord_output_dir: Path,
    stem: str,
    candidate: dict[str, Any],
    face_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    source = load_rgb(chord_output_dir / stem / "basecolor.png")
    shape_hw = candidate.get("_shape_hw")
    box = candidate.get("_box_yx_size")
    crop = candidate.get("inner_crop_box_y0_y1_x0_x1") or candidate.get("crop_box_y0_y1_x0_x1")
    if shape_hw is None or box is None or crop is None:
        raise ValueError(f"missing inverse-crop metadata for {stem}")
    scale_y = face_shape[0] / float(shape_hw[0])
    scale_x = face_shape[1] / float(shape_hw[1])
    box_y, box_x, _ = [float(v) for v in box]
    cy0, cy1, cx0, cx1 = [float(v) for v in crop]
    y0 = int(round((box_y + cy0) * scale_y))
    y1 = int(round((box_y + cy1) * scale_y))
    x0 = int(round((box_x + cx0) * scale_x))
    x1 = int(round((box_x + cx1) * scale_x))
    y0 = max(0, min(face_shape[0] - 1, y0))
    x0 = max(0, min(face_shape[1] - 1, x0))
    y1 = max(y0 + 1, min(face_shape[0], y1))
    x1 = max(x0 + 1, min(face_shape[1], x1))
    patch = resize_area(source, (y1 - y0, x1 - x0))
    info = {
        "stem": stem,
        "face": candidate["_face"],
        "region": int(candidate["_region"]),
        "material_id": int(candidate["_material_id"]),
        "source_basecolor": str(chord_output_dir / stem / "basecolor.png"),
        "source_basecolor_shape": [int(source.shape[0]), int(source.shape[1])],
        "metadata_shape_hw": [int(shape_hw[0]), int(shape_hw[1])],
        "face_shape_hw": [int(face_shape[0]), int(face_shape[1])],
        "scale_yx": [float(scale_y), float(scale_x)],
        "region_box_yx_size": [int(v) for v in box],
        "inner_crop_box_y0_y1_x0_x1": [int(v) for v in crop],
        "inverse_face_box_y0_y1_x0_x1": [int(y0), int(y1), int(x0), int(x1)],
        "inverse_patch_shape_hw": [int(patch.shape[0]), int(patch.shape[1])],
    }
    return patch, info


def feather_alpha(shape: tuple[int, int], feather: int) -> np.ndarray:
    h, w = shape
    alpha = np.ones((h, w), dtype=np.float32)
    if feather <= 0:
        return alpha
    fy = min(feather, max(1, h // 2))
    fx = min(feather, max(1, w // 2))
    ramp_y = np.ones(h, dtype=np.float32)
    ramp_x = np.ones(w, dtype=np.float32)
    if fy > 0:
        r = np.linspace(0.0, 1.0, fy, endpoint=False, dtype=np.float32)
        r = 0.5 - 0.5 * np.cos(np.pi * r)
        ramp_y[:fy] = r
        ramp_y[-fy:] = r[::-1]
    if fx > 0:
        r = np.linspace(0.0, 1.0, fx, endpoint=False, dtype=np.float32)
        r = 0.5 - 0.5 * np.cos(np.pi * r)
        ramp_x[:fx] = r
        ramp_x[-fx:] = r[::-1]
    alpha *= ramp_y[:, None]
    alpha *= ramp_x[None, :]
    return np.clip(alpha, 0.0, 1.0)


def crop_from_periodic(source: np.ndarray, y: int, x: int, h: int, w: int) -> np.ndarray:
    yy = (np.arange(h, dtype=np.int32) + int(y)) % source.shape[0]
    xx = (np.arange(w, dtype=np.int32) + int(x)) % source.shape[1]
    return source[yy[:, None], xx[None, :]].copy()


def vertical_min_cut(error: np.ndarray) -> np.ndarray:
    h, w = error.shape
    if h <= 0 or w <= 0:
        return np.ones((h, w), dtype=bool)
    cost = error.astype(np.float32).copy()
    back = np.zeros((h, w), dtype=np.int8)
    for y in range(1, h):
        previous = cost[y - 1]
        for x in range(w):
            lo = max(0, x - 1)
            hi = min(w, x + 2)
            local = previous[lo:hi]
            arg = int(np.argmin(local))
            cost[y, x] += float(local[arg])
            back[y, x] = int(lo + arg - x)
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(cost[-1]))
    for y in range(h - 2, -1, -1):
        seam[y] = int(np.clip(seam[y + 1] + int(back[y + 1, seam[y + 1]]), 0, w - 1))
    xx = np.arange(w, dtype=np.int32)[None, :]
    return xx >= seam[:, None]


def horizontal_min_cut(error: np.ndarray) -> np.ndarray:
    return vertical_min_cut(error.T).T


def quilt_texture(
    source: np.ndarray,
    shape: tuple[int, int],
    rng: np.random.Generator,
    block_frac: float,
    overlap_frac: float,
    min_block: int,
    max_block: int,
    auxiliary_source: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    h, w = shape
    source = np.clip(source.astype(np.float32), 0.0, 1.0)
    auxiliary = None if auxiliary_source is None else auxiliary_source.astype(np.float32)
    if auxiliary is not None and auxiliary.shape[:2] != source.shape[:2]:
        raise ValueError(
            f"auxiliary source shape {auxiliary.shape[:2]} does not match base {source.shape[:2]}"
        )
    if min(source.shape[:2]) < 12:
        scale = 12.0 / max(1.0, float(min(source.shape[:2])))
        resized_shape = (
            max(12, int(round(source.shape[0] * scale))),
            max(12, int(round(source.shape[1] * scale))),
        )
        source = resize_area(source, resized_shape)
        if auxiliary is not None:
            auxiliary = cv2.resize(
                auxiliary, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_CUBIC
            ).astype(np.float32)
    source_side = min(source.shape[:2])
    block = int(round(source_side * float(block_frac)))
    block = max(int(min_block), min(int(max_block), block))
    block = max(8, min(block, max(h, 1), max(w, 1)))
    overlap = int(round(block * float(overlap_frac)))
    overlap = max(2, min(block - 1, overlap))
    step = max(1, block - overlap)
    out = np.zeros((h, w, 3), dtype=np.float32)
    auxiliary_out = (
        np.zeros((h, w, auxiliary.shape[2]), dtype=np.float32)
        if auxiliary is not None
        else None
    )
    filled = np.zeros((h, w), dtype=bool)
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    if ys[-1] != max(0, h - block):
        ys.append(max(0, h - block))
    if xs[-1] != max(0, w - block):
        xs.append(max(0, w - block))
    ys = sorted(set(ys))
    xs = sorted(set(xs))
    candidate_count = 36
    for y in ys:
        for x in xs:
            yy = min(max(0, y), max(0, h - 1))
            xx = min(max(0, x), max(0, w - 1))
            bh = min(block, h - yy)
            bw = min(block, w - xx)
            if bh <= 0 or bw <= 0:
                continue
            best_patch = None
            best_auxiliary_patch = None
            best_score = None
            current_filled = filled[yy : yy + bh, xx : xx + bw]
            for _ in range(candidate_count):
                # The CHORD exemplar is not guaranteed to be seamless.  A
                # periodic crop can wrap inside one candidate and manufacture
                # a hard line before graph-cut placement even begins.  Sample
                # only valid native-scale crops whenever the exemplar fits.
                if source.shape[0] >= bh and source.shape[1] >= bw:
                    sy = int(rng.integers(0, source.shape[0] - bh + 1))
                    sx = int(rng.integers(0, source.shape[1] - bw + 1))
                    patch = source[sy : sy + bh, sx : sx + bw].copy()
                    auxiliary_patch = (
                        auxiliary[sy : sy + bh, sx : sx + bw].copy()
                        if auxiliary is not None
                        else None
                    )
                else:
                    sy = int(rng.integers(0, max(1, source.shape[0])))
                    sx = int(rng.integers(0, max(1, source.shape[1])))
                    patch = crop_from_periodic(source, sy, sx, bh, bw)
                    auxiliary_patch = (
                        crop_from_periodic(auxiliary, sy, sx, bh, bw)
                        if auxiliary is not None
                        else None
                    )
                if np.any(current_filled):
                    diff = out[yy : yy + bh, xx : xx + bw] - patch
                    err = np.sum(diff * diff, axis=2)
                    score = float(np.mean(err[current_filled]))
                else:
                    score = float(rng.random() * 1e-6)
                if best_score is None or score < best_score:
                    best_score = score
                    best_patch = patch
                    best_auxiliary_patch = auxiliary_patch
            patch = best_patch if best_patch is not None else crop_from_periodic(source, 0, 0, bh, bw)
            if auxiliary is not None and best_auxiliary_patch is None:
                best_auxiliary_patch = crop_from_periodic(auxiliary, 0, 0, bh, bw)
            paste_mask = np.ones((bh, bw), dtype=bool)
            current_filled = filled[yy : yy + bh, xx : xx + bw]
            ow = min(overlap, bw)
            if xx > 0 and ow > 0 and np.any(current_filled[:, :ow]):
                err = np.sum((out[yy : yy + bh, xx : xx + ow] - patch[:, :ow]) ** 2, axis=2)
                paste_mask[:, :ow] &= vertical_min_cut(err)
            oh = min(overlap, bh)
            if yy > 0 and oh > 0 and np.any(current_filled[:oh, :]):
                err = np.sum((out[yy : yy + oh, xx : xx + bw] - patch[:oh, :]) ** 2, axis=2)
                paste_mask[:oh, :] &= horizontal_min_cut(err)
            paste_mask |= ~current_filled
            target = out[yy : yy + bh, xx : xx + bw]
            # A narrow alpha transition removes one-pixel graph-cut color
            # discontinuities without changing the block's texel scale.
            alpha = cv2.GaussianBlur(
                paste_mask.astype(np.float32), (0, 0), sigmaX=1.25, sigmaY=1.25
            )
            alpha[~current_filled] = 1.0
            target = target * (1.0 - alpha[..., None]) + patch * alpha[..., None]
            out[yy : yy + bh, xx : xx + bw] = target
            if auxiliary_out is not None and best_auxiliary_patch is not None:
                auxiliary_target = auxiliary_out[yy : yy + bh, xx : xx + bw]
                auxiliary_target = (
                    auxiliary_target * (1.0 - alpha[..., None])
                    + best_auxiliary_patch * alpha[..., None]
                )
                auxiliary_out[yy : yy + bh, xx : xx + bw] = auxiliary_target
            filled[yy : yy + bh, xx : xx + bw] |= paste_mask
    missing = ~filled
    if np.any(missing):
        base = crop_from_periodic(source, 0, 0, h, w)
        out[missing] = base[missing]
        if auxiliary_out is not None and auxiliary is not None:
            auxiliary_base = crop_from_periodic(auxiliary, 0, 0, h, w)
            auxiliary_out[missing] = auxiliary_base[missing]
    base_result = np.clip(out, 0.0, 1.0)
    if auxiliary_out is None:
        return base_result
    return base_result, auxiliary_out


def anchor_patch(field: np.ndarray, patch: np.ndarray, box: list[int], feather_frac: float) -> np.ndarray:
    y0, y1, x0, x1 = [int(v) for v in box]
    h, w = y1 - y0, x1 - x0
    patch_resized = resize_area(patch, (h, w)) if patch.shape[:2] != (h, w) else patch
    out = field.copy()
    feather = int(round(min(h, w) * float(feather_frac)))
    alpha = feather_alpha((h, w), feather)
    out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1.0 - alpha[..., None]) + patch_resized * alpha[..., None]
    return np.clip(out, 0.0, 1.0)


def copy_reference_images(args: argparse.Namespace, face: str, shape: tuple[int, int]) -> None:
    if args.strict_observed_dir is not None:
        raw = args.strict_observed_dir / "debug" / f"{face}_raw_projected.png"
        keep = args.strict_observed_dir / "debug" / f"{face}_final_keep_mask.png"
        if raw.exists():
            image = load_rgb(raw)
            if image.shape[:2] != shape:
                image = resize_area(image, shape)
            if keep.exists():
                mask = load_gray(keep, shape) > 0.5
                image = image.copy()
                image[~mask] = 0.0
            save_rgb(args.out_dir / "observed_reference" / f"{face}.png", image)
    if args.completed_observed_dir is not None:
        completed_root = args.completed_observed_dir
        completed = completed_root / "completed_observed" / f"{face}.png"
        if not completed.exists():
            completed = completed_root / f"{face}.png"
        if completed.exists():
            image = load_rgb(completed)
            if image.shape[:2] != shape:
                image = resize_area(image, shape)
            save_rgb(args.out_dir / "completed_observed_reference" / f"{face}.png", image)


def thumbnail(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = width / max(1, image.width)
    height = max(1, int(round(image.height * scale)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def save_overview(args: argparse.Namespace, faces: list[str], records: list[dict[str, Any]]) -> None:
    columns = [
        ("observed", "observed_reference"),
        ("completed", "completed_observed_reference"),
        ("labels", "labels"),
        ("soft weights", "soft_weights"),
        ("inversecrop base", "textures_base"),
    ]
    thumb_w = int(args.preview_thumb_width)
    row_gap = 46
    col_gap = 14
    rows: list[Image.Image] = []
    font = ImageFont.load_default()
    for face in faces:
        thumbs = []
        max_h = 1
        for _, folder in columns:
            path = args.out_dir / folder / f"{face}.png"
            if path.exists():
                img = thumbnail(path, thumb_w)
            else:
                img = Image.new("RGB", (thumb_w, max(80, thumb_w // 3)), (20, 20, 20))
            thumbs.append(img)
            max_h = max(max_h, img.height)
        row = Image.new("RGB", (len(columns) * thumb_w + (len(columns) - 1) * col_gap, max_h + row_gap), (245, 245, 245))
        draw = ImageDraw.Draw(row)
        material_count = next((r["material_count"] for r in records if r["face"] == face), 0)
        draw.text((0, 4), f"{face} ({material_count} materials)", fill=(20, 20, 20), font=font)
        for i, ((title, _), img) in enumerate(zip(columns, thumbs)):
            x = i * (thumb_w + col_gap)
            draw.text((x, 22), title, fill=(20, 20, 20), font=font)
            row.paste(img, (x, row_gap))
        rows.append(row)
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    (args.out_dir / "previews").mkdir(parents=True, exist_ok=True)
    canvas.save(args.out_dir / "previews" / "inversecrop_nontile_base_overview.jpg", quality=94)


def save_material_sheet(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    panel = 190
    gap = 12
    row_h = 255
    rows = len(records)
    max_mats = max((len(r["materials"]) for r in records), default=1)
    width = gap + (1 + max_mats * 3) * (panel + gap)
    height = 48 + rows * row_h
    canvas = Image.new("RGB", (width, height), (246, 246, 246))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 14), "CHORD basecolor -> inverse-crop patch -> anchored non-tile material field", fill=(20, 20, 20), font=font)
    for row_index, record in enumerate(records):
        y = 48 + row_index * row_h
        draw.text((gap, y + 8), record["face"], fill=(20, 20, 20), font=font)
        col = 1
        for material in record["materials"]:
            for key, title in (("source_preview", "chord"), ("inverse_patch", "inverse"), ("material_field", "field")):
                path = Path(material[key])
                img = thumbnail(path, panel)
                x = gap + col * (panel + gap)
                draw.text((x, y + 8), f"m{material['material_index']} {title}", fill=(20, 20, 20), font=font)
                canvas.paste(img, (x, y + 30))
                col += 1
    (args.out_dir / "previews").mkdir(parents=True, exist_ok=True)
    canvas.save(args.out_dir / "previews" / "inversecrop_materials_sheet.jpg", quality=94)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, candidates, metadata_faces = load_metadata(args.chord_input_metadata)
    by_face = material_map_by_face(candidates, args.chord_output_dir)
    faces = args.faces or metadata_faces
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    for face in faces:
        weights = load_layout_weights(args.layout_dir, face)
        if not args.use_soft_weights:
            hard = np.argmax(weights, axis=0)
            weights = np.stack([(hard == index).astype(np.float32) for index in range(weights.shape[0])], axis=0)
        weights = normalize_weights(weights)
        material_count, h, w = weights.shape
        face_shape = (int(h), int(w))
        copy_reference_images(args, face, face_shape)
        save_rgb(args.out_dir / "labels" / f"{face}.png", label_image(weights))
        save_rgb(args.out_dir / "soft_weights" / f"{face}.png", soft_weight_image(weights))
        (args.out_dir / "labels_npy").mkdir(parents=True, exist_ok=True)
        np.save(args.out_dir / "labels_npy" / f"{face}_soft_weights.npy", weights.astype(np.float32))
        material_ids = sorted(by_face.get(face, {}).keys())
        if len(material_ids) < material_count:
            raise RuntimeError(f"{face}: only {len(material_ids)} CHORD materials for {material_count} layout channels")
        material_ids = material_ids[:material_count]
        fields = []
        materials = []
        for material_index, material_id in enumerate(material_ids):
            stem = by_face[face][material_id]
            candidate = candidates[stem]
            patch, patch_info = inverse_patch_from_chord(args.chord_output_dir, stem, candidate, face_shape)
            patch_path = args.out_dir / "materials_inverse" / face / f"material_{material_index:02d}_id{material_id}_{stem}_inverse_patch.png"
            save_rgb(patch_path, patch)
            source_preview_path = args.out_dir / "materials_source" / face / f"material_{material_index:02d}_id{material_id}_{stem}_chord_basecolor.png"
            save_rgb(source_preview_path, load_rgb(args.chord_output_dir / stem / "basecolor.png"))
            metrics = texture_metrics(patch)
            if metrics["sobel_mean"] >= float(args.high_texture_sobel_threshold):
                synthesis_method = "mincut_quilt_high_texture"
                field = quilt_texture(
                    patch,
                    face_shape,
                    rng,
                    args.quilt_block_frac,
                    args.quilt_overlap_frac,
                    args.min_quilt_block,
                    args.max_quilt_block,
                )
                field = anchor_patch(field, patch, patch_info["inverse_face_box_y0_y1_x0_x1"], args.anchor_feather_frac)
            else:
                synthesis_method = "inverse_patch_stretch_low_texture"
                field = resize_area(patch, face_shape)
                field = anchor_patch(field, patch, patch_info["inverse_face_box_y0_y1_x0_x1"], args.anchor_feather_frac)
            field_path = args.out_dir / "material_fields" / face / f"material_{material_index:02d}_id{material_id}_{stem}_field.png"
            save_rgb(field_path, field)
            fields.append(field)
            materials.append(
                {
                    "material_index": int(material_index),
                    "material_id": int(material_id),
                    "stem": stem,
                    "source_preview": str(source_preview_path),
                    "inverse_patch": str(patch_path),
                    "material_field": str(field_path),
                    "synthesis_method": synthesis_method,
                    "texture_metrics": metrics,
                    "inverse_crop": patch_info,
                    "soft_weight_fraction": float(np.mean(weights[material_index])),
                    "hard_label_fraction": float(np.mean(np.argmax(weights, axis=0) == material_index)),
                }
            )
        atlas = np.zeros((h, w, 3), dtype=np.float32)
        for index, field in enumerate(fields):
            atlas += weights[index, ..., None] * field
        save_rgb(args.out_dir / "textures_base" / f"{face}.png", atlas)
        records.append(
            {
                "face": face,
                "shape_hw": [int(h), int(w)],
                "material_count": int(material_count),
                "materials": materials,
            }
        )
        print(
            f"[inversecrop-nontile] {face}: materials={material_count} "
            f"stems={[m['stem'] for m in materials]}",
            flush=True,
        )
    save_overview(args, faces, records)
    save_material_sheet(args, records)
    metadata = {
        "method": "inversecrop_nontile_atlas_v1",
        "summary": (
            "Each 2048 CHORD basecolor is first resized back to the original "
            "atlas_rectified inner crop size and anchored at its source face location. "
            "A non-rotating stochastic quilt expands that same-scale patch over the "
            "face; existing notile soft weights assign/blend materials. No real "
            "observed pixels or low-frequency color transfer are pasted into textures_base."
        ),
        "chord_input_metadata": str(args.chord_input_metadata),
        "chord_output_dir": str(args.chord_output_dir),
        "layout_dir": str(args.layout_dir),
        "strict_observed_dir": str(args.strict_observed_dir) if args.strict_observed_dir else None,
        "completed_observed_dir": str(args.completed_observed_dir) if args.completed_observed_dir else None,
        "parameters": {
            "seed": int(args.seed),
            "use_soft_weights": bool(args.use_soft_weights),
            "quilt_block_frac": float(args.quilt_block_frac),
            "quilt_overlap_frac": float(args.quilt_overlap_frac),
            "max_quilt_block": int(args.max_quilt_block),
            "min_quilt_block": int(args.min_quilt_block),
            "high_texture_sobel_threshold": float(args.high_texture_sobel_threshold),
            "anchor_feather_frac": float(args.anchor_feather_frac),
        },
        "faces": records,
    }
    (args.out_dir / "metadata_inversecrop_nontile_atlas.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
