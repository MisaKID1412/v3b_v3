#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CHORD inputs with patch-preserving texture synthesis. The selected "
            "material patch stays the only source; this stage only removes crop/lighting "
            "artifacts and makes a seamless material reference."
        )
    )
    parser.add_argument("--current-run-dir", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--chord-input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=144)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--candidates-per-site", type=int, default=12)
    parser.add_argument("--lighting-normalize-strength", type=float, default=0.28)
    parser.add_argument(
        "--seamless-mode",
        choices=("blend", "mincut"),
        default="blend",
        help=(
            "blend: legacy linear cross-fade of opposite borders (size/12 px band). "
            "mincut: hide the wrap seam under a min-cut joined patch of the same tile, "
            "avoiding the blurred border frame that shows as a grid under periodic tiling."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--preview-thumb", type=int, default=180)
    return parser.parse_args()


def selected_materials(layout_dir: Path) -> list[dict[str, Any]]:
    metadata = json.loads((layout_dir / "metadata_material_placement.json").read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for face in metadata["faces"]:
        for material in sorted(face["materials"], key=lambda item: int(item["material_index"])):
            out.append(
                {
                    "face": face["face"],
                    "material_index": int(material["material_index"]),
                    "material_id": int(material.get("material_id", material["material_index"])),
                    "stem": material["chosen_stem"],
                }
            )
    return out


def load_rgb(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)).save(path)


def pil_blur(image: np.ndarray, radius: float) -> np.ndarray:
    pil = Image.fromarray(np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.float32) / 255.0


def normalize_lighting(image: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image.astype(np.float32)
    sigma = max(18.0, min(image.shape[:2]) / 5.5)
    low = pil_blur(image, sigma)
    mean = np.mean(low.reshape(-1, 3), axis=0).reshape(1, 1, 3)
    corrected = np.clip(image / np.maximum(low, 1e-3) * mean, 0.0, 1.0)
    return np.clip((1.0 - strength) * image + strength * corrected, 0.0, 1.0).astype(np.float32)


def gray(image: np.ndarray) -> np.ndarray:
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)


def edge_map(image: np.ndarray) -> np.ndarray:
    g = gray(image)
    dx = np.zeros_like(g)
    dy = np.zeros_like(g)
    dx[:, 1:-1] = 0.5 * (g[:, 2:] - g[:, :-2])
    dy[1:-1, :] = 0.5 * (g[2:, :] - g[:-2, :])
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def long_line_score(edge: np.ndarray, lum: np.ndarray, y: int, x: int, patch_size: int) -> float:
    e = edge[y : y + patch_size, x : x + patch_size]
    l = lum[y : y + patch_size, x : x + patch_size]
    if e.size == 0:
        return 0.0
    row_e = e.mean(axis=1)
    col_e = e.mean(axis=0)
    row_dark = (1.0 - l).mean(axis=1)
    col_dark = (1.0 - l).mean(axis=0)

    def zmax(v: np.ndarray) -> float:
        return float((v.max() - np.median(v)) / (np.std(v) + 1e-6))

    return max(zmax(row_e), zmax(col_e), 0.45 * zmax(row_dark), 0.45 * zmax(col_dark))


def candidate_windows(image: np.ndarray, patch_size: int, stride: int) -> list[tuple[int, int, float]]:
    h, w = image.shape[:2]
    edge = edge_map(image)
    lum = gray(image)
    ys = list(range(0, max(1, h - patch_size + 1), stride))
    xs = list(range(0, max(1, w - patch_size + 1), stride))
    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)
    windows = [(y, x, long_line_score(edge, lum, y, x, patch_size)) for y in ys for x in xs]
    windows.sort(key=lambda item: item[2])
    keep = max(24, min(len(windows), 220))
    return windows[:keep]


def overlap_cost(canvas: np.ndarray, valid: np.ndarray, y: int, x: int, patch: np.ndarray, overlap: int) -> float:
    ph, pw = patch.shape[:2]
    total = 0.0
    count = 0
    if y > 0:
        region = canvas[y : y + overlap, x : x + pw]
        mask = valid[y : y + overlap, x : x + pw]
        diff = region - patch[:overlap, :]
        total += float(np.mean((diff[mask] ** 2))) if np.any(mask) else 0.0
        count += int(np.any(mask))
    if x > 0:
        region = canvas[y : y + ph, x : x + overlap]
        mask = valid[y : y + ph, x : x + overlap]
        diff = region - patch[:, :overlap]
        total += float(np.mean((diff[mask] ** 2))) if np.any(mask) else 0.0
        count += int(np.any(mask))
    return total / max(count, 1)


def paste_patch(canvas: np.ndarray, valid: np.ndarray, y: int, x: int, patch: np.ndarray, overlap: int) -> None:
    ph, pw = patch.shape[:2]
    existing = canvas[y : y + ph, x : x + pw]
    mask = valid[y : y + ph, x : x + pw]
    blended = patch.copy()
    if y > 0:
        alpha = np.linspace(0.0, 1.0, overlap, dtype=np.float32).reshape(overlap, 1, 1)
        blended[:overlap, :] = (1.0 - alpha) * existing[:overlap, :] + alpha * blended[:overlap, :]
    if x > 0:
        alpha = np.linspace(0.0, 1.0, overlap, dtype=np.float32).reshape(1, overlap, 1)
        blended[:, :overlap] = (1.0 - alpha) * existing[:, :overlap] + alpha * blended[:, :overlap]
    existing[~mask] = blended[~mask]
    existing[mask] = 0.55 * existing[mask] + 0.45 * blended[mask]
    valid[y : y + ph, x : x + pw] = True


def quilt_texture(image: np.ndarray, size: int, patch_size: int, overlap: int, top_k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    patch_size = int(min(patch_size, size))
    overlap = int(min(overlap, max(8, patch_size // 2 - 1)))
    step = patch_size - overlap
    starts_y = list(range(0, max(1, size - patch_size + 1), step))
    starts_x = list(range(0, max(1, size - patch_size + 1), step))
    if starts_y[-1] != size - patch_size:
        starts_y.append(size - patch_size)
    if starts_x[-1] != size - patch_size:
        starts_x.append(size - patch_size)

    windows = candidate_windows(image, patch_size, max(10, patch_size // 8))
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    valid = np.zeros((size, size), dtype=bool)
    for y in starts_y:
        for x in starts_x:
            scored: list[tuple[float, int, int]] = []
            for wy, wx, line_score in windows:
                patch = image[wy : wy + patch_size, wx : wx + patch_size]
                cost = overlap_cost(canvas, valid, y, x, patch, overlap)
                cost += 0.012 * float(line_score)
                cost += 0.002 * float(rng.random())
                scored.append((cost, wy, wx))
            scored.sort(key=lambda item: item[0])
            pool = scored[: max(1, min(top_k, len(scored)))]
            rank = int(rng.integers(0, len(pool)))
            _, wy, wx = pool[rank]
            paste_patch(canvas, valid, y, x, image[wy : wy + patch_size, wx : wx + patch_size], overlap)
    return np.clip(canvas, 0.0, 1.0)


def make_edge_seamless(image: np.ndarray, seam_px: int) -> np.ndarray:
    h, w = image.shape[:2]
    seam_px = int(min(max(4, seam_px), h // 3, w // 3))
    out = image.astype(np.float32, copy=True)
    original = out.copy()
    for x in range(seam_px):
        alpha = (x + 1.0) / (seam_px + 1.0)
        out[:, x] = alpha * original[:, x] + (1.0 - alpha) * original[:, w - seam_px + x]
        out[:, w - seam_px + x] = alpha * original[:, w - seam_px + x] + (1.0 - alpha) * original[:, x]
    original = out.copy()
    for y in range(seam_px):
        alpha = (y + 1.0) / (seam_px + 1.0)
        out[y, :] = alpha * original[y, :] + (1.0 - alpha) * original[h - seam_px + y, :]
        out[h - seam_px + y] = alpha * original[h - seam_px + y] + (1.0 - alpha) * original[y]
    return np.clip(out, 0.0, 1.0)


def _vertical_min_cut(error: np.ndarray) -> np.ndarray:
    """Return, per row, the column index of the minimum-cost top-to-bottom cut."""
    h, w = error.shape
    cost = error.astype(np.float64, copy=True)
    back = np.zeros((h, w), dtype=np.int32)
    for y in range(1, h):
        prev = cost[y - 1]
        left = np.concatenate([[np.inf], prev[:-1]])
        right = np.concatenate([prev[1:], [np.inf]])
        stacked = np.stack([left, prev, right], axis=0)
        choice = np.argmin(stacked, axis=0)
        back[y] = choice - 1
        cost[y] += stacked[choice, np.arange(w)]
    cut = np.zeros(h, dtype=np.int32)
    cut[-1] = int(np.argmin(cost[-1]))
    for y in range(h - 1, 0, -1):
        cut[y - 1] = cut[y] + back[y, cut[y]]
    return cut


def _mincut_wrap_seam_x(image: np.ndarray, band: int, rng: np.random.Generator) -> np.ndarray:
    """Make the horizontal wrap seamless by rolling half a tile, then hiding the
    centre discontinuity under a continuous patch of the *same* tile joined on
    both sides with vertical min-cuts. Only the centre band is modified, so the
    left/right wrap stays continuous by construction."""
    h, w = image.shape[:2]
    band = int(min(max(8, band), w // 4))
    rolled = np.roll(image, w // 2, axis=1)
    x0 = w // 2 - band
    x1 = w // 2 + band
    target = rolled[:, x0:x1]
    # search a continuous 2*band-wide patch of the original tile whose two
    # outer columns match the rolled content at the band edges best
    best = None
    edges_l = rolled[:, x0 - 2 : x0]
    edges_r = rolled[:, x1 : x1 + 2]
    starts = list(range(0, w - 2 * band, max(1, (w - 2 * band) // 64)))
    rng.shuffle(starts)
    for sx in starts:
        patch = image[:, sx : sx + 2 * band]
        err = float(np.mean((patch[:, :2] - edges_l) ** 2) + np.mean((patch[:, -2:] - edges_r) ** 2))
        if best is None or err < best[0]:
            best = (err, sx)
    patch = image[:, best[1] : best[1] + 2 * band].astype(np.float32)
    # Transfer only the patch's high-frequency detail: a tile usually carries a
    # residual lighting gradient, so a strip taken elsewhere has a different
    # local mean and would show up as a faint repeated band on plain surfaces.
    low_sigma = max(4.0, band / 2.0)
    patch = np.clip(patch - pil_blur(patch, low_sigma) + pil_blur(target, low_sigma), 0.0, 1.0)
    err_map = np.mean((patch - target) ** 2, axis=2)
    # left cut inside the left half of the band, right cut inside the right half
    cut_l = _vertical_min_cut(err_map[:, :band]) + 0
    cut_r = _vertical_min_cut(err_map[:, band:]) + band
    out = rolled.astype(np.float32, copy=True)
    xs = np.arange(2 * band)[None, :]
    use_patch = (xs >= cut_l[:, None]) & (xs <= cut_r[:, None])
    band_out = np.where(use_patch[..., None], patch, target)
    out[:, x0:x1] = band_out
    return np.roll(out, -(w // 2), axis=1)


def make_edge_seamless_mincut(image: np.ndarray, band: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = _mincut_wrap_seam_x(image.astype(np.float32), band, rng)
    out = _mincut_wrap_seam_x(np.transpose(out, (1, 0, 2)), band, rng)
    return np.clip(np.transpose(out, (1, 0, 2)), 0.0, 1.0)


def match_mean_std(image: np.ndarray, reference: np.ndarray, strength: float = 0.8) -> np.ndarray:
    src_mean = image.reshape(-1, 3).mean(axis=0)
    src_std = image.reshape(-1, 3).std(axis=0)
    ref_mean = reference.reshape(-1, 3).mean(axis=0)
    ref_std = reference.reshape(-1, 3).std(axis=0)
    matched = (image - src_mean.reshape(1, 1, 3)) / np.maximum(src_std.reshape(1, 1, 3), 1e-5)
    matched = matched * ref_std.reshape(1, 1, 3) + ref_mean.reshape(1, 1, 3)
    return np.clip((1.0 - strength) * image + strength * matched, 0.0, 1.0)


def synthesize(
    image: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = normalize_lighting(image, args.lighting_normalize_strength)
    texture_energy = float(np.std(edge_map(normalized)))
    color_std = float(np.mean(np.std(normalized.reshape(-1, 3), axis=0)))
    if texture_energy < 0.012 and color_std < 0.055:
        quilted = normalized.copy()
    else:
        quilted = quilt_texture(
            normalized,
            args.size,
            args.patch_size,
            args.overlap,
            args.candidates_per_site,
            seed,
        )
    if getattr(args, "seamless_mode", "blend") == "mincut":
        tileable = make_edge_seamless_mincut(quilted, max(16, args.size // 12), seed)
    else:
        tileable = make_edge_seamless(quilted, max(16, args.size // 12))
    tileable = match_mean_std(tileable, normalized, strength=0.75)
    return normalized, quilted, tileable


def thumb(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    height = max(1, int(round(image.height * width / max(1, image.width))))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def save_preview(out_dir: Path, rows: list[dict[str, Any]], width: int) -> None:
    font = ImageFont.load_default()
    gap = 14
    label_w = 230
    cols = [
        ("selected patch", "source"),
        ("mild normalized", "normalized"),
        ("quilted source", "quilted"),
        ("tileable CHORD input", "tileable"),
    ]
    row_h = width + 62
    canvas = Image.new("RGB", (label_w + len(cols) * (width + gap) + gap, 44 + row_h * len(rows)), (246, 246, 246))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), "Route3B: selected patch -> patch-preserving tileable CHORD input", fill=(20, 20, 20), font=font)
    for row_i, row in enumerate(rows):
        y = 44 + row_i * row_h
        draw.multiline_text(
            (10, y + 8),
            f"{row['face']} m{row['material_index']}\n{row['stem']}",
            fill=(20, 20, 20),
            font=font,
            spacing=3,
        )
        for col, (title, key) in enumerate(cols):
            x = label_w + col * (width + gap)
            draw.text((x, y + 6), title, fill=(20, 20, 20), font=font)
            canvas.paste(thumb(Path(row[key]), width), (x, y + 26))
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / "previews" / "route3b_patch_preserving_inputs_sheet.jpg", quality=94)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(selected_materials(args.layout_dir)):
        stem = item["stem"]
        source_path = args.chord_input_dir / f"{stem}.png"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source = load_rgb(source_path, args.size)
        normalized, quilted, tileable = synthesize(
            source,
            args,
            args.seed + index * 101,
        )
        source_copy = args.out_dir / "selected_patch_inputs" / f"{stem}.png"
        normalized_path = args.out_dir / "normalized_inputs" / f"{stem}.png"
        quilted_path = args.out_dir / "quilted_inputs" / f"{stem}.png"
        tileable_path = args.out_dir / "tileable_chord_inputs" / f"{stem}.png"
        save_rgb(source_copy, source)
        save_rgb(normalized_path, normalized)
        save_rgb(quilted_path, quilted)
        save_rgb(tileable_path, tileable)
        rows.append(
            {
                **item,
                "source": str(source_copy),
                "normalized": str(normalized_path),
                "quilted": str(quilted_path),
                "tileable": str(tileable_path),
            }
        )
        print(f"[route3b-prepare] {item['face']} m{item['material_index']} {stem}", flush=True)

    metadata = {
        "method": "route3b_patch_preserving_tileable_inputs_v1",
        "summary": (
            "Selected material patches are converted into tileable CHORD inputs with "
            "non-generative patch quilting. The selected patch is the only source, so "
            "fine texture statistics are preserved better than SDXL/StableMaterials."
        ),
        "current_run_dir": str(args.current_run_dir),
        "layout_dir": str(args.layout_dir),
        "chord_input_dir": str(args.chord_input_dir),
        "size": int(args.size),
        "patch_size": int(args.patch_size),
        "overlap": int(args.overlap),
        "lighting_normalize_strength": float(args.lighting_normalize_strength),
        "items": rows,
    }
    (args.out_dir / "metadata_route3b_patch_preserving_inputs.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_preview(args.out_dir, rows, args.preview_thumb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
