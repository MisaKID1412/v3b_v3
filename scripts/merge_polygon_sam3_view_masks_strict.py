#!/usr/bin/env python3
"""Merge SAM3 view-level reject masks for the polygon v53 texture path.

The source projector only consumes a single per-view object mask. The SAM3 pass
also writes architectural/surface masks such as whiteboards, wall art, signs,
and doors. This script folds those masks into the projector-facing object mask
so rejected texels are blocked before weighted projection.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


INDEX_RE = re.compile(r"^(?:view_)?(\d+)_object_mask$")


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    if shape is not None and arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return arr > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def prefix_variants(index: int) -> list[str]:
    return [
        f"{index:06d}",
        f"{index:03d}",
        f"view_{index:06d}",
        f"view_{index:03d}",
    ]


def collect_indices(mask_dir: Path) -> list[int]:
    indices: set[int] = set()
    for path in mask_dir.glob("*_object_mask.png"):
        match = INDEX_RE.match(path.stem)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def collect_component_paths(mask_dir: Path, index: int) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for prefix in prefix_variants(index):
        for path in mask_dir.glob(f"{prefix}_surface*.png"):
            if path not in seen:
                paths.append(path)
                seen.add(path)
        for path in mask_dir.glob(f"{prefix}_cutout*.png"):
            if path not in seen:
                paths.append(path)
                seen.add(path)
        obj = mask_dir / f"{prefix}_object_mask.png"
        if obj.exists() and obj not in seen:
            paths.insert(0, obj)
            seen.add(obj)
    return paths


def clean_mask(mask: np.ndarray, close_px: int, dilate_px: int, min_area: int) -> np.ndarray:
    out = mask.astype(np.uint8)
    if close_px > 0 and np.any(out):
        k = np.ones((2 * close_px + 1, 2 * close_px + 1), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if min_area > 0 and np.any(out):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        keep = np.zeros_like(out)
        for label in range(1, n):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                keep[labels == label] = 1
        out = keep
    if dilate_px > 0 and np.any(out):
        k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        out = cv2.dilate(out, k, iterations=1)
    return out.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--close-px", type=int, default=2)
    parser.add_argument("--dilate-px", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=24)
    parser.add_argument("--copy-view-face-masks", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    indices = collect_indices(args.input_dir)
    records = []
    for index in indices:
        component_paths = collect_component_paths(args.input_dir, index)
        if not component_paths:
            continue
        first = read_mask(component_paths[0])
        merged = np.zeros(first.shape, dtype=bool)
        component_counts = {}
        for path in component_paths:
            mask = read_mask(path, first.shape)
            merged |= mask
            component_counts[path.name] = int(np.count_nonzero(mask))
        merged = clean_mask(merged, args.close_px, args.dilate_px, args.min_area)
        for prefix in prefix_variants(index):
            write_mask(args.out_dir / f"{prefix}_object_mask.png", merged)
        records.append(
            {
                "index": index,
                "texels": int(np.count_nonzero(merged)),
                "components": component_counts,
            }
        )

    src_face_masks = args.input_dir / "view_face_masks"
    if args.copy_view_face_masks and src_face_masks.exists():
        shutil.copytree(src_face_masks, args.out_dir / "view_face_masks")

    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "num_views": len(records),
                "close_px": args.close_px,
                "dilate_px": args.dilate_px,
                "min_area": args.min_area,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
