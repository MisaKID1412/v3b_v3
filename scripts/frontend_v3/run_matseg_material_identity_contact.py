#!/usr/bin/env python3
"""Resolve v3b proposal identity with one common-frame MatSeg inference.

Lab/appearance clustering supplies spatial proposals only.  For each proposal,
the highest projected-weight traceable rectified candidates are placed into a
single contact sheet.  MatSeg then compares all candidates in one forward pass,
which is the coordinate frame in which its class-agnostic descriptors are
defined.  No room name, face type, target material count, or Lab identity rule
is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from matseg_identity_resolver import (
    DEFAULT_COMMON_FRAME_STRONG_SIMILARITY,
    resolve_material_ids,
)
from run_matseg_floor_diagnostic import MatSegRunner, draw_similarity_matrix, robust_descriptor, unit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--face", required=True)
    parser.add_argument("--max-side", type=int, default=700)
    parser.add_argument("--candidates-per-region", type=int, default=2)
    parser.add_argument("--contact-tile-side", type=int, default=256)
    parser.add_argument(
        "--common-frame-strong-similarity",
        type=float,
        default=DEFAULT_COMMON_FRAME_STRONG_SIMILARITY,
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_candidate_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
    if mask is None:
        return np.ones(shape, dtype=bool)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def crop_identity_fallback_to_mask(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove black atlas context before a thin proposal enters MatSeg."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return image, mask
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = image[y0:y1, x0:x1].copy()
    selected = mask[y0:y1, x0:x1].copy()
    if np.any(selected):
        fill = np.median(crop[selected], axis=0).astype(np.uint8)
        crop[~selected] = fill
    return crop, selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    face_record = next(item for item in metadata["stats"] if item["face"] == args.face)
    regions = sorted(face_record["regions"], key=lambda item: int(item["region"]))
    if not regions:
        raise RuntimeError(f"Face {args.face} has no traceable region proposals")

    entries = []
    for region in regions:
        region_entries_before = len(entries)
        candidates = sorted(
            region.get("view_candidates", []),
            key=lambda item: (
                int(item.get("source_rank_by_weight", 10**9)),
                -float(item.get("weight", 0.0)),
            ),
        )
        for candidate in candidates[: args.candidates_per_region]:
            image_path = Path(candidate["chord_input"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            mask = load_candidate_mask(
                Path(candidate.get("candidate_mask", "")), image.shape[:2]
            )
            entries.append(
                {
                    "region": int(region["region"]),
                    "stem": str(candidate.get("stem", image_path.stem)),
                    "view_name": str(candidate.get("view_name", "")),
                    "source_rank_by_weight": int(
                        candidate.get("source_rank_by_weight", 10**9)
                    ),
                    "weight": float(candidate.get("weight", 1.0)),
                    "image": image,
                    "mask": mask,
                }
            )
        if len(entries) == region_entries_before:
            # A spatial proposal can be valid even when the strict rectifier
            # cannot yet produce a CHORD-sized crop (usually a narrow strip).
            # Keep it available to MatSeg using its atlas proposal tile; this
            # fallback is identity-only and is never forwarded to CHORD.
            image_path = Path(region.get("target_tile", ""))
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                mask = load_candidate_mask(
                    Path(region.get("target_mask", "")), image.shape[:2]
                )
                image, mask = crop_identity_fallback_to_mask(image, mask)
                entries.append(
                    {
                        "region": int(region["region"]),
                        "stem": f"{args.face}_r{int(region['region']):02d}_atlas_identity_fallback",
                        "view_name": "atlas_identity_only_fallback",
                        "source_rank_by_weight": 10**9,
                        "weight": float(region.get("cluster_score", 1.0)),
                        "image": image,
                        "mask": mask,
                    }
                )
    found_regions = {int(entry["region"]) for entry in entries}
    missing = [int(region["region"]) for region in regions if int(region["region"]) not in found_regions]
    if missing:
        raise RuntimeError(f"No readable rectified identity candidates for regions {missing}")

    tile_side = max(96, int(args.contact_tile_side))
    columns = int(math.ceil(math.sqrt(len(entries))))
    rows = int(math.ceil(len(entries) / columns))
    gutter = max(4, tile_side // 32)
    height = rows * tile_side + (rows + 1) * gutter
    width = columns * tile_side + (columns + 1) * gutter
    contact = np.full((height, width, 3), 127, np.uint8)
    masks = []
    manifest = []
    for index, entry in enumerate(entries):
        row, column = divmod(index, columns)
        y0 = gutter + row * (tile_side + gutter)
        x0 = gutter + column * (tile_side + gutter)
        image = cv2.resize(entry["image"], (tile_side, tile_side), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(
            entry["mask"].astype(np.uint8),
            (tile_side, tile_side),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        safe_border = max(8, tile_side // 12)
        interior = np.zeros((tile_side, tile_side), dtype=bool)
        interior[safe_border:-safe_border, safe_border:-safe_border] = True
        mask &= interior
        contact[y0 : y0 + tile_side, x0 : x0 + tile_side] = image
        sheet_mask = np.zeros((height, width), dtype=bool)
        sheet_mask[y0 : y0 + tile_side, x0 : x0 + tile_side] = mask
        masks.append(sheet_mask)
        manifest.append(
            {
                key: entry[key]
                for key in (
                    "region",
                    "stem",
                    "view_name",
                    "source_rank_by_weight",
                    "weight",
                )
            }
            | {"box_y0_y1_x0_x1": [y0, y0 + tile_side, x0, x0 + tile_side]}
        )

    contact_path = args.output_dir / "matseg_identity_contact_sheet.png"
    cv2.imwrite(str(contact_path), contact)
    runner = MatSegRunner(args.vendor_dir, args.checkpoint, args.device)
    feature, _resized = runner.infer(contact, args.max_side)
    descriptors_by_region: dict[int, list[tuple[np.ndarray, float]]] = {
        int(region["region"]): [] for region in regions
    }
    for entry, mask in zip(entries, masks):
        descriptors_by_region[int(entry["region"])].append(
            (robust_descriptor(feature, mask), max(float(entry["weight"]), 1e-8))
        )

    region_vectors = []
    report_regions = []
    resolver_regions = []
    for region in regions:
        region_id = int(region["region"])
        items = descriptors_by_region[region_id]
        weights = np.asarray([weight for _vector, weight in items], dtype=np.float64)
        weights /= max(float(np.sum(weights)), 1e-12)
        item_vectors = np.stack([vector for vector, _weight in items])
        vector = unit(np.sum(item_vectors * weights[:, None], axis=0))
        region_vectors.append(vector)
        uncertainty = None
        if len(items) >= 2:
            item_similarity = item_vectors @ item_vectors.T
            upper = item_similarity[np.triu_indices(len(items), k=1)]
            uncertainty = float(1.0 - np.median(upper))
        measured = {
            "region": region_id,
            "old_material_id": int(region["material_id"]),
            "source": str(region.get("source", "")),
            "discovery_index": region.get("discovery_index"),
            "material_box_purity": float(region.get("material_box_purity", 0.0)),
            "box_yx_size": [int(value) for value in region["box_yx_size"]],
            "descriptor_uncertainty_distance": uncertainty,
        }
        report_regions.append(measured)
        resolver_row = dict(region)
        if uncertainty is not None:
            resolver_row["descriptor_uncertainty_distance"] = uncertainty
        resolver_regions.append(resolver_row)

    vectors = np.stack(region_vectors).astype(np.float32)
    similarity = vectors @ vectors.T
    labels, audit = resolve_material_ids(
        resolver_regions,
        similarity,
        common_frame_strong_similarity=args.common_frame_strong_similarity,
    )
    for row, label in zip(report_regions, labels.tolist()):
        row["matseg_material_id"] = int(label)
    draw_similarity_matrix(
        similarity,
        [f"r{row['region']:02d}" for row in report_regions],
        args.output_dir / "matseg_identity_similarity.png",
    )
    report = {
        "method": "matseg_rectified_contact_material_identity_v1",
        "face": args.face,
        "identity_resolution": audit,
        "identity_similarity_policy": (
            "one MatSeg forward pass over descending-atlas-weight rectified candidates"
        ),
        "material_count": int(len(np.unique(labels))),
        "max_view_side": int(args.max_side),
        "candidates_per_region": int(args.candidates_per_region),
        "common_frame_strong_similarity": float(args.common_frame_strong_similarity),
        "regions": report_regions,
        "similarity_matrix": similarity.tolist(),
        "contact_sheet": str(contact_path),
        "contact_manifest": manifest,
        "gpu_peak_memory_mib": float(
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        )
        if torch.cuda.is_available()
        else 0.0,
    }
    output = args.output_dir / "matseg_material_identity_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "face": args.face,
                "material_count": report["material_count"],
                "report": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
