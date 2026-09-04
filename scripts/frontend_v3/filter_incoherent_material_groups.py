#!/usr/bin/env python3
"""Reject geometrically incoherent projection-contamination material groups.

This stage runs after MatSeg identity resolution.  It never decides whether
two appearances are the same material.  It only rejects a whole resolved group
when its projected atlas evidence is a severe connectedness outlier: many
scattered components with no dominant territory. Structural bands/stripes are
protected from generic shape pruning, but must still provide a native-scale
source crop; this rejects upscaled blank strips without deciding material
identity.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--region-assets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-coherence-score", type=float, default=0.10)
    parser.add_argument("--relative-coherence-ratio", type=float, default=0.30)
    parser.add_argument("--min-significant-components", type=int, default=10)
    parser.add_argument("--significant-component-fraction", type=float, default=0.005)
    parser.add_argument("--max-unstructured-singleton-fraction", type=float, default=0.05)
    parser.add_argument("--min-structural-native-crop-side", type=int, default=64)
    return parser.parse_args()


def structural(region: dict[str, Any]) -> bool:
    source = str(region.get("source", "")).lower()
    discovery = region.get("discovery_index")
    return bool(
        "band" in source
        or "stripe" in source
        or (isinstance(discovery, str) and not discovery.lstrip("-+").isdigit())
    )


def group_mask(
    region_assets_dir: Path, face: str, members: list[dict[str, Any]]
) -> np.ndarray:
    merged = None
    for region in members:
        path = (
            region_assets_dir
            / "debug"
            / f"{face}_region_{int(region['region']):02d}_material_mask.png"
        )
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            support_path = (
                region_assets_dir
                / "debug"
                / f"{face}_region_{int(region['region']):02d}_support.png"
            )
            mask = cv2.imread(str(support_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(path)
        selected = mask > 127
        merged = selected if merged is None else (merged | selected)
    if merged is None:
        raise RuntimeError(f"No region masks for {face}")
    return merged


def coherence(mask: np.ndarray, significant_fraction: float) -> dict[str, Any]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    areas = sorted(
        [int(value) for value in stats[1:count, cv2.CC_STAT_AREA]], reverse=True
    )
    pixels = int(np.count_nonzero(mask))
    significant_floor = max(32, int(math.ceil(significant_fraction * pixels)))
    significant = sum(area >= significant_floor for area in areas)
    largest_fraction = float(areas[0] / max(pixels, 1)) if areas else 0.0
    score = largest_fraction / math.sqrt(max(significant, 1))
    return {
        "pixels": pixels,
        "component_count": len(areas),
        "significant_component_count": int(significant),
        "significant_component_floor_pixels": int(significant_floor),
        "largest_component_fraction": largest_fraction,
        "coherence_score": float(score),
    }


def best_native_crop_side(members: list[dict[str, Any]]) -> int | None:
    sides = []
    for region in members:
        for candidate in region.get("view_candidates", []):
            side = candidate.get("inner_crop_side")
            if side is not None:
                sides.append(int(side))
    return max(sides) if sides else None


def main() -> None:
    args = parse_args()
    source = json.loads(args.metadata.read_text(encoding="utf-8"))
    result = copy.deepcopy(source)
    face_audits = []
    for face_record in result["stats"]:
        face = str(face_record["face"])
        groups: dict[int, list[dict[str, Any]]] = {}
        for region in face_record["regions"]:
            groups.setdefault(int(region["material_id"]), []).append(region)
        measurements = {}
        for material_id, members in sorted(groups.items()):
            row = coherence(
                group_mask(args.region_assets_dir, face, members),
                args.significant_component_fraction,
            )
            row["structural_boundary"] = any(structural(region) for region in members)
            row["regions"] = [int(region["region"]) for region in members]
            row["best_native_crop_side"] = best_native_crop_side(members)
            measurements[material_id] = row

        eligible = [
            material_id
            for material_id, row in measurements.items()
            if not row["structural_boundary"]
        ]
        scores = [measurements[material_id]["coherence_score"] for material_id in eligible]
        median_score = float(np.median(scores)) if scores else 0.0
        rejected_ids = set()
        for material_id, row in measurements.items():
            native_side = row["best_native_crop_side"]
            if (
                row["structural_boundary"]
                and native_side is not None
                and native_side < args.min_structural_native_crop_side
            ):
                rejected_ids.add(material_id)
                row["rejection_reason"] = (
                    "structural_proposal_lacks_native_scale_source_evidence"
                )
        if len(groups) >= 3:
            for material_id, members in groups.items():
                if any(structural(region) for region in members) or len(members) != 1:
                    continue
                material_fraction = float(members[0].get("material_fraction", 1.0))
                if material_fraction < args.max_unstructured_singleton_fraction:
                    rejected_ids.add(material_id)
                    measurements[material_id]["rejection_reason"] = (
                        "tiny_unstructured_singleton_projection_evidence"
                    )
                    measurements[material_id]["material_fraction"] = material_fraction
        if len(groups) >= 3 and len(eligible) >= 3 and median_score > 0.0:
            for material_id in eligible:
                row = measurements[material_id]
                relative = float(row["coherence_score"] / median_score)
                row["relative_to_face_median"] = relative
                if (
                    row["coherence_score"] < args.max_coherence_score
                    and relative < args.relative_coherence_ratio
                    and row["significant_component_count"]
                    >= args.min_significant_components
                ):
                    rejected_ids.add(material_id)
                    row["rejection_reason"] = (
                        "severe_disconnected_projection_evidence_outlier"
                    )

        kept = [
            region
            for region in face_record["regions"]
            if int(region["material_id"]) not in rejected_ids
        ]
        if not kept:
            kept = face_record["regions"]
            rejected_ids.clear()
        remap: dict[int, int] = {}
        for region in kept:
            old = int(region["material_id"])
            remap.setdefault(old, len(remap))
            region["material_id"] = remap[old]
        face_record["regions"] = kept
        face_record["material_count"] = len(remap)
        face_audits.append(
            {
                "face": face,
                "source_material_count": len(groups),
                "filtered_material_count": len(remap),
                "median_nonstructural_coherence_score": median_score,
                "measurements": measurements,
                "rejected_material_ids": sorted(rejected_ids),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    receipt = {
        "method": "post_matseg_geometric_coherence_outlier_filter_v1",
        "material_identity_changed": False,
        "room_face_or_material_count_profile_used": False,
        "parameters": {
            "max_coherence_score": args.max_coherence_score,
            "relative_coherence_ratio": args.relative_coherence_ratio,
            "min_significant_components": args.min_significant_components,
            "significant_component_fraction": args.significant_component_fraction,
            "max_unstructured_singleton_fraction": (
                args.max_unstructured_singleton_fraction
            ),
            "min_structural_native_crop_side": args.min_structural_native_crop_side,
        },
        "faces": face_audits,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
