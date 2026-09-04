#!/usr/bin/env python3
"""Recover support from an untraceable structural duplicate without generating it.

The first trace-back pass may prove that a narrow band/stripe proposal has no
native-scale CHORD crop.  Such a proposal must not become its own generated
material.  Its strict projected support is still useful when two independent
MatSeg measurements agree that it is a duplicate observation of a surviving
material.  In that case only its material ID is reassigned and a second
trace-back pass merges its support into the surviving material territory.

No room, face, material count, color, or Lab threshold is consulted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ORIGINAL_VIEW_SIMILARITY_FLOOR = 0.90
COMMON_FRAME_SIMILARITY_FLOOR = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-metadata", type=Path, required=True)
    parser.add_argument("--first-trace-metadata", type=Path, required=True)
    parser.add_argument("--original-view-reports-root", type=Path, required=True)
    parser.add_argument("--contact-reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural(region: dict[str, Any]) -> bool:
    source = str(region.get("source", "")).lower()
    discovery = region.get("discovery_index")
    return bool(
        "band" in source
        or "stripe" in source
        or (isinstance(discovery, str) and not discovery.lstrip("-+").isdigit())
    )


def load_report(path: Path) -> tuple[dict[int, int], np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8"))
    regions = sorted(report["regions"], key=lambda row: int(row["region"]))
    return (
        {int(region["region"]): index for index, region in enumerate(regions)},
        np.asarray(report["similarity_matrix"], dtype=np.float64),
    )


def main() -> None:
    args = parse_args()
    identity = json.loads(args.identity_metadata.read_text(encoding="utf-8"))
    traced = json.loads(args.first_trace_metadata.read_text(encoding="utf-8"))
    result = copy.deepcopy(traced)
    identity_faces = {str(face["face"]): face for face in identity["stats"]}
    receipts = []

    for output_face in result["stats"]:
        face = str(output_face["face"])
        source_face = identity_faces[face]
        source_regions = {
            int(region["region"]): region for region in source_face.get("regions", [])
        }
        kept_regions = {
            int(region["region"]): region for region in output_face.get("regions", [])
        }
        missing_ids = sorted(set(source_regions) - set(kept_regions))
        original_path = (
            args.original_view_reports_root
            / face
            / "matseg_material_identity_report.json"
        )
        contact_path = (
            args.contact_reports_root
            / "reports"
            / face
            / "matseg_material_identity_report.json"
        )
        if not missing_ids or not kept_regions or not original_path.exists() or not contact_path.exists():
            receipts.append(
                {
                    "face": face,
                    "missing_regions": missing_ids,
                    "recovered": [],
                    "reason": "no_eligible_missing_region_or_matseg_evidence",
                }
            )
            continue

        original_index, original_similarity = load_report(original_path)
        contact_index, contact_similarity = load_report(contact_path)
        recovered = []
        for region_id in missing_ids:
            region = source_regions[region_id]
            if not structural(region):
                continue
            if region_id not in original_index or region_id not in contact_index:
                continue

            best: tuple[float, float, int, int] | None = None
            for kept_id, kept in kept_regions.items():
                if kept_id not in original_index or kept_id not in contact_index:
                    continue
                original_value = float(
                    original_similarity[
                        original_index[region_id], original_index[kept_id]
                    ]
                )
                contact_value = float(
                    contact_similarity[contact_index[region_id], contact_index[kept_id]]
                )
                joint = min(original_value, contact_value)
                candidate = (
                    joint,
                    original_value + contact_value,
                    int(kept["material_id"]),
                    kept_id,
                )
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            if best is None:
                continue
            _joint, _sum_similarity, target_material, exemplar_region = best
            original_value = float(
                original_similarity[
                    original_index[region_id], original_index[exemplar_region]
                ]
            )
            contact_value = float(
                contact_similarity[
                    contact_index[region_id], contact_index[exemplar_region]
                ]
            )
            accepted = bool(
                original_value >= ORIGINAL_VIEW_SIMILARITY_FLOOR
                and contact_value >= COMMON_FRAME_SIMILARITY_FLOOR
            )
            decision = {
                "region": region_id,
                "source": str(region.get("source", "")),
                "target_material_id": target_material,
                "target_exemplar_region": exemplar_region,
                "original_view_similarity": original_value,
                "common_frame_similarity": contact_value,
                "original_view_similarity_floor": ORIGINAL_VIEW_SIMILARITY_FLOOR,
                "common_frame_similarity_floor": COMMON_FRAME_SIMILARITY_FLOOR,
                "accepted": accepted,
            }
            if not accepted:
                recovered.append(decision)
                continue
            restored = copy.deepcopy(region)
            restored["material_id"] = target_material
            output_face["regions"].append(restored)
            kept_regions[region_id] = restored
            decision["reason"] = (
                "untraceable_structural_proposal_is_matseg_duplicate_of_"
                "surviving_material_support"
            )
            recovered.append(decision)

        output_face["regions"] = sorted(
            output_face.get("regions", []), key=lambda row: int(row["region"])
        )
        material_ids = sorted(
            {int(region["material_id"]) for region in output_face["regions"]}
        )
        remap = {old: new for new, old in enumerate(material_ids)}
        for region in output_face["regions"]:
            region["material_id"] = remap[int(region["material_id"])]
        output_face["material_count"] = len(material_ids)
        receipts.append(
            {
                "face": face,
                "missing_regions": missing_ids,
                "recovered": recovered,
            }
        )

    result["untraceable_structural_support_recovery"] = {
        "method": "two_frame_matseg_duplicate_support_recovery_v1",
        "new_material_generated": False,
        "geometry_changed": False,
        "room_face_or_material_profile_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    receipt = {
        "method": "two_frame_matseg_duplicate_support_recovery_v1",
        "identity_metadata": str(args.identity_metadata),
        "first_trace_metadata": str(args.first_trace_metadata),
        "output_metadata": str(args.output),
        "output_sha256": sha256(args.output),
        "new_material_generated": False,
        "room_face_or_material_profile_used": False,
        "thresholds": {
            "original_view_similarity": ORIGINAL_VIEW_SIMILARITY_FLOOR,
            "common_frame_similarity": COMMON_FRAME_SIMILARITY_FLOOR,
        },
        "faces": receipts,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
