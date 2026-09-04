#!/usr/bin/env python3
"""Fail unless MatSeg changed only material identity fields in v3b metadata."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original = json.loads(args.original.read_text(encoding="utf-8"))
    corrected = json.loads(args.corrected.read_text(encoding="utf-8"))
    normalized = copy.deepcopy(corrected)
    original_faces = {str(item["face"]): item for item in original["stats"]}
    normalized_faces = {str(item["face"]): item for item in normalized["stats"]}
    if set(original_faces) != set(normalized_faces):
        raise RuntimeError("MatSeg changed the face set")

    changes = []
    for face in sorted(original_faces):
        original_face = original_faces[face]
        normalized_face = normalized_faces[face]
        original_regions = {int(item["region"]): item for item in original_face["regions"]}
        normalized_regions = {int(item["region"]): item for item in normalized_face["regions"]}
        if set(original_regions) != set(normalized_regions):
            raise RuntimeError(f"MatSeg changed the region set for {face}")
        face_changes = []
        for region_id in sorted(original_regions):
            old_id = int(original_regions[region_id]["material_id"])
            new_id = int(normalized_regions[region_id]["material_id"])
            if old_id != new_id:
                face_changes.append(
                    {
                        "region": region_id,
                        "old_material_id": old_id,
                        "new_material_id": new_id,
                    }
                )
            normalized_regions[region_id]["material_id"] = old_id
        normalized_face["material_count"] = original_face["material_count"]
        changes.append(
            {
                "face": face,
                "old_material_count": int(original_face["material_count"]),
                "new_material_count": int(corrected["stats"][
                    next(i for i, item in enumerate(corrected["stats"]) if str(item["face"]) == face)
                ]["material_count"]),
                "region_changes": face_changes,
            }
        )

    if normalized != original:
        raise RuntimeError(
            "MatSeg identity-only contract failed: fields outside material_id/material_count changed"
        )
    report = {
        "status": "material_identity_only_contract_verified",
        "allowed_fields": ["stats[].regions[].material_id", "stats[].material_count"],
        "all_other_metadata_identical": True,
        "changes": changes,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
