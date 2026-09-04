#!/usr/bin/env python3
"""Replace only v3b material IDs/counts from MatSeg identity reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    original = json.loads(args.metadata.read_text(encoding="utf-8"))
    corrected = json.loads(args.metadata.read_text(encoding="utf-8"))
    face_stats = {str(item["face"]): item for item in corrected["stats"]}
    receipt_faces = []

    for report_path in args.reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("method") != "matseg_region_material_identity_threshold_v1":
            raise RuntimeError(f"Unexpected MatSeg report method: {report_path}")
        face = str(report["face"])
        face_stat = face_stats[face]
        regions = {int(item["region"]): item for item in face_stat["regions"]}
        report_regions = {int(item["region"]): item for item in report["regions"]}
        if set(regions) != set(report_regions):
            raise RuntimeError(f"Region mismatch for {face}")

        changes = []
        new_ids = set()
        for region_id in sorted(regions):
            region = regions[region_id]
            old_id = int(region["material_id"])
            new_id = int(report_regions[region_id]["matseg_material_id"])
            new_ids.add(new_id)
            region["material_id"] = new_id
            if old_id != new_id:
                changes.append(
                    {
                        "region": region_id,
                        "old_material_id": old_id,
                        "new_material_id": new_id,
                    }
                )
        expected_ids = set(range(len(new_ids)))
        if new_ids != expected_ids:
            raise RuntimeError(f"Non-contiguous MatSeg material IDs for {face}: {new_ids}")
        old_count = int(face_stat["material_count"])
        face_stat["material_count"] = len(new_ids)
        receipt_faces.append(
            {
                "face": face,
                "old_material_count": old_count,
                "new_material_count": len(new_ids),
                "similarity_threshold": float(report["similarity_threshold"]),
                "changes": changes,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corrected, indent=2), encoding="utf-8")
    receipt = {
        "contract": "only region.material_id and face.material_count may change",
        "source_metadata": str(args.metadata.resolve()),
        "source_sha256": sha256(args.metadata),
        "output_metadata": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "faces": receipt_faces,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
