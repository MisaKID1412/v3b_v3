#!/usr/bin/env python3
"""Resolve material identity from original-view MatSeg evidence.

Region proposals and their geometric boundaries remain untouched.  MatSeg is
used only to decide whether those proposals represent the same material.  The
primary descriptor matrix is measured at reverse-projected pixels in the
original photographs.  A contact-sheet matrix may only corroborate the
otherwise under-determined case of two non-structural singleton proposals.

No room name, face name, material count profile, or dataset-specific branch is
used.  The output contract permits changes only to ``region.material_id`` and
``face.material_count``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from matseg_identity_resolver import resolve_material_ids


ORIGINAL_VIEW_SINGLETON_SIMILARITY = 0.87
COMMON_FRAME_SINGLETON_SIMILARITY = 0.92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--original-view-reports-root", type=Path, required=True)
    parser.add_argument("--contact-reports-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports-output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(labels: list[int]) -> np.ndarray:
    mapping: dict[int, int] = {}
    result = []
    for label in labels:
        mapping.setdefault(int(label), len(mapping))
        result.append(mapping[int(label)])
    return np.asarray(result, dtype=np.int32)


def structural(region: dict[str, Any]) -> bool:
    source = str(region.get("source", "")).lower()
    discovery = region.get("discovery_index")
    return bool(
        "band" in source
        or "stripe" in source
        or (isinstance(discovery, str) and not discovery.lstrip("-+").isdigit())
    )


def load_subset_matrix(
    report_path: Path, region_ids: list[int]
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_regions = sorted(report["regions"], key=lambda row: int(row["region"]))
    index = {int(row["region"]): offset for offset, row in enumerate(report_regions)}
    missing = sorted(set(region_ids) - set(index))
    if missing:
        raise RuntimeError(f"{report_path}: missing regions {missing}")
    offsets = [index[region_id] for region_id in region_ids]
    matrix = np.asarray(report["similarity_matrix"], dtype=np.float64)
    expected = (len(report_regions), len(report_regions))
    if matrix.shape != expected:
        raise RuntimeError(
            f"{report_path}: similarity shape {matrix.shape}, expected {expected}"
        )
    subset = matrix[np.ix_(offsets, offsets)]
    measured = [report_regions[offset] for offset in offsets]
    return report, subset, measured


def maybe_merge_two_corroborated_singletons(
    regions: list[dict[str, Any]],
    labels: np.ndarray,
    original_similarity: np.ndarray,
    contact_similarity: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Resolve the only case whose within-group spread cannot be estimated.

    Two singleton Lab proposals have no same-group descriptor baseline.  They
    are merged only when two independently constructed MatSeg measurements
    agree: features sampled in the original photographs and features measured
    together in a common contact-sheet forward pass.
    """
    groups = sorted(set(labels.tolist()))
    if len(groups) != 2 or len(regions) != 2 or contact_similarity is None:
        return labels, None
    if any(structural(region) for region in regions):
        return labels, None
    if any(int(np.count_nonzero(labels == group)) != 1 for group in groups):
        return labels, None
    original_value = float(original_similarity[0, 1])
    contact_value = float(contact_similarity[0, 1])
    accepted = bool(
        original_value >= ORIGINAL_VIEW_SINGLETON_SIMILARITY
        and contact_value >= COMMON_FRAME_SINGLETON_SIMILARITY
    )
    decision = {
        "regions": [int(region["region"]) for region in regions],
        "original_view_similarity": original_value,
        "original_view_similarity_floor": ORIGINAL_VIEW_SINGLETON_SIMILARITY,
        "common_frame_similarity": contact_value,
        "common_frame_similarity_floor": COMMON_FRAME_SINGLETON_SIMILARITY,
        "accepted": accepted,
        "reason": (
            "two_independent_matseg_frames_agree_singletons_are_same_material"
            if accepted
            else "independent_matseg_frames_do_not_both_support_singleton_merge"
        ),
    }
    if accepted:
        labels = np.zeros_like(labels)
    return labels, decision


def main() -> None:
    args = parse_args()
    source = json.loads(args.metadata.read_text(encoding="utf-8"))
    result = copy.deepcopy(source)
    args.reports_output_dir.mkdir(parents=True, exist_ok=True)
    face_receipts = []

    for face_record in result["stats"]:
        face = str(face_record["face"])
        regions = sorted(face_record["regions"], key=lambda row: int(row["region"]))
        region_ids = [int(region["region"]) for region in regions]
        original_count = int(face_record["material_count"])
        if original_count <= 1:
            labels = np.zeros(len(regions), dtype=np.int32)
            audit: dict[str, Any] = {
                "rule": "single_existing_material_identity_is_already_resolved",
                "room_or_face_profile_used": False,
                "initial_material_count": original_count,
                "resolved_material_count": original_count,
            }
            source_report = None
            original_similarity = np.eye(len(regions), dtype=np.float64)
            measured_regions: list[dict[str, Any]] = []
            singleton_decision = None
        else:
            original_path = (
                args.original_view_reports_root
                / face
                / "matseg_material_identity_report.json"
            )
            if not original_path.exists():
                raise FileNotFoundError(
                    f"multi-material face {face} needs original-view MatSeg report: "
                    f"{original_path}"
                )
            source_report, original_similarity, measured_regions = load_subset_matrix(
                original_path, region_ids
            )
            enriched = []
            for region, measured in zip(regions, measured_regions):
                row = dict(region)
                # Legacy original-view reports do not expose this uncertainty;
                # leaving it unknown prevents an uncalibrated absolute merge.
                if "descriptor_uncertainty_distance" in measured:
                    row["descriptor_uncertainty_distance"] = measured[
                        "descriptor_uncertainty_distance"
                    ]
                enriched.append(row)
            labels, audit = resolve_material_ids(enriched, original_similarity)

            contact_similarity = None
            contact_path = None
            if args.contact_reports_root is not None:
                contact_path = (
                    args.contact_reports_root
                    / "reports"
                    / face
                    / "matseg_material_identity_report.json"
                )
                if contact_path.exists():
                    _contact_report, contact_similarity, _contact_regions = (
                        load_subset_matrix(contact_path, region_ids)
                    )
            labels, singleton_decision = maybe_merge_two_corroborated_singletons(
                enriched, labels, original_similarity, contact_similarity
            )
            labels = canonical(labels.tolist())
            audit["corroborated_two_singleton_decision"] = singleton_decision
            audit["resolved_material_count"] = int(len(set(labels.tolist())))
            audit["descriptor_frame"] = "reverse_projected_original_photographs"
            audit["contact_sheet_role"] = "corroboration_only_for_two_singletons"
            audit["source_original_view_report"] = str(original_path)
            audit["source_contact_report"] = (
                str(contact_path) if contact_path is not None and contact_path.exists() else None
            )

        changes = []
        for region, label in zip(regions, labels.tolist()):
            old_id = int(region["material_id"])
            region["material_id"] = int(label)
            if old_id != int(label):
                changes.append(
                    {
                        "region": int(region["region"]),
                        "old_material_id": old_id,
                        "new_material_id": int(label),
                    }
                )
        face_record["regions"] = regions
        face_record["material_count"] = int(len(set(labels.tolist())))

        output_report = {
            "method": "matseg_original_view_identity_with_common_frame_corroboration_v1",
            "face": face,
            "material_count": face_record["material_count"],
            "regions": [
                {
                    "region": int(region["region"]),
                    "old_material_id": int(source_region["material_id"]),
                    "matseg_material_id": int(label),
                    "source": str(region.get("source", "")),
                    "discovery_index": region.get("discovery_index"),
                }
                for region, source_region, label in zip(
                    regions,
                    sorted(
                        next(
                            item for item in source["stats"] if item["face"] == face
                        )["regions"],
                        key=lambda row: int(row["region"]),
                    ),
                    labels.tolist(),
                )
            ],
            "similarity_matrix": original_similarity.tolist(),
            "identity_resolution": audit,
        }
        report_path = args.reports_output_dir / face / "identity_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(output_report, indent=2), encoding="utf-8")
        face_receipts.append(
            {
                "face": face,
                "old_material_count": original_count,
                "new_material_count": face_record["material_count"],
                "changes": changes,
                "identity_resolution": audit,
                "report": str(report_path),
            }
        )

    result["material_identity_stage"] = {
        "method": "original_view_matseg_identity_with_common_frame_corroboration_v1",
        "geometry_or_region_boundaries_changed": False,
        "room_or_face_profile_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    receipt = {
        "contract": "only region.material_id and face.material_count may change",
        "method": "original_view_matseg_identity_with_common_frame_corroboration_v1",
        "source_metadata": str(args.metadata.resolve()),
        "source_sha256": sha256(args.metadata),
        "output_metadata": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "room_or_face_profile_used": False,
        "thresholds": {
            "original_view_singleton_similarity": ORIGINAL_VIEW_SINGLETON_SIMILARITY,
            "common_frame_singleton_similarity": COMMON_FRAME_SINGLETON_SIMILARITY,
        },
        "faces": face_receipts,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
