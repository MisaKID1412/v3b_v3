#!/usr/bin/env python3
"""Resolve face-local material identity from MatSeg without a global cosine cut.

The appearance stage supplies spatial region proposals, not final material
identity.  This module keeps those regions atomic and uses MatSeg only for two
identity operations:

1. reassign a secondary region when another material prototype is a clearly
   better relative match than its current prototype;
2. merge two proposed materials only when their cross-group descriptor spread
   is indistinguishable from their own within-group spread.

Narrow structural territories (for example an automatically detected wall
band) are never erased by an appearance-only merge.  No room name, face count,
material count, or absolute MatSeg similarity threshold is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


RELATIVE_REASSIGN_DISTANCE_RATIO = 0.85
WITHIN_CALIBRATED_MERGE_DISTANCE_RATIO = 1.50
MUTUAL_NEAREST_SEPARATION_RATIO = 0.70
DEFAULT_COMMON_FRAME_STRONG_SIMILARITY = 0.92
STRUCTURAL_COMMON_FRAME_STRONG_SIMILARITY = 0.97


def _canonical(labels: list[int]) -> np.ndarray:
    mapping: dict[int, int] = {}
    result = []
    for label in labels:
        if label not in mapping:
            mapping[label] = len(mapping)
        result.append(mapping[label])
    return np.asarray(result, dtype=np.int32)


def _structural_region(region: dict[str, Any]) -> bool:
    source = str(region.get("source", "")).lower()
    discovery = region.get("discovery_index")
    return (
        "band" in source
        or "stripe" in source
        or (isinstance(discovery, str) and not discovery.lstrip("-+").isdigit())
    )


def _primary_region(region: dict[str, Any]) -> bool:
    return str(region.get("source", "")).lower() == "material_cluster"


def _median_similarity(
    similarity: np.ndarray, index: int, members: list[int]
) -> float | None:
    others = [member for member in members if member != index]
    if not others:
        return None
    return float(np.median(similarity[index, others]))


def _within_distance(
    similarity: np.ndarray,
    members: list[int],
    descriptor_uncertainty: np.ndarray,
) -> float | None:
    if len(members) >= 2:
        values = [
            1.0 - float(similarity[first, second])
            for offset, first in enumerate(members)
            for second in members[offset + 1 :]
        ]
        return float(np.median(values))
    uncertainty = float(descriptor_uncertainty[members[0]])
    return uncertainty if np.isfinite(uncertainty) and uncertainty > 0.0 else None


def resolve_material_ids(
    regions: list[dict[str, Any]],
    similarity: np.ndarray,
    *,
    common_frame_strong_similarity: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return contiguous IDs and a complete, auditable decision record."""
    count = len(regions)
    if similarity.shape != (count, count):
        raise ValueError(
            f"similarity shape {similarity.shape} does not match {count} regions"
        )
    old_labels = _canonical([int(region["material_id"]) for region in regions])
    labels = old_labels.copy()
    structural = np.asarray([_structural_region(region) for region in regions])
    primary = np.asarray([_primary_region(region) for region in regions])
    uncertainty = np.asarray(
        [
            float(value) if value is not None else np.nan
            for region in regions
            for value in [region.get("descriptor_uncertainty_distance", np.nan)]
        ],
        dtype=np.float64,
    )
    reassignments: list[dict[str, Any]] = []

    # Reassignment is evaluated against the original proposals simultaneously;
    # a moved region cannot pull a second region behind it in the same pass.
    proposed = labels.copy()
    for index in range(count):
        if structural[index] or primary[index]:
            continue
        own = int(labels[index])
        own_members = np.flatnonzero(labels == own).tolist()
        own_similarity = _median_similarity(similarity, index, own_members)
        if own_similarity is None:
            continue
        choices = []
        for other in sorted(set(labels.tolist())):
            if other == own:
                continue
            members = np.flatnonzero(labels == other).tolist()
            value = _median_similarity(similarity, index, members + [index])
            if value is not None:
                choices.append((value, int(other)))
        if not choices:
            continue
        best_similarity, best = max(choices)
        own_distance = max(1.0 - own_similarity, np.finfo(np.float32).eps)
        best_distance = max(1.0 - best_similarity, 0.0)
        ratio = best_distance / own_distance
        if ratio < RELATIVE_REASSIGN_DISTANCE_RATIO:
            proposed[index] = best
            reassignments.append(
                {
                    "region": int(regions[index]["region"]),
                    "from_group": own,
                    "to_group": best,
                    "own_prototype_similarity": own_similarity,
                    "other_prototype_similarity": best_similarity,
                    "relative_descriptor_distance_ratio": float(ratio),
                    "reason": "other_prototype_has_decisively_smaller_relative_matseg_distance",
                }
            )
    labels = proposed

    # Conservative agglomeration.  Cross-group distances must fit inside the
    # spread learned from the groups themselves; this replaces the old global
    # 0.87 cosine threshold.  Structural groups carry a hard boundary veto.
    merge_decisions: list[dict[str, Any]] = []
    while True:
        groups = sorted(set(labels.tolist()))
        best_merge: tuple[float, int, int, dict[str, Any]] | None = None
        for offset, first in enumerate(groups):
            first_members = np.flatnonzero(labels == first).tolist()
            for second in groups[offset + 1 :]:
                second_members = np.flatnonzero(labels == second).tolist()
                boundary_veto = bool(
                    np.any(structural[first_members]) or np.any(structural[second_members])
                )
                cross_distances = np.asarray(
                    [
                        1.0 - float(similarity[a, b])
                        for a in first_members
                        for b in second_members
                    ],
                    dtype=np.float64,
                )
                cross_distance = float(np.quantile(cross_distances, 0.75))
                first_scale = _within_distance(
                    similarity, first_members, uncertainty
                )
                second_scale = _within_distance(
                    similarity, second_members, uncertainty
                )
                scales = [value for value in (first_scale, second_scale) if value is not None]
                calibrated_scale = max(scales) if scales else None
                singleton_view_calibrated_duplicate = bool(
                    boundary_veto
                    and len(first_members) == 1
                    and len(second_members) == 1
                    and first_scale is not None
                    and second_scale is not None
                )
                effective_boundary_veto = bool(
                    boundary_veto and not singleton_view_calibrated_duplicate
                )
                accepted = bool(
                    not effective_boundary_veto
                    and calibrated_scale is not None
                    and cross_distance
                    <= WITHIN_CALIBRATED_MERGE_DISTANCE_RATIO * calibrated_scale
                )
                decision = {
                    "first_group": int(first),
                    "second_group": int(second),
                    "first_regions": [int(regions[i]["region"]) for i in first_members],
                    "second_regions": [int(regions[i]["region"]) for i in second_members],
                    "cross_descriptor_distance_q75": cross_distance,
                    "within_calibrated_distance": calibrated_scale,
                    "structural_boundary_veto": boundary_veto,
                    "singleton_view_calibrated_duplicate_override": (
                        singleton_view_calibrated_duplicate
                    ),
                    "effective_structural_boundary_veto": effective_boundary_veto,
                    "accepted": accepted,
                }
                if accepted:
                    normalized = cross_distance / max(
                        float(calibrated_scale), np.finfo(np.float32).eps
                    )
                    if best_merge is None or normalized < best_merge[0]:
                        best_merge = (normalized, first, second, decision)
                else:
                    merge_decisions.append(decision)
        if best_merge is None:
            break
        _, first, second, decision = best_merge
        decision["reason"] = "cross_group_spread_matches_within_group_matseg_spread"
        merge_decisions.append(decision)
        labels[labels == second] = first

    # When Lab is used only to over-segment, one physical material can still
    # remain as several internally coherent proposal groups.  Collapse a pair
    # only when they are mutual nearest neighbours and their MatSeg distance is
    # separated from every alternative by a clear relative gap.  Requiring at
    # least three non-structural groups prevents the ambiguous two-group case
    # (often a real upper/lower wall boundary) from being guessed away.
    separated_neighbour_merges: list[dict[str, Any]] = []
    while True:
        groups = sorted(set(labels.tolist()))
        nonstructural_groups = [
            group
            for group in groups
            if not np.any(structural[np.flatnonzero(labels == group)])
        ]
        if len(nonstructural_groups) < 3:
            break
        distances: dict[tuple[int, int], float] = {}
        for offset, first in enumerate(nonstructural_groups):
            first_members = np.flatnonzero(labels == first).tolist()
            for second in nonstructural_groups[offset + 1 :]:
                second_members = np.flatnonzero(labels == second).tolist()
                values = [
                    1.0 - float(similarity[a, b])
                    for a in first_members
                    for b in second_members
                ]
                distances[(first, second)] = float(np.median(values))

        ordered: dict[int, list[tuple[float, int]]] = {}
        for group in nonstructural_groups:
            rows = []
            for other in nonstructural_groups:
                if group == other:
                    continue
                key = (min(group, other), max(group, other))
                rows.append((distances[key], other))
            ordered[group] = sorted(rows)

        candidates = []
        for first in nonstructural_groups:
            pair_distance, second = ordered[first][0]
            if ordered[second][0][1] != first or first > second:
                continue
            first_alternative = ordered[first][1][0]
            second_alternative = ordered[second][1][0]
            first_ratio = pair_distance / max(
                first_alternative, np.finfo(np.float32).eps
            )
            second_ratio = pair_distance / max(
                second_alternative, np.finfo(np.float32).eps
            )
            separation_ratio = max(first_ratio, second_ratio)
            strong_enough = bool(
                common_frame_strong_similarity is None
                or pair_distance <= 1.0 - common_frame_strong_similarity
            )
            if (
                separation_ratio < MUTUAL_NEAREST_SEPARATION_RATIO
                and strong_enough
            ):
                candidates.append(
                    (
                        separation_ratio,
                        pair_distance,
                        first,
                        second,
                        first_alternative,
                        second_alternative,
                    )
                )
        if not candidates:
            break
        (
            separation_ratio,
            pair_distance,
            first,
            second,
            first_alternative,
            second_alternative,
        ) = min(candidates)
        first_members = np.flatnonzero(labels == first).tolist()
        second_members = np.flatnonzero(labels == second).tolist()
        separated_neighbour_merges.append(
            {
                "first_group": int(first),
                "second_group": int(second),
                "first_regions": [int(regions[i]["region"]) for i in first_members],
                "second_regions": [int(regions[i]["region"]) for i in second_members],
                "pair_descriptor_distance": float(pair_distance),
                "first_next_alternative_distance": float(first_alternative),
                "second_next_alternative_distance": float(second_alternative),
                "relative_separation_ratio": float(separation_ratio),
                "reason": "mutual_nearest_matseg_groups_separated_from_all_alternatives",
            }
        )
        labels[labels == second] = first

    # A MatSeg contact sheet places candidates traced from different source
    # views in one forward pass, where cosine scores share the model's intended
    # coordinate frame.  Strong residual matches can therefore coalesce even
    # the final two groups.  This is deliberately disabled for legacy reports
    # whose descriptors came from separate network calls.
    common_frame_strong_merges: list[dict[str, Any]] = []
    if common_frame_strong_similarity is not None:
        max_distance = 1.0 - float(common_frame_strong_similarity)
        while True:
            groups = sorted(set(labels.tolist()))
            candidate = None
            for offset, first in enumerate(groups):
                first_members = np.flatnonzero(labels == first).tolist()
                for second in groups[offset + 1 :]:
                    second_members = np.flatnonzero(labels == second).tolist()
                    structural_pair = bool(
                        np.any(structural[first_members])
                        or np.any(structural[second_members])
                    )
                    distances = [
                        1.0 - float(similarity[a, b])
                        for a in first_members
                        for b in second_members
                    ]
                    robust_distance = float(np.quantile(distances, 0.75))
                    pair_max_distance = (
                        1.0 - STRUCTURAL_COMMON_FRAME_STRONG_SIMILARITY
                        if structural_pair
                        else max_distance
                    )
                    if robust_distance <= pair_max_distance and (
                        candidate is None or robust_distance < candidate[0]
                    ):
                        candidate = (
                            robust_distance,
                            first,
                            second,
                            first_members,
                            second_members,
                        )
            if candidate is None:
                break
            robust_distance, first, second, first_members, second_members = candidate
            common_frame_strong_merges.append(
                {
                    "first_group": int(first),
                    "second_group": int(second),
                    "first_regions": [int(regions[i]["region"]) for i in first_members],
                    "second_regions": [int(regions[i]["region"]) for i in second_members],
                    "cross_descriptor_distance_q75": float(robust_distance),
                    "common_frame_similarity_floor": float(
                        STRUCTURAL_COMMON_FRAME_STRONG_SIMILARITY
                        if (
                            np.any(structural[first_members])
                            or np.any(structural[second_members])
                        )
                        else common_frame_strong_similarity
                    ),
                    "reason": "strong_matseg_match_inside_one_common_forward_pass",
                }
            )
            labels[labels == second] = first

    labels = _canonical(labels.tolist())
    audit = {
        "rule": "relative_matseg_identity_with_structural_boundary_veto_v2",
        "absolute_similarity_threshold_used": bool(
            common_frame_strong_similarity is not None
        ),
        "common_frame_similarity_floor": common_frame_strong_similarity,
        "room_or_face_profile_used": False,
        "initial_material_count": int(len(np.unique(old_labels))),
        "resolved_material_count": int(len(np.unique(labels))),
        "reassignments": reassignments,
        "merge_decisions": merge_decisions,
        "separated_neighbour_merges": separated_neighbour_merges,
        "common_frame_strong_merges": common_frame_strong_merges,
        "structural_regions": [
            int(regions[index]["region"])
            for index in range(count)
            if structural[index]
        ],
    }
    return labels, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--common-frame-strong-similarity", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    report = json.loads(args.input_report.read_text(encoding="utf-8"))
    face_record = next(item for item in metadata["stats"] if item["face"] == report["face"])
    regions = sorted(face_record["regions"], key=lambda item: int(item["region"]))
    report_regions = sorted(report["regions"], key=lambda item: int(item["region"]))
    if [int(item["region"]) for item in regions] != [
        int(item["region"]) for item in report_regions
    ]:
        raise RuntimeError("metadata/report region mismatch")
    enriched = []
    for source, measured in zip(regions, report_regions):
        row = dict(source)
        row.update(
            {
                key: measured[key]
                for key in ("descriptor_uncertainty_distance",)
                if key in measured
            }
        )
        enriched.append(row)
    labels, audit = resolve_material_ids(
        enriched,
        np.asarray(report["similarity_matrix"], dtype=np.float64),
        common_frame_strong_similarity=args.common_frame_strong_similarity,
    )
    report["method"] = "matseg_region_material_identity_relative_structural_v2"
    report.pop("similarity_threshold", None)
    report["material_count"] = int(len(np.unique(labels)))
    report["identity_resolution"] = audit
    for row, label, source in zip(report_regions, labels.tolist(), regions):
        row["old_material_id"] = int(source["material_id"])
        row["matseg_material_id"] = int(label)
        row["source"] = str(source.get("source", ""))
        row["discovery_index"] = source.get("discovery_index")
    report["regions"] = report_regions
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
