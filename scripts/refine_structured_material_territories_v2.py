#!/usr/bin/env python3
"""Fit simple material territories from MatSeg-backed strict projection seeds.

This stage deliberately does not reclassify completed-observed RGB/Lab values.
MatSeg decides material identity upstream; here the projected strict supports are
only regularized into plausible architectural boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, median_filter


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
            "Regularize material territories using strict MatSeg-backed support seeds only; "
            "completed-observed Lab colors never create or relabel seeds."
        )
    )
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--source-package-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--faces", nargs="*", default=None)
    parser.add_argument("--min-axis-accuracy", type=float, default=0.94)
    parser.add_argument("--min-axis-class-accuracy", type=float, default=0.88)
    parser.add_argument("--min-tangent-span", type=float, default=0.16)
    parser.add_argument("--wall-horizontal-min-tangent-span", type=float, default=0.08)
    parser.add_argument("--min-normal-separation", type=float, default=0.10)
    parser.add_argument("--curve-max-deviation-frac", type=float, default=0.055)
    parser.add_argument("--curve-smooth-frac", type=float, default=0.035)
    parser.add_argument("--curve-min-accuracy-gain", type=float, default=0.012)
    parser.add_argument("--lowfreq-boundary-feather-frac", type=float, default=0.006)
    parser.add_argument("--upstream-structured-min-accuracy", type=float, default=0.84)
    parser.add_argument(
        "--upstream-structured-min-class-accuracy", type=float, default=0.68
    )
    parser.add_argument("--upstream-structured-min-material-fraction", type=float, default=0.0015)
    return parser.parse_args()


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def label_rgb(labels: np.ndarray) -> np.ndarray:
    safe = np.maximum(labels, 0)
    return LABEL_COLORS[safe % len(LABEL_COLORS)]


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(path)


def fill_nearest(known_labels: np.ndarray, known: np.ndarray) -> np.ndarray:
    if not np.any(known):
        return np.zeros(known.shape, dtype=np.int16)
    _, indices = distance_transform_edt(~known, return_indices=True)
    labels = known_labels.copy()
    labels[~known] = known_labels[indices[0][~known], indices[1][~known]]
    return labels.astype(np.int16)


def seed_stats(
    labels: np.ndarray,
    known_labels: np.ndarray,
    known: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    active = known & valid
    if not np.any(active):
        return {
            "seed_accuracy": 0.0,
            "seed_min_class_accuracy": 0.0,
            "valid_boundary_complexity": 1.0,
        }
    values = known_labels[active]
    predicted = labels[active]
    class_accuracy = []
    for material in np.unique(values):
        selected = values == material
        class_accuracy.append(float(np.mean(predicted[selected] == material)))
    horizontal = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical = (labels[1:, :] != labels[:-1, :]) & valid[1:, :] & valid[:-1, :]
    denom = max(int(np.count_nonzero(valid[:, 1:] & valid[:, :-1])) + int(np.count_nonzero(valid[1:, :] & valid[:-1, :])), 1)
    return {
        "seed_accuracy": float(np.mean(predicted == values)),
        "seed_min_class_accuracy": float(min(class_accuracy)),
        "valid_boundary_complexity": float((np.count_nonzero(horizontal) + np.count_nonzero(vertical)) / denom),
    }


def axis_candidate(
    face: str,
    known_labels: np.ndarray,
    known: np.ndarray,
    valid: np.ndarray,
    axis: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    active = known & valid
    ys, xs = np.nonzero(active)
    values = known_labels[active]
    if np.unique(values).size != 2:
        return None, {"method": "axis", "reason": "not_two_materials"}
    h, w = known.shape
    normal = ys if axis == 0 else xs
    tangent = xs if axis == 0 else ys
    normal_size = h if axis == 0 else w
    tangent_size = w if axis == 0 else h
    axis_name = "y" if axis == 0 else "x"
    unique = np.sort(np.unique(values))
    tangent_spans = []
    normal_medians = []
    for label in unique:
        selected = values == label
        if np.count_nonzero(selected) < 64:
            return None, {"method": "axis", "axis": axis_name, "reason": "small_seed_class"}
        tangent_spans.append(
            float((np.percentile(tangent[selected], 95) - np.percentile(tangent[selected], 5)) / max(tangent_size - 1, 1))
        )
        normal_medians.append(float(np.median(normal[selected]) / max(normal_size - 1, 1)))
    tangent_span = float(min(tangent_spans))
    normal_separation = float(abs(normal_medians[0] - normal_medians[1]))
    min_tangent_span = (
        args.wall_horizontal_min_tangent_span
        if face.startswith("wall_") and axis == 0
        else args.min_tangent_span
    )
    if tangent_span < min_tangent_span or normal_separation < args.min_normal_separation:
        return None, {
            "method": "axis",
            "axis": axis_name,
            "reason": "insufficient_axis_evidence",
            "tangent_span_min": tangent_span,
            "required_tangent_span": float(min_tangent_span),
            "normal_median_separation": normal_separation,
        }

    best: tuple[float, int, int, dict[str, float]] | None = None
    thresholds = np.arange(normal_size, dtype=np.int32)
    counts = {
        int(label): np.bincount(normal[values == label], minlength=normal_size).astype(np.float64)
        for label in unique
    }
    prefix_before = {
        label: np.concatenate([[0.0], np.cumsum(counts[label])])[:-1]
        for label in counts
    }
    for positive_value in unique:
        positive_label = int(positive_value)
        negative_label = int(unique[0] if positive_value == unique[1] else unique[1])
        correct_negative = prefix_before[negative_label]
        correct_positive = float(np.sum(counts[positive_label])) - prefix_before[positive_label]
        accuracy = (correct_negative + correct_positive) / max(float(values.size), 1.0)
        negative_accuracy = correct_negative / max(float(np.sum(counts[negative_label])), 1.0)
        positive_accuracy = correct_positive / max(float(np.sum(counts[positive_label])), 1.0)
        minimum_accuracy = np.minimum(negative_accuracy, positive_accuracy)
        scores = accuracy + 0.30 * minimum_accuracy
        selected = int(np.argmax(scores))
        score = float(scores[selected])
        if best is None or score > best[0]:
            best = (
                score,
                int(thresholds[selected]),
                positive_label,
                {
                    "seed_accuracy": float(accuracy[selected]),
                    "seed_min_class_accuracy": float(minimum_accuracy[selected]),
                },
            )
    assert best is not None
    _, threshold, positive_label, fit = best
    negative_label = int(unique[0] if positive_label == unique[1] else unique[1])
    yy, xx = np.indices(known.shape)
    coordinate = yy if axis == 0 else xx
    labels = np.where(coordinate >= threshold, positive_label, negative_label).astype(np.int16)
    if fit["seed_accuracy"] < args.min_axis_accuracy or fit["seed_min_class_accuracy"] < args.min_axis_class_accuracy:
        return None, {
            "method": "axis",
            "axis": axis_name,
            "reason": "failed_seed_validation",
            **fit,
            "tangent_span_min": tangent_span,
            "normal_median_separation": normal_separation,
        }
    stats = {
        "method": "axis_boundary_strict_seed",
        "axis": axis_name,
        "threshold": threshold,
        "positive_side_label": positive_label,
        "tangent_span_min": tangent_span,
        "normal_median_separation": normal_separation,
        **fit,
    }
    stats.update(seed_stats(labels, known_labels, known, valid))
    return labels, stats


def seed_distance_curve(
    base_labels: np.ndarray,
    base_stats: dict[str, Any],
    known_labels: np.ndarray,
    known: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    axis = 0 if base_stats["axis"] == "y" else 1
    h, w = known.shape
    normal_size = h if axis == 0 else w
    tangent_size = w if axis == 0 else h
    positive = int(base_stats["positive_side_label"])
    negative = int(1 - positive)
    positive_seed = known & valid & (known_labels == positive)
    negative_seed = known & valid & (known_labels == negative)
    distance_positive = distance_transform_edt(~positive_seed)
    distance_negative = distance_transform_edt(~negative_seed)
    signed = distance_negative - distance_positive
    if axis == 1:
        signed = signed.T
    base = int(base_stats["threshold"])
    deviation = max(2, int(round(args.curve_max_deviation_frac * normal_size)))
    low = max(1, base - deviation)
    high = min(normal_size - 2, base + deviation)
    curves = np.full(tangent_size, float(base), dtype=np.float32)
    confidence = np.zeros(tangent_size, dtype=np.float32)
    for tangent in range(tangent_size):
        line = signed[low : high + 1, tangent]
        if line.size == 0:
            continue
        # The zero of the two strict-seed distance fields is the geometric
        # bisector. No RGB/Lab value is consulted here.
        index = int(np.argmin(np.abs(line)))
        curves[tangent] = float(low + index)
        confidence[tangent] = float(1.0 / (1.0 + abs(float(line[index]))))
    kernel = max(3, int(round(args.curve_smooth_frac * tangent_size)))
    if kernel % 2 == 0:
        kernel += 1
    kernel = min(kernel, tangent_size if tangent_size % 2 else max(1, tangent_size - 1))
    if kernel >= 3:
        curves = median_filter(curves, size=kernel, mode="nearest")
    sigma = max(0.8, args.curve_smooth_frac * tangent_size)
    curves = cv2.GaussianBlur(curves.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
    curves = np.clip(curves, low, high)
    yy, xx = np.indices((h, w))
    if axis == 0:
        labels = np.where(yy >= curves[xx], positive, negative).astype(np.int16)
    else:
        labels = np.where(xx >= curves[yy], positive, negative).astype(np.int16)
    stats: dict[str, Any] = {
        "method": "axis_curve_strict_seed_distance",
        "axis": base_stats["axis"],
        "base_threshold": base,
        "positive_side_label": positive,
        "curve_min": float(np.min(curves)),
        "curve_max": float(np.max(curves)),
        "curve_std": float(np.std(curves)),
        "curve_mean_seed_distance_confidence": float(np.mean(confidence)),
    }
    stats.update(seed_stats(labels, known_labels, known, valid))
    return labels, stats


def select_candidate(
    face: str,
    known_labels: np.ndarray,
    valid: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    known = (known_labels >= 0) & valid
    materials = np.unique(known_labels[known]) if np.any(known) else np.empty(0, dtype=np.int16)
    if materials.size <= 1:
        label = int(materials[0]) if materials.size else 0
        labels = np.full(known.shape, label, dtype=np.int16)
        return labels, {
            "method": "single_material_strict_seed",
            "strict_seed_fraction": float(np.mean(known)),
            "geometry_valid_fraction": float(np.mean(valid)),
            "candidate_solutions": [],
        }
    nearest = fill_nearest(known_labels, known)
    nearest_stats: dict[str, Any] = {"method": "nearest_strict_seed"}
    nearest_stats.update(seed_stats(nearest, known_labels, known, valid))
    candidates: list[tuple[np.ndarray, dict[str, Any]]] = [(nearest, nearest_stats)]
    if materials.size == 2 and np.array_equal(materials, np.asarray([0, 1], dtype=materials.dtype)):
        for axis in (0, 1):
            labels_axis, stats_axis = axis_candidate(face, known_labels, known, valid, axis, args)
            if labels_axis is None:
                nearest_stats.setdefault("rejected_structured_candidates", []).append(stats_axis)
                continue
            candidates.append((labels_axis, stats_axis))
            curve_labels, curve_stats = seed_distance_curve(
                labels_axis, stats_axis, known_labels, known, valid, args
            )
            if (
                curve_stats["seed_accuracy"] >= stats_axis["seed_accuracy"] + args.curve_min_accuracy_gain
                and curve_stats["seed_min_class_accuracy"] >= args.min_axis_class_accuracy
            ):
                candidates.append((curve_labels, curve_stats))

    structured = [item for item in candidates if item[1]["method"].startswith("axis_")]
    if structured:
        # Architectural Occam rule: after strict-seed validation, select the
        # simplest well-fitting primitive. A curve is only present after it
        # earned a clear fit gain; a wall-horizontal prior is weak enough that
        # a genuinely vertical column boundary can still win.
        def structured_score(item: tuple[np.ndarray, dict[str, Any]]) -> float:
            stats = item[1]
            score = float(stats["seed_accuracy"]) + 0.30 * float(stats["seed_min_class_accuracy"])
            if "curve" in stats["method"]:
                score -= 0.004
            if face.startswith("wall_") and stats.get("axis") == "x":
                score -= 0.012
            return score

        structured.sort(
            key=lambda item: -structured_score(item)
        )
        selected_labels, selected_stats = structured[0]
    else:
        selected_labels, selected_stats = nearest, nearest_stats
    selected_stats = dict(selected_stats)
    selected_stats["strict_seed_fraction"] = float(np.mean(known))
    selected_stats["geometry_valid_fraction"] = float(np.mean(valid))
    selected_stats["identity_evidence"] = "MatSeg-grouped strict projected supports only"
    selected_stats["rgb_lab_reclassification_used"] = False
    selected_stats["candidate_solutions"] = [
        {key: value for key, value in stats.items() if isinstance(value, (str, int, float, bool))}
        for _, stats in candidates
    ]
    return selected_labels, selected_stats


def lowfreq_weights(labels: np.ndarray, count: int, valid: np.ndarray, radius: float) -> np.ndarray:
    hard = np.stack([(labels == index).astype(np.float32) for index in range(count)], axis=0)
    if count <= 1 or radius <= 0:
        return hard
    blurred = np.stack(
        [cv2.GaussianBlur(channel, (0, 0), sigmaX=radius, sigmaY=radius) for channel in hard],
        axis=0,
    )
    blurred *= valid[None, ...].astype(np.float32)
    denom = np.sum(blurred, axis=0, keepdims=True)
    return np.where(denom > 1e-6, blurred / np.maximum(denom, 1e-6), hard).astype(np.float32)


def retain_validated_upstream_structure(
    old_labels: np.ndarray,
    known_labels: np.ndarray,
    valid: np.ndarray,
    selected_labels: np.ndarray,
    selected_stats: dict[str, Any],
    upstream_stats: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep a validated architectural primitive that sparse seeds cannot refit.

    The upstream v3b layout can fit arbitrary-N horizontal/vertical layers,
    while this stricter refinement stage only constructs new two-material axis
    candidates.  Replacing an already validated N-layer layout with nearest
    seed fill creates the jagged boundaries this stage exists to remove.  The
    existing layout is therefore an eligible candidate, never an instruction:
    current MatSeg-grouped strict supports must independently validate it.
    """
    selected_method = str(selected_stats.get("method", ""))
    upstream_method = str(upstream_stats.get("method", ""))
    structured_upstream = bool(
        upstream_method.startswith("axis_")
        or upstream_method in {"axis_layers", "linear_split", "linear_layers"}
    )
    if selected_method != "nearest_strict_seed" or not structured_upstream:
        return selected_labels, selected_stats

    known = (known_labels >= 0) & valid
    if not np.any(known):
        return selected_labels, selected_stats
    materials = np.unique(known_labels[known])
    upstream_candidate = old_labels
    retained_method = upstream_method
    # The upstream color-aware stage may bend a strong global axis into a
    # curve.  If sparse MatSeg supports cannot fit their own axis candidate,
    # first test the simpler base boundary recorded by that curve.  It is kept
    # only when those current supports validate it independently.
    if (
        upstream_method == "axis_curve_boundary"
        and materials.size == 2
        and upstream_stats.get("axis") in {"x", "y"}
        and upstream_stats.get("base_threshold") is not None
        and upstream_stats.get("positive_side_label") is not None
    ):
        axis = 0 if upstream_stats["axis"] == "y" else 1
        threshold = int(upstream_stats["base_threshold"])
        positive = int(upstream_stats["positive_side_label"])
        other = [int(value) for value in materials if int(value) != positive]
        if len(other) == 1:
            coordinate = np.indices(old_labels.shape)[axis]
            upstream_candidate = np.where(
                coordinate >= threshold, positive, other[0]
            ).astype(np.int16)
            retained_method = "axis_boundary_from_upstream_curve_base"

    evidence = seed_stats(upstream_candidate, known_labels, known, valid)
    valid_count = max(int(np.count_nonzero(valid)), 1)
    fractions = [
        float(np.count_nonzero((upstream_candidate == material) & valid) / valid_count)
        for material in materials
    ]
    labels_match = bool(
        materials.size >= 2
        and set(np.unique(upstream_candidate[valid]).tolist()) == set(materials.tolist())
    )
    accepted = bool(
        labels_match
        and evidence["seed_accuracy"] >= args.upstream_structured_min_accuracy
        and evidence["seed_min_class_accuracy"]
        >= args.upstream_structured_min_class_accuracy
        and fractions
        and min(fractions) >= args.upstream_structured_min_material_fraction
    )
    decision = {
        "upstream_method": upstream_method,
        "candidate_method": retained_method,
        "seed_accuracy": evidence["seed_accuracy"],
        "seed_min_class_accuracy": evidence["seed_min_class_accuracy"],
        "material_fractions": fractions,
        "labels_match_strict_material_set": labels_match,
        "min_accuracy": args.upstream_structured_min_accuracy,
        "min_class_accuracy": args.upstream_structured_min_class_accuracy,
        "min_material_fraction": args.upstream_structured_min_material_fraction,
        "accepted": accepted,
        "room_face_or_material_profile_used": False,
    }
    if not accepted:
        selected_stats = dict(selected_stats)
        selected_stats["upstream_structured_candidate"] = decision
        return selected_labels, selected_stats

    retained = dict(evidence)
    retained.update(
        {
            "method": "validated_upstream_architectural_structure",
            "retained_upstream_method": retained_method,
            "upstream_structured_candidate": decision,
            "reason": (
                "existing_axis_primitive_is_validated_by_current_matseg_strict_supports_"
                "and_cannot_be_replaced_by_unstructured_nearest_fill"
            ),
        }
    )
    return upstream_candidate.copy(), retained


def save_preview(
    path: Path,
    face: str,
    old_labels: np.ndarray,
    strict_labels: np.ndarray,
    new_labels: np.ndarray,
    valid: np.ndarray,
    stats: dict[str, Any],
) -> None:
    panels = [label_rgb(old_labels), label_rgb(strict_labels), label_rgb(new_labels)]
    panels[1][strict_labels < 0] = np.asarray([18, 18, 18], dtype=np.uint8)
    for panel in panels:
        panel[~valid] = np.asarray([25, 25, 25], dtype=np.uint8)
    h, w = old_labels.shape
    thumb_w = 520
    thumb_h = max(1, int(round(h * thumb_w / max(w, 1))))
    gap = 16
    canvas = Image.new("RGB", (3 * thumb_w + 4 * gap, thumb_h + 82), (242, 242, 242))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((gap, 12), f"{face}: old / strict seeds / structured v2", fill=(20, 20, 20), font=font)
    draw.text((gap, 32), f"selected={stats.get('method')}  Lab-reclassification=false", fill=(20, 20, 20), font=font)
    for index, panel in enumerate(panels):
        image = Image.fromarray(panel, mode="RGB").resize((thumb_w, thumb_h), Image.Resampling.NEAREST)
        canvas.paste(image, (gap + index * (thumb_w + gap), 66))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


def main() -> int:
    args = parse_args()
    metadata_path = args.layout_dir / "metadata_material_placement.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    face_records = {record["face"]: record for record in metadata.get("faces", [])}
    faces = args.faces or list(face_records)
    output_records = []
    for face in faces:
        old_labels = np.load(args.layout_dir / "labels_npy" / f"{face}.npy").astype(np.int16)
        known_labels = np.load(args.layout_dir / "labels_npy" / f"{face}_known_labels.npy").astype(np.int16)
        valid_path = args.source_package_dir / "debug" / f"{face}_valid_mask.png"
        valid = load_mask(valid_path, old_labels.shape) if valid_path.exists() else np.ones(old_labels.shape, dtype=bool)
        labels, stats = select_candidate(face, known_labels, valid, args)
        labels, stats = retain_validated_upstream_structure(
            old_labels,
            known_labels,
            valid,
            labels,
            stats,
            face_records[face].get("placement_stats", {}),
            args,
        )
        # Invalid UV texels never influence geometry or evaluation. Fill them
        # from the nearest valid texel only so downstream image files are dense.
        if not np.all(valid) and np.any(valid):
            _, nearest_valid = distance_transform_edt(~valid, return_indices=True)
            labels[~valid] = labels[nearest_valid[0][~valid], nearest_valid[1][~valid]]
        count = int(face_records[face].get("material_count", int(labels.max()) + 1))
        hard = np.stack([(labels == index).astype(np.float32) for index in range(count)], axis=0)
        radius = args.lowfreq_boundary_feather_frac * min(labels.shape)
        lowfreq = lowfreq_weights(labels, count, valid, radius)
        (args.out_dir / "labels_npy").mkdir(parents=True, exist_ok=True)
        np.save(args.out_dir / "labels_npy" / f"{face}.npy", labels.astype(np.int16))
        np.save(args.out_dir / "labels_npy" / f"{face}_known_labels.npy", known_labels.astype(np.int16))
        np.save(args.out_dir / "labels_npy" / f"{face}_soft_weights.npy", hard.astype(np.float32))
        np.save(args.out_dir / "labels_npy" / f"{face}_lowfreq_weights.npy", lowfreq.astype(np.float32))
        save_rgb(args.out_dir / "labels" / f"{face}.png", label_rgb(labels))
        save_rgb(
            args.out_dir / "soft_weights" / f"{face}.png",
            np.tensordot(np.moveaxis(hard, 0, -1), LABEL_COLORS[:count].astype(np.float32), axes=([2], [0])),
        )
        save_preview(
            args.out_dir / "previews" / f"{face}_structured_territory.jpg",
            face,
            old_labels,
            known_labels,
            labels,
            valid,
            stats,
        )
        old_valid = old_labels[valid]
        new_valid = labels[valid]
        stats["changed_valid_fraction"] = float(np.mean(old_valid != new_valid)) if old_valid.size else 0.0
        stats["hard_material_boundary"] = True
        stats["lowfreq_boundary_feather_radius_px"] = float(radius)
        record = face_records[face]
        record["placement_stats_before_structured_v2"] = record.get("placement_stats")
        record["placement_stats"] = stats
        for index, material in enumerate(record.get("materials", [])):
            material["territory_fraction"] = float(np.mean(new_valid == index)) if new_valid.size else 0.0
            material["soft_weight_fraction"] = material["territory_fraction"]
        output_records.append({"face": face, **stats})
        print(
            f"[structured-territory-v2] {face}: {stats['method']} "
            f"changed_valid={stats['changed_valid_fraction']:.4f}",
            flush=True,
        )
    metadata["method"] = "matseg_strict_seed_structured_territory_v2"
    metadata["structured_territory_v2"] = {
        "source_layout": str(args.layout_dir),
        "source_package_dir": str(args.source_package_dir),
        "identity_evidence": "MatSeg-grouped strict projected supports only",
        "completed_observed_lab_used_for_identity_or_boundary": False,
        "hard_high_frequency_material_boundary": True,
        "low_frequency_boundary_weights_file_suffix": "_lowfreq_weights.npy",
        "settings": {
            key: value
            for key, value in vars(args).items()
            if key not in {"layout_dir", "source_package_dir", "out_dir", "faces"}
        },
        "records": output_records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metadata_material_placement.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
