#!/usr/bin/env python3
"""Decide v3b region material identity with MatSeg, and nothing else.

The v3b regions remain the atomic proposals.  MatSeg replaces only the old Lab
same-material decision by thresholding cosine similarity between full-view
region descriptors.  In particular, the old ``material_count`` is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from run_matseg_floor_diagnostic import MatSegRunner, draw_similarity_matrix, unit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--face", default="floor")
    parser.add_argument("--max-side", type=int, default=700)
    parser.add_argument("--views-per-region", type=int, default=3)
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.87,
        help="Regions at or above this MatSeg cosine similarity are the same material.",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def material_identity_components(similarity: np.ndarray, threshold: float) -> np.ndarray:
    """Return transitive same-material components from pairwise MatSeg decisions."""
    count = int(similarity.shape[0])
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(count):
        for second in range(first):
            if float(similarity[first, second]) >= threshold:
                union(first, second)

    roots = [find(index) for index in range(count)]
    root_order = {root: label for label, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([root_order[root] for root in roots], dtype=np.int32)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def sample_feature(feature: np.ndarray, u: np.ndarray, v: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_shape
    x = np.clip(np.round(u * feature.shape[1] / max(image_w, 1)).astype(np.int32), 0, feature.shape[1] - 1)
    y = np.clip(np.round(v * feature.shape[0] / max(image_h, 1)).astype(np.int32), 0, feature.shape[0] - 1)
    values = feature[y, x]
    if values.shape[0] > 128:
        norms = np.linalg.norm(values, axis=1)
        lo, hi = np.percentile(norms, [5.0, 95.0])
        trimmed = values[(norms >= lo) & (norms <= hi)]
        if trimmed.shape[0] >= 128:
            values = trimmed
    return unit(np.mean(values, axis=0))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    face_stat = next(item for item in metadata["stats"] if item["face"] == args.face)
    regions = sorted(face_stat["regions"], key=lambda item: int(item["region"]))

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    import generate_chord_view_contributor_region_priors as generator
    projection = generator.proj

    projection_params = dict(metadata["params"])
    for key in (
        "source_dir",
        "polygon_source_dir",
        "dataset_dir",
        "colmap_model_dir",
        "da3_dir",
        "object_mask_dir",
        "out_dir",
        "chord_output_dir",
    ):
        value = projection_params.get(key)
        if value is not None:
            projection_params[key] = Path(value)
    projection_args = SimpleNamespace(**projection_params)
    manifest, metas, _ = generator.manifest_and_faces(projection_args)
    poses, similarity, raw_to_room = projection.load_da3_hfalign_poses(
        projection_args.dataset_dir, projection_args.da3_dir
    )
    da3_views = projection.load_da3_views(projection_args.da3_dir, poses)
    all_faces = projection.face_names(manifest, None)
    caches = {"zbuffer": {}, "face_id": {}, "reject": {}, "depth_calib": {}}

    raw_path = generator.source_image_path(projection_args.source_dir, args.face)
    raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(raw_path)
    support_dir = args.metadata.parent / "debug"

    per_region_contributors: dict[int, list[dict]] = {}
    for region in regions:
        region_id = int(region["region"])
        support = load_mask(support_dir / f"{args.face}_region_{region_id:02d}_support.png", raw.shape[:2])
        contributors = generator.trace_region_contributors(
            projection_args,
            args.face,
            region_id,
            support,
            poses,
            similarity,
            manifest,
            metas,
            all_faces,
            da3_views,
            raw_to_room,
            caches,
        )
        per_region_contributors[region_id] = contributors[: args.views_per_region]

    selected_by_image: dict[int, list[tuple[int, dict]]] = {}
    for region_id, contributors in per_region_contributors.items():
        for contributor in contributors:
            selected_by_image.setdefault(int(contributor["pose"].image_id), []).append((region_id, contributor))

    runner = MatSegRunner(args.vendor_dir, args.checkpoint, args.device)
    region_view_vectors: dict[int, list[tuple[np.ndarray, float, dict]]] = {
        int(region["region"]): [] for region in regions
    }
    source_view_records: list[dict] = []
    for image_id, assignments in sorted(selected_by_image.items()):
        pose = assignments[0][1]["pose"]
        image = cv2.imread(str(pose.image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        feature, resized = runner.infer(image, args.max_side)
        overlay = resized.copy()
        assignment_records = []
        for region_id, contributor in assignments:
            vector = sample_feature(feature, contributor["u"], contributor["v"], image.shape[:2])
            weight = float(contributor.get("weight", contributor.get("count", 1.0)))
            info = {
                "image_id": int(image_id),
                "view_name": str(pose.name),
                "sample_count": int(contributor["count"]),
                "weight": weight,
                "mean_depth_residual": float(contributor["mean_depth_residual"]),
                "mean_surface_distance": float(contributor["mean_surface_distance"]),
            }
            region_view_vectors[region_id].append((vector, weight, info))
            assignment_records.append({"region": region_id, **info})
            x = np.clip(np.round(contributor["u"] * resized.shape[1] / image.shape[1]).astype(np.int32), 0, resized.shape[1] - 1)
            y = np.clip(np.round(contributor["v"] * resized.shape[0] / image.shape[0]).astype(np.int32), 0, resized.shape[0] - 1)
            stride = max(1, x.size // 1500)
            color = [(60, 210, 80), (210, 140, 50), (70, 80, 230), (220, 70, 210), (40, 210, 220)][region_id % 5]
            overlay[y[::stride], x[::stride]] = color
        cv2.imwrite(str(args.output_dir / f"source_view_{image_id:06d}_samples.png"), overlay)
        source_view_records.append(
            {
                "image_id": int(image_id),
                "view_name": str(pose.name),
                "image_path": str(pose.image_path),
                "assignments": assignment_records,
            }
        )

    region_vectors: list[np.ndarray] = []
    region_records: list[dict] = []
    for region in regions:
        region_id = int(region["region"])
        items = region_view_vectors[region_id]
        if not items:
            raise RuntimeError(f"No full-view descriptors for region {region_id}")
        weights = np.asarray([item[1] for item in items], dtype=np.float64)
        weights /= max(float(np.sum(weights)), 1e-12)
        vector = unit(np.sum(np.stack([item[0] for item in items]) * weights[:, None], axis=0))
        region_vectors.append(vector)
        region_records.append(
            {
                "region": region_id,
                "old_material_id": int(region["material_id"]),
                "source": str(region.get("source", "")),
                "material_box_purity": float(region.get("material_box_purity", 0.0)),
                "box_yx_size": [int(value) for value in region["box_yx_size"]],
                "views": [item[2] for item in items],
            }
        )

    vectors = np.stack(region_vectors).astype(np.float32)
    similarity_matrix = vectors @ vectors.T
    cluster_labels = material_identity_components(
        similarity_matrix,
        float(args.similarity_threshold),
    )
    for record, cluster in zip(region_records, cluster_labels.tolist()):
        record["matseg_material_id"] = int(cluster)
    draw_similarity_matrix(
        similarity_matrix,
        [f"r{record['region']:02d}" for record in region_records],
        args.output_dir / "fullview_region_similarity.png",
    )

    report = {
        "method": "matseg_region_material_identity_threshold_v1",
        "face": args.face,
        "similarity_threshold": float(args.similarity_threshold),
        "material_count": int(len(np.unique(cluster_labels))),
        "source_shape_hw": [int(raw.shape[0]), int(raw.shape[1])],
        "max_view_side": int(args.max_side),
        "views_per_region": int(args.views_per_region),
        "regions": region_records,
        "similarity_matrix": similarity_matrix.tolist(),
        "source_views": source_view_records,
        "gpu_peak_memory_mib": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() else 0.0,
    }
    (args.output_dir / "matseg_material_identity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
