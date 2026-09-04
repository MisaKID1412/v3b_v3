#!/usr/bin/env python3
"""Evaluate zero-shot MatSeg features on v3b material candidates and an atlas face.

This script is deliberately read-only with respect to existing v3b outputs.  It
writes candidate embeddings, similarity diagnostics, and a low-resolution atlas
material proposal into a separate output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--face", default="floor")
    parser.add_argument("--max-side", type=int, default=700)
    parser.add_argument("--views-per-region", type=int, default=3)
    parser.add_argument("--margin-threshold", type=float, default=0.025)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def unit(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), eps)


def robust_descriptor(feature: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != feature.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (feature.shape[1], feature.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if np.count_nonzero(mask) > 256:
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    values = feature[mask]
    if values.shape[0] < 32:
        values = feature.reshape(-1, feature.shape[-1])
    # The official training objective averages per-pixel descriptors.  A light
    # norm trim prevents a few invalid boundary pixels from owning the average.
    norms = np.linalg.norm(values, axis=1)
    lo, hi = np.percentile(norms, [5.0, 95.0])
    trimmed = values[(norms >= lo) & (norms <= hi)]
    if trimmed.shape[0] >= 32:
        values = trimmed
    return unit(np.mean(values, axis=0))


class MatSegRunner:
    def __init__(self, vendor_dir: Path, checkpoint: Path, device: str) -> None:
        sys.path.insert(0, str(vendor_dir))
        original_convnext_base = tv_models.convnext_base

        def convnext_without_imagenet_download(*args, **kwargs):
            kwargs["weights"] = None
            return original_convnext_base(*args, **kwargs)

        tv_models.convnext_base = convnext_without_imagenet_download
        try:
            import Desnet  # type: ignore

            self.net = Desnet.Net(descriptor_depth=128)
        finally:
            tv_models.convnext_base = original_convnext_base
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.net.load_state_dict(state, strict=True)
        self.net.eval()
        self.device = device

    @torch.inference_mode()
    def infer(self, image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray]:
        height, width = image_bgr.shape[:2]
        scale = min(1.0, float(max_side) / max(height, width))
        if scale < 1.0:
            resized = cv2.resize(
                image_bgr,
                (max(16, int(round(width * scale))), max(16, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = image_bgr
        batch = resized[None].astype(np.uint8)
        roi = np.zeros(batch.shape[:3], dtype=np.float32)
        fmap = self.net(batch, roi, TrainMode=False)
        fmap = F.normalize(fmap, dim=1)[0].permute(1, 2, 0).detach().cpu().numpy()
        return fmap.astype(np.float32), resized


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.ones(shape, dtype=bool)
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


def load_face_stat(metadata: dict, face: str) -> dict:
    return next(item for item in metadata["stats"] if item["face"] == face)


def cluster_unit_vectors(vectors: np.ndarray, count: int, seed: int = 20260825) -> np.ndarray:
    data = np.asarray(vectors, dtype=np.float32)
    cv2.setRNGSeed(seed)
    _, labels, _ = cv2.kmeans(
        data,
        count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        64,
        cv2.KMEANS_PP_CENTERS,
    )
    return labels.reshape(-1).astype(np.int32)


def draw_similarity_matrix(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    cell = 150
    header = 130
    canvas = np.full((header + cell * len(labels), header + cell * len(labels), 3), 248, np.uint8)
    minimum = float(np.min(matrix))
    maximum = float(np.max(matrix))
    denom = max(maximum - minimum, 1e-6)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = float(matrix[row, col])
            norm = (value - minimum) / denom
            color = cv2.applyColorMap(np.array([[int(round(norm * 255))]], np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0]
            y0 = header + row * cell
            x0 = header + col * cell
            canvas[y0 : y0 + cell, x0 : x0 + cell] = color
            cv2.putText(canvas, f"{value:.3f}", (x0 + 25, y0 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    for index, label in enumerate(labels):
        cv2.putText(canvas, label, (8, header + index * cell + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (header + index * cell + 10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def normalized_gray(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros(values.shape, np.uint8)
    selected = values[valid]
    if selected.size:
        lo, hi = np.percentile(selected, [1.0, 99.0])
        scaled = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        output = np.round(scaled * 255.0).astype(np.uint8)
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    face_stat = load_face_stat(metadata, args.face)
    regions = sorted(face_stat["regions"], key=lambda item: int(item["region"]))
    runner = MatSegRunner(args.vendor_dir, args.checkpoint, args.device)

    region_records: list[dict] = []
    region_vectors: list[np.ndarray] = []
    for region in regions:
        view_vectors: list[np.ndarray] = []
        view_records: list[dict] = []
        for candidate in region.get("view_candidates", [])[: args.views_per_region]:
            image_path = Path(candidate["chord_input"])
            mask_path = Path(candidate["candidate_mask"])
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            feature, resized = runner.infer(image, args.max_side)
            mask = load_mask(mask_path, image.shape[:2])
            if resized.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            vector = robust_descriptor(feature, mask)
            view_vectors.append(vector)
            view_records.append(
                {
                    "stem": candidate["stem"],
                    "view_name": candidate["view_name"],
                    "selection_score": float(candidate.get("selection_score", 0.0)),
                }
            )
        if not view_vectors:
            raise RuntimeError(f"No readable candidates for {args.face} region {region['region']}")
        region_vector = unit(np.mean(np.stack(view_vectors), axis=0))
        region_vectors.append(region_vector)
        region_records.append(
            {
                "region": int(region["region"]),
                "old_material_id": int(region["material_id"]),
                "source": str(region.get("source", "")),
                "material_box_purity": float(region.get("material_box_purity", 0.0)),
                "box_yx_size": [int(value) for value in region["box_yx_size"]],
                "support_texels": int(region.get("support_texels", 0)),
                "views": view_records,
            }
        )

    vectors = np.stack(region_vectors).astype(np.float32)
    similarity = vectors @ vectors.T
    material_count = int(face_stat.get("material_count", len({item["material_id"] for item in regions})))
    material_count = max(1, min(material_count, len(regions)))
    cluster_labels = cluster_unit_vectors(vectors, material_count)
    for record, cluster in zip(region_records, cluster_labels.tolist()):
        record["matseg_cluster"] = int(cluster)

    cluster_prototypes: list[np.ndarray] = []
    cluster_anchor_regions: list[int] = []
    for cluster in range(material_count):
        members = np.flatnonzero(cluster_labels == cluster)
        if members.size == 0:
            raise RuntimeError(f"Empty MatSeg cluster {cluster}")
        primary = [
            int(index)
            for index in members
            if region_records[int(index)]["source"] == "material_cluster"
        ]
        pool = primary if primary else members.tolist()
        anchor_index = max(pool, key=lambda index: region_records[index]["material_box_purity"])
        cluster_anchor_regions.append(int(region_records[anchor_index]["region"]))
        cluster_prototypes.append(vectors[anchor_index])

    labels = [f"r{item['region']:02d}" for item in region_records]
    draw_similarity_matrix(similarity, labels, args.output_dir / "candidate_region_similarity.png")

    source_dir = Path(metadata["source_dir"])
    raw_path = source_dir / "debug" / f"{args.face}_raw_projected.png"
    if not raw_path.exists():
        raw_path = source_dir / "textures" / f"{args.face}.png"
    raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(raw_path)
    observed_path = source_dir / "debug" / f"{args.face}_final_keep_mask.png"
    observed_full = load_mask(observed_path, raw.shape[:2])

    scale = min(1.0, float(args.max_side) / max(raw.shape[:2]))
    model_size = (max(16, int(round(raw.shape[1] * scale))), max(16, int(round(raw.shape[0] * scale))))
    raw_small = cv2.resize(raw, model_size, interpolation=cv2.INTER_AREA)
    observed_small = cv2.resize(observed_full.astype(np.uint8), model_size, interpolation=cv2.INTER_NEAREST).astype(bool)
    # Only the model input is filled.  Scores and metrics remain restricted to
    # strict observed texels, so completion never becomes material evidence.
    model_input = cv2.inpaint(raw_small, (~observed_small).astype(np.uint8) * 255, 5.0, cv2.INPAINT_TELEA)
    atlas_feature, model_input = runner.infer(model_input, args.max_side)
    prototype_matrix = np.stack(cluster_prototypes).astype(np.float32)
    scores = np.einsum("hwc,kc->hwk", atlas_feature, prototype_matrix)
    order = np.argsort(scores, axis=2)
    best = order[..., -1].astype(np.uint8)
    if material_count > 1:
        margin = np.take_along_axis(scores, order[..., -1:], axis=2)[..., 0] - np.take_along_axis(scores, order[..., -2:-1], axis=2)[..., 0]
    else:
        margin = np.ones(best.shape, np.float32)
    known = observed_small & (margin >= float(args.margin_threshold))
    labels_small = np.full(best.shape, 255, np.uint8)
    labels_small[known] = best[known]

    color_table = np.array([[67, 160, 71], [194, 86, 214], [52, 152, 219], [230, 192, 62]], np.uint8)
    color = np.zeros((*best.shape, 3), np.uint8)
    for cluster in range(material_count):
        color[labels_small == cluster] = color_table[cluster % len(color_table)]
    overlay = cv2.addWeighted(model_input, 0.55, color, 0.45, 0.0)
    overlay[~observed_small] = model_input[~observed_small] // 3
    cv2.imwrite(str(args.output_dir / f"{args.face}_model_input.png"), model_input)
    cv2.imwrite(str(args.output_dir / f"{args.face}_matseg_clusters.png"), color)
    cv2.imwrite(str(args.output_dir / f"{args.face}_matseg_overlay.png"), overlay)
    cv2.imwrite(str(args.output_dir / f"{args.face}_matseg_margin.png"), normalized_gray(margin, observed_small))
    np.save(args.output_dir / f"{args.face}_matseg_scores.npy", scores.astype(np.float16))
    np.save(args.output_dir / f"{args.face}_matseg_labels.npy", labels_small)

    region_dir = args.metadata.parent
    debug_dir = region_dir / "debug"
    per_region_scores: list[dict] = []
    for record in region_records:
        support_path = debug_dir / f"{args.face}_region_{record['region']:02d}_support.png"
        support = load_mask(support_path, raw.shape[:2])
        support_small = cv2.resize(support.astype(np.uint8), model_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        valid = support_small & observed_small
        if np.count_nonzero(valid):
            mean_scores = np.mean(scores[valid], axis=0)
            assigned_fraction = [float(np.mean(best[valid] == cluster)) for cluster in range(material_count)]
            known_fraction = float(np.mean(known[valid]))
        else:
            mean_scores = np.zeros(material_count, np.float32)
            assigned_fraction = [0.0] * material_count
            known_fraction = 0.0
        per_region_scores.append(
            {
                "region": record["region"],
                "mean_atlas_scores": [float(value) for value in mean_scores],
                "assigned_fraction": assigned_fraction,
                "known_fraction": known_fraction,
            }
        )

    observed_count = max(int(np.count_nonzero(observed_small)), 1)
    cluster_fractions = [float(np.count_nonzero((best == cluster) & observed_small) / observed_count) for cluster in range(material_count)]
    known_cluster_fractions = [float(np.count_nonzero((labels_small == cluster) & observed_small) / observed_count) for cluster in range(material_count)]
    report = {
        "method": "matseg_zero_shot_candidate_cluster_and_atlas_proposal_v1",
        "face": args.face,
        "checkpoint": str(args.checkpoint),
        "source_image": str(raw_path),
        "source_shape_hw": [int(raw.shape[0]), int(raw.shape[1])],
        "model_shape_hw": [int(model_input.shape[0]), int(model_input.shape[1])],
        "margin_threshold": float(args.margin_threshold),
        "regions": region_records,
        "candidate_similarity": similarity.tolist(),
        "cluster_anchor_regions": cluster_anchor_regions,
        "cluster_fractions_argmax": cluster_fractions,
        "cluster_fractions_known": known_cluster_fractions,
        "unknown_fraction": float(np.count_nonzero(observed_small & ~known) / observed_count),
        "per_region_atlas_scores": per_region_scores,
        "gpu_peak_memory_mib": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)) if torch.cuda.is_available() else 0.0,
    }
    (args.output_dir / "matseg_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
