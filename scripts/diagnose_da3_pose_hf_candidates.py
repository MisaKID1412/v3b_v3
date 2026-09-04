#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_polygon_photo_source_from_colmap import (
    ImagePose,
    identity_similarity,
    build_face_id_and_zbuffer,
    face_meta_map,
    face_names,
    fit_depth_affine,
    load_hf_alignment,
    sample_float_map,
    scaled_da3_intrinsics_for_image,
    to_4x4,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose which DA3 extrinsic/hf_alignment composition best projects "
            "a polygon room shell into the input images."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--da3-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--faces", default=None)
    parser.add_argument("--view-stride", type=int, default=8)
    parser.add_argument("--max-views", type=int, default=28)
    parser.add_argument("--zbuffer-stride", type=int, default=6)
    parser.add_argument("--sample-stride", type=int, default=16)
    return parser.parse_args()


def da3_names(da3_dir: Path, count: int) -> list[str]:
    meta_path = da3_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw = meta.get("image_names") or meta.get("images") or meta.get("image_paths") or []
        names = [Path(str(x)).name for x in raw]
        if len(names) == count:
            return names
    return [f"view_{i:06d}.png" for i in range(count)]


def pose_from_matrix(
    idx: int,
    name: str,
    image_path: Path,
    depth_shape: tuple[int, int],
    intrinsic: np.ndarray,
    w2c: np.ndarray,
) -> ImagePose:
    with Image.open(image_path) as im:
        width, height = im.size
    k = scaled_da3_intrinsics_for_image(intrinsic, depth_shape, (width, height))
    r = w2c[:3, :3].astype(np.float64)
    t = w2c[:3, 3].astype(np.float64)
    center = (-r.T @ t).astype(np.float64)
    return ImagePose(
        image_id=int(idx),
        name=Path(name).name,
        camera_id=int(idx),
        width=int(width),
        height=int(height),
        model="PINHOLE",
        params=np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=np.float64),
        Rcw=r,
        tcw=t,
        center_colmap=center,
        center_da3=center,
        image_path=image_path,
    )


def candidate_w2c(raw: np.ndarray, hf: np.ndarray) -> dict[str, np.ndarray]:
    raw = to_4x4(raw)
    inv_raw = np.linalg.inv(raw)
    inv_hf = np.linalg.inv(hf)
    return {
        "raw_w2c__glb_eq_hf_raw__E_raw_inv_hf": raw @ inv_hf,
        "raw_w2c__glb_eq_inv_hf_raw__E_raw_hf": raw @ hf,
        "raw_c2w__glb_eq_hf_raw__inv_raw_inv_hf": inv_raw @ inv_hf,
        "raw_c2w__glb_eq_inv_hf_raw__inv_raw_hf": inv_raw @ hf,
        "raw_w2c_no_hf": raw,
        "raw_c2w_no_hf": inv_raw,
    }


def depth_error_for_pose(
    pose: ImagePose,
    depth: np.ndarray,
    source_dir: Path,
    manifest: dict,
    metas: dict,
    faces: list[str],
    zbuffer_stride: int,
    sample_stride: int,
) -> dict:
    _, zbuf = build_face_id_and_zbuffer(
        pose,
        faces,
        source_dir,
        identity_similarity(),
        manifest,
        metas,
        zbuffer_stride,
    )
    yy, xx = np.mgrid[0 : pose.height : sample_stride, 0 : pose.width : sample_stride]
    z = zbuf[yy, xx].astype(np.float32).reshape(-1)
    x = xx.astype(np.float32).reshape(-1)
    y = yy.astype(np.float32).reshape(-1)
    dh, dw = depth.shape[:2]
    du = x * (float(dw) / max(float(pose.width), 1.0))
    dv = y * (float(dh) / max(float(pose.height), 1.0))
    d = sample_float_map(depth, du, dv, border_value=np.nan)
    valid = np.isfinite(z) & np.isfinite(d) & (z > 1e-6) & (d > 1e-6)
    if int(np.count_nonzero(valid)) < 128:
        return {"samples": int(np.count_nonzero(valid)), "median_abs": None, "p80_abs": None, "mode": None}
    z = z[valid].astype(np.float64)
    d = d[valid].astype(np.float64)
    best = None
    for mode, xvals in (("linear", d), ("inverse", 1.0 / np.maximum(d, 1e-6))):
        scale, shift, err = fit_depth_affine(xvals, z)
        keep = err <= np.percentile(err, 80.0)
        if int(np.count_nonzero(keep)) >= 128:
            scale, shift, _ = fit_depth_affine(xvals[keep], z[keep])
            err = np.abs(scale * xvals + shift - z)
        item = {
            "samples": int(z.size),
            "median_abs": float(np.median(err)),
            "p80_abs": float(np.percentile(err, 80.0)),
            "mode": mode,
            "scale": float(scale),
            "shift": float(shift),
            "covered_pixels": int(np.count_nonzero(np.isfinite(zbuf))),
        }
        if best is None or item["median_abs"] < best["median_abs"]:
            best = item
    return best


def main() -> int:
    args = parse_args()
    manifest_path = args.source_dir / "metadata.json"
    if not manifest_path.exists():
        manifest_path = args.source_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metas = face_meta_map(manifest)
    faces = face_names(manifest, args.faces)

    depth = np.load(args.da3_dir / "depth.npy")
    extrinsics = np.load(args.da3_dir / "extrinsics.npy").astype(np.float64)
    intrinsics = np.load(args.da3_dir / "intrinsics.npy").astype(np.float64)
    names = da3_names(args.da3_dir, depth.shape[0])
    hf = load_hf_alignment(args.dataset_dir / "scene.glb")
    image_dir = args.dataset_dir / "input_images"
    view_indices = list(range(0, depth.shape[0], max(1, args.view_stride)))[: args.max_views]
    results: dict[str, list[dict]] = {}

    for idx in view_indices:
        image_path = image_dir / Path(names[idx]).name
        if not image_path.exists():
            continue
        for cand_name, w2c in candidate_w2c(extrinsics[idx], hf).items():
            pose = pose_from_matrix(idx, names[idx], image_path, tuple(depth.shape[1:3]), intrinsics[idx], w2c)
            err = depth_error_for_pose(
                pose,
                depth[idx],
                args.source_dir,
                manifest,
                metas,
                faces,
                args.zbuffer_stride,
                args.sample_stride,
            )
            err["view"] = int(idx)
            err["image"] = Path(names[idx]).name
            results.setdefault(cand_name, []).append(err)
            print(cand_name, idx, err, flush=True)

    summary = {}
    for cand, rows in results.items():
        vals = [r["median_abs"] for r in rows if r.get("median_abs") is not None]
        p80 = [r["p80_abs"] for r in rows if r.get("p80_abs") is not None]
        summary[cand] = {
            "usable_views": len(vals),
            "median_of_median_abs": float(np.median(vals)) if vals else None,
            "median_of_p80_abs": float(np.median(p80)) if p80 else None,
            "views": rows,
        }
    ordered = sorted(
        summary.items(),
        key=lambda item: float("inf") if item[1]["median_of_median_abs"] is None else item[1]["median_of_median_abs"],
    )
    out = {"source_dir": str(args.source_dir), "da3_dir": str(args.da3_dir), "view_indices": view_indices, "summary": dict(ordered)}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best": ordered[0] if ordered else None, "out_json": str(args.out_json)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
