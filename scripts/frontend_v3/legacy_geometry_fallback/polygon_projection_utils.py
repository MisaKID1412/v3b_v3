#!/usr/bin/env python3
from __future__ import annotations

import colorsys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

RESAMPLE_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")

from build_polygon_photo_source_from_colmap import (
    ImagePose,
    Similarity,
    align_colmap_to_da3,
    face_meta_map,
    face_names,
    face_points_for_indices,
    face_texture_size,
    load_da3_hfalign_poses,
    load_da3_raw_poses,
    project_points,
    pycolmap_reconstruction,
    load_image_poses,
    valid_indices_for_face,
)


def load_point_cloud_glb(path: Path):
    from build_polygon_room_texture_from_da3_glb import load_point_cloud_glb as _load_point_cloud_glb

    return _load_point_cloud_glb(path)


def load_polygon_context(
    dataset_dir: Path,
    source_dir: Path,
    colmap_model_dir: Path,
    faces_arg: str | None,
    pose_source: str = "colmap_icp",
    da3_dir: Path | None = None,
    seed: int = 17,
    icp_points_colmap: int = 70000,
    icp_points_da3: int = 150000,
    icp_trim_percentile: float = 72.0,
    icp_iters: int = 28,
):
    manifest = __import__("json").loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    metas = face_meta_map(manifest)
    faces = face_names(manifest, faces_arg)
    pose_source = "colmap_icp" if pose_source == "colmap" else pose_source
    if pose_source == "da3_hfalign":
        if da3_dir is None:
            raise ValueError("pose_source=da3_hfalign requires da3_dir")
        poses, sim, _ = load_da3_hfalign_poses(dataset_dir, da3_dir)
        return manifest, metas, faces, sim, poses
    if pose_source == "da3_raw":
        if da3_dir is None:
            raise ValueError("pose_source=da3_raw requires da3_dir")
        poses, sim = load_da3_raw_poses(dataset_dir, da3_dir)
        return manifest, metas, faces, sim, poses
    rec = pycolmap_reconstruction(colmap_model_dir)
    colmap_points = np.asarray([p.xyz for p in rec.points3D.values()], dtype=np.float64)
    da3_points, _ = load_point_cloud_glb(dataset_dir / "scene.glb")
    align_args = SimpleNamespace(
        seed=seed,
        icp_points_colmap=icp_points_colmap,
        icp_points_da3=icp_points_da3,
        icp_trim_percentile=icp_trim_percentile,
        icp_iters=icp_iters,
    )
    sim = align_colmap_to_da3(colmap_points, da3_points.astype(np.float64), align_args)
    poses = load_image_poses(rec, dataset_dir, sim)
    return manifest, metas, faces, sim, poses


def face_palette(faces: list[str]) -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {}
    for idx, face in enumerate(faces):
        if face == "floor":
            colors[face] = (90, 180, 255)
            continue
        if face == "ceiling":
            colors[face] = (180, 120, 255)
            continue
        hue = (0.05 + 0.73 * (idx + 1) / max(1, len(faces))) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 1.0)
        colors[face] = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
    return colors


def image_keys(pose: ImagePose) -> list[str]:
    stem = Path(pose.name).stem
    return [
        f"view_{pose.image_id:03d}",
        f"view_{pose.image_id:06d}",
        stem,
    ]


def candidate_view_mask_paths(mask_dir: Path, pose: ImagePose, suffix: str) -> list[Path]:
    paths = []
    for key in image_keys(pose):
        paths.append(mask_dir / f"{key}_{suffix}.png")
    if suffix == "object_mask":
        for key in image_keys(pose):
            paths.append(mask_dir / f"{key}.png")
    return paths


def load_bool_mask(path: Path | None, shape_hw: tuple[int, int] | None = None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    image = Image.open(path).convert("L")
    if shape_hw is not None and image.size != (shape_hw[1], shape_hw[0]):
        image = image.resize((shape_hw[1], shape_hw[0]), RESAMPLE_NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not np.any(mask):
        return mask.astype(bool)
    kernel = np.ones((2 * px + 1, 2 * px + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def morph_close(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not np.any(mask):
        return mask.astype(bool)
    kernel = np.ones((2 * px + 1, 2 * px + 1), dtype=np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1 or not np.any(mask):
        return mask.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            keep |= labels == label
    return keep


def fill_holes(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)
    h, w = mask.shape
    inv = (~mask.astype(bool)).astype(np.uint8)
    flood = inv.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    holes = flood == 0
    return mask.astype(bool) | holes


def finalize_mask(counts: np.ndarray, dilate_px: int, close_px: int, min_area: int, fill: bool = False) -> np.ndarray:
    mask = counts > 0
    mask = morph_close(mask, close_px)
    if fill:
        mask = fill_holes(mask)
    mask = remove_small_components(mask, min_area)
    return dilate(mask, dilate_px)


def build_view_face_id_and_zbuffer(
    pose: ImagePose,
    faces: list[str],
    source_dir: Path,
    sim: Similarity,
    manifest: dict,
    metas: dict[str, dict],
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    zbuf = np.full((pose.height, pose.width), np.inf, dtype=np.float32)
    face_id = np.full((pose.height, pose.width), 255, dtype=np.uint16)
    flat_z = zbuf.reshape(-1)
    flat_f = face_id.reshape(-1)
    for face_idx, face in enumerate(faces):
        size = face_texture_size(source_dir, metas[face])
        w, h = size
        valid = valid_indices_for_face(source_dir, face, size)
        if valid.size == 0:
            continue
        rows_all = (valid // w).astype(np.int32)
        cols_all = (valid % w).astype(np.int32)
        sample = ((rows_all % max(1, stride)) == 0) & ((cols_all % max(1, stride)) == 0)
        rows_all = rows_all[sample]
        cols_all = cols_all[sample]
        for start in range(0, len(rows_all), 240000):
            rows = rows_all[start : start + 240000]
            cols = cols_all[start : start + 240000]
            pts_da3 = face_points_for_indices(face, rows, cols, size, manifest, metas)
            pts_col = sim.da3_to_colmap(pts_da3)
            u, v, z = project_points(pts_col, pose)
            ok = (z > 0.05) & (u >= 0) & (u < pose.width) & (v >= 0) & (v < pose.height)
            if not np.any(ok):
                continue
            px = np.clip(np.round(u[ok]).astype(np.int32), 0, pose.width - 1)
            py = np.clip(np.round(v[ok]).astype(np.int32), 0, pose.height - 1)
            pix = py * pose.width + px
            zz = z[ok]
            order = np.lexsort((zz, pix))
            pix = pix[order]
            zz = zz[order]
            first = np.r_[True, pix[1:] != pix[:-1]]
            pix = pix[first]
            zz = zz[first]
            closer = zz < flat_z[pix]
            if np.any(closer):
                pix = pix[closer]
                flat_z[pix] = zz[closer]
                flat_f[pix] = face_idx
    kernel = np.ones((max(3, int(stride * 2 + 1)), max(3, int(stride * 2 + 1))), dtype=np.uint8)
    large = np.float32(1e9)
    zfilled = zbuf.copy()
    zfilled[~np.isfinite(zfilled)] = large
    zmin = cv2.erode(zfilled, kernel, iterations=1)
    face_out = face_id.copy()
    unknown = face_out == 255
    if np.any(unknown):
        for idx in range(len(faces)):
            m = face_id == idx
            if np.any(m):
                grown = cv2.dilate(m.astype(np.uint8), kernel, iterations=1).astype(bool)
                fill = unknown & grown
                face_out[fill] = idx
                unknown &= ~fill
    zmin[zmin >= large * 0.5] = np.inf
    return face_out, zmin


def face_id_to_rgb(face_id: np.ndarray, faces: list[str], palette: dict[str, tuple[int, int, int]]) -> np.ndarray:
    rgb = np.zeros((*face_id.shape, 3), dtype=np.uint8)
    for idx, face in enumerate(faces):
        rgb[face_id == idx] = palette[face]
    return rgb


def accumulate_view_mask_to_atlas(
    view_mask: np.ndarray,
    face: str,
    counts: np.ndarray,
    pose: ImagePose,
    source_dir: Path,
    sim: Similarity,
    manifest: dict,
    metas: dict[str, dict],
    zbuf: np.ndarray,
    depth_abs_tol: float,
    depth_rel_tol: float,
    chunk_size: int = 180000,
) -> int:
    size = face_texture_size(source_dir, metas[face])
    w, h = size
    valid_flat = valid_indices_for_face(source_dir, face, size)
    hits_total = 0
    for start in range(0, len(valid_flat), chunk_size):
        flat = valid_flat[start : start + chunk_size]
        rows = (flat // w).astype(np.int32)
        cols = (flat % w).astype(np.int32)
        pts_da3 = face_points_for_indices(face, rows, cols, size, manifest, metas)
        pts_col = sim.da3_to_colmap(pts_da3)
        u, v, z = project_points(pts_col, pose)
        in_frame = (
            (z > 1e-6)
            & (u >= 0.0)
            & (v >= 0.0)
            & (u <= pose.width - 1.0)
            & (v <= pose.height - 1.0)
        )
        if not np.any(in_frame):
            continue
        idx = np.flatnonzero(in_frame)
        px = np.clip(np.round(u[idx]).astype(np.int32), 0, pose.width - 1)
        py = np.clip(np.round(v[idx]).astype(np.int32), 0, pose.height - 1)
        z_shell = zbuf[py, px]
        has_depth = np.isfinite(z_shell)
        depth_tol = depth_abs_tol + depth_rel_tol * np.maximum(z[idx], 0.0)
        visible = has_depth & (np.abs(z[idx] - z_shell) <= depth_tol)
        if not np.any(visible):
            continue
        idx = idx[visible]
        px = px[visible]
        py = py[visible]
        hit = view_mask[py, px]
        if np.any(hit):
            flat_hit = flat[idx[hit]]
            rr = (flat_hit // w).astype(np.int32)
            cc = (flat_hit % w).astype(np.int32)
            np.add.at(counts, (rr, cc), 1)
            hits_total += int(np.count_nonzero(hit))
    return hits_total


def project_atlas_mask_to_view(
    atlas_mask: np.ndarray,
    face: str,
    pose: ImagePose,
    source_dir: Path,
    sim: Similarity,
    manifest: dict,
    metas: dict[str, dict],
    zbuf: np.ndarray,
    depth_abs_tol: float,
    depth_rel_tol: float,
    chunk_size: int = 180000,
) -> np.ndarray:
    out = np.zeros((pose.height, pose.width), dtype=bool)
    if atlas_mask is None or not np.any(atlas_mask):
        return out
    size = face_texture_size(source_dir, metas[face])
    w, h = size
    if atlas_mask.shape != (h, w):
        atlas_mask = cv2.resize(atlas_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    flat_all = np.flatnonzero(atlas_mask.reshape(-1))
    valid_flat = valid_indices_for_face(source_dir, face, size)
    valid = np.zeros(h * w, dtype=bool)
    valid[valid_flat] = True
    flat_all = flat_all[valid[flat_all]]
    for start in range(0, len(flat_all), chunk_size):
        flat = flat_all[start : start + chunk_size]
        rows = (flat // w).astype(np.int32)
        cols = (flat % w).astype(np.int32)
        pts_da3 = face_points_for_indices(face, rows, cols, size, manifest, metas)
        pts_col = sim.da3_to_colmap(pts_da3)
        u, v, z = project_points(pts_col, pose)
        in_frame = (
            (z > 1e-6)
            & (u >= 0.0)
            & (v >= 0.0)
            & (u <= pose.width - 1.0)
            & (v <= pose.height - 1.0)
        )
        if not np.any(in_frame):
            continue
        idx = np.flatnonzero(in_frame)
        px = np.clip(np.round(u[idx]).astype(np.int32), 0, pose.width - 1)
        py = np.clip(np.round(v[idx]).astype(np.int32), 0, pose.height - 1)
        z_shell = zbuf[py, px]
        has_depth = np.isfinite(z_shell)
        depth_tol = depth_abs_tol + depth_rel_tol * np.maximum(z[idx], 0.0)
        visible = has_depth & (np.abs(z[idx] - z_shell) <= depth_tol)
        if np.any(visible):
            out[py[visible], px[visible]] = True
    return out
