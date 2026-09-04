#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

RESAMPLE_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")
NEAREST_EVIDENCE_SCHEMA = "v3b-nearest-visible-evidence-v1"
NEAREST_SELECTION_POLICY = "min_camera_distance_then_max_projection_weight"
NEAREST_DISTANCE_TIE_ATOL = 1.0e-8
NEAREST_RGB_RANGE_ATOL = 1.0e-6


def load_point_cloud_glb(path: Path):
    from build_polygon_room_texture_from_da3_glb import load_point_cloud_glb as _load_point_cloud_glb

    return _load_point_cloud_glb(path)


def rotate2(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    r = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ r.T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build v53/v61-style source atlases for a polygonal room by projecting "
            "registered COLMAP input images onto the DA3-derived polygon room shell."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--polygon-source-dir", type=Path, required=True)
    parser.add_argument("--colmap-model-dir", type=Path, required=True)
    parser.add_argument(
        "--pose-source",
        choices=["colmap_icp", "colmap", "da3_hfalign", "da3_raw"],
        default="colmap_icp",
        help=(
            "Projection camera source. colmap_icp matches the original polygon port. "
            "da3_hfalign uses DA3 numeric poses transformed by the scene.glb hf_alignment, "
            "so cameras and polygon shell live in the same exported GLB coordinate frame."
        ),
    )
    parser.add_argument(
        "--da3-dir",
        type=Path,
        default=None,
        help="Optional full DA3 numeric output containing depth.npy/conf.npy/extrinsics.npy/intrinsics.npy.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--emit-nearest-visible-evidence",
        action="store_true",
        help=(
            "Alongside the unchanged weighted atlas, record the closest accepted source "
            "view per texel after the exact same strict gates and nonzero-weight test. "
            "Distance ties prefer the larger original projection weight."
        ),
    )
    parser.add_argument("--scene-name", default="room_empty")
    parser.add_argument("--faces", default=None, help="Comma-separated face subset. Default: infer from polygon manifest.")
    parser.add_argument(
        "--views-per-face",
        type=int,
        default=0,
        help="0 means use every registered COLMAP view, matching the v53/v60 all-view projection loop.",
    )
    parser.add_argument("--chunk-size", type=int, default=180000)
    parser.add_argument("--min-view-cos", "--min-cos", dest="min_view_cos", type=float, default=0.10)
    parser.add_argument("--depth-abs-tol", type=float, default=0.045)
    parser.add_argument("--depth-rel-tol", type=float, default=0.035)
    parser.add_argument("--distance-weight-scale", "--distance-scale", dest="distance_weight_scale", type=float, default=1.15)
    parser.add_argument("--distance-weight-power", "--distance-power", dest="distance_weight_power", type=float, default=1.15)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--min-valid-views", type=int, default=1)
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--hole-dilate-px", type=int, default=1)
    parser.add_argument("--mask-boundary-safe-px", type=float, default=18.0)
    parser.add_argument("--mask-boundary-power", type=float, default=1.15)
    parser.add_argument(
        "--min-mask-boundary-trust",
        type=float,
        default=0.0,
        help=(
            "Hard-reject samples too close to a reject-mask boundary. "
            "0 keeps the legacy soft boundary weighting behavior."
        ),
    )
    parser.add_argument(
        "--object-risk-hard-thresh",
        type=float,
        default=1.01,
        help=(
            "Hard-reject samples inside the dilated/blurred object-risk field. "
            "Values above 1 disable this gate."
        ),
    )
    parser.add_argument("--footprint-min-area", type=float, default=0.32)
    parser.add_argument("--footprint-power", type=float, default=0.85)
    parser.add_argument(
        "--adaptive-short-face-footprint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Normalize the output footprint-quality reference for physically short wall faces. "
            "The per-sample v3b footprint weight is unchanged; only the final contamination gate "
            "is made invariant to the texture density of newly introduced narrow returns."
        ),
    )
    parser.add_argument("--short-face-length-median-frac", type=float, default=0.50)
    parser.add_argument("--short-face-footprint-median-multiplier", type=float, default=3.50)
    parser.add_argument("--short-face-footprint-min-area", type=float, default=0.008)
    parser.add_argument(
        "--adaptive-horizontal-footprint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Normalize the final footprint-quality reference for floor/ceiling when valid "
            "low-resolution projections are systematically smaller than the global reference. "
            "Per-sample projection weights and all geometry/object hard gates remain unchanged."
        ),
    )
    parser.add_argument("--horizontal-footprint-median-multiplier", type=float, default=1.25)
    parser.add_argument("--horizontal-footprint-min-area", type=float, default=0.008)
    parser.add_argument(
        "--surface-distance-tol",
        type=float,
        default=0.0,
        help=(
            "Optional 3D surface-consistency term in room coordinates. When >0, "
            "the sampled DA3 depth point is back-projected and compared against "
            "the current structural face plane."
        ),
    )
    parser.add_argument(
        "--surface-distance-clean-tol",
        type=float,
        default=0.0,
        help=(
            "Optional scale used only by the soft clean-score penalty for surface distance. "
            "Values <=0 reuse --surface-distance-tol. The hard geometric acceptance gate "
            "always continues to use --surface-distance-tol."
        ),
    )
    parser.add_argument("--surface-distance-power", type=float, default=1.0)
    parser.add_argument(
        "--surface-distance-hard-gate",
        action="store_true",
        help=(
            "Reject samples whose DA3 back-projected depth point is farther than "
            "--surface-distance-tol from the current structural face."
        ),
    )
    parser.add_argument(
        "--surface-normal-min-cos",
        type=float,
        default=0.0,
        help=(
            "Optional DA3 depth-normal consistency gate. When positive, a source pixel is "
            "accepted for a structural face only when its local depth normal has at least "
            "this absolute cosine with the face normal. This prevents floor/ceiling patches "
            "from being sampled from a nearby wall when depth scale is imperfect."
        ),
    )
    parser.add_argument("--color-std-clean-tol", type=float, default=0.18)
    parser.add_argument("--valid-ratio-penalty", type=float, default=0.45)
    parser.add_argument(
        "--min-sample-weight",
        type=float,
        default=0.0,
        help="Hard-reject individual projected samples below this final weight.",
    )
    parser.add_argument(
        "--min-output-reliability",
        type=float,
        default=0.0,
        help="When strict output is enabled, leave texels below this reliability empty.",
    )
    parser.add_argument(
        "--min-output-clean-score",
        type=float,
        default=0.0,
        help="When strict output is enabled, leave texels below this clean score empty.",
    )
    parser.add_argument(
        "--max-output-contamination-score",
        type=float,
        default=1.01,
        help="When strict output is enabled, leave texels above this contamination score empty.",
    )
    parser.add_argument(
        "--max-output-object-risk",
        type=float,
        default=1.01,
        help="When strict output is enabled, leave texels above this object risk empty.",
    )
    parser.add_argument(
        "--strict-empty-low-quality",
        action="store_true",
        help=(
            "Write black/empty texels for low-quality regions in textures/*.png "
            "instead of inpainting them into the projection preview."
        ),
    )
    parser.add_argument("--object-mask-dir", type=Path, default=None)
    parser.add_argument("--object-mask-dilate-px", type=int, default=2)
    parser.add_argument("--object-risk-dilate-px", type=int, default=4)
    parser.add_argument("--object-risk-blur-px", type=int, default=7)
    parser.add_argument("--zbuffer-stride", type=int, default=5)
    parser.add_argument("--icp-points-colmap", type=int, default=70000)
    parser.add_argument("--icp-points-da3", type=int, default=150000)
    parser.add_argument("--icp-trim-percentile", type=float, default=72.0)
    parser.add_argument("--icp-iters", type=int, default=28)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


@dataclass
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    score: float

    def colmap_to_da3(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (points @ self.rotation.T) + self.translation[None, :]

    def da3_to_colmap(self, points: np.ndarray) -> np.ndarray:
        return ((points - self.translation[None, :]) @ self.rotation) / max(self.scale, 1e-8)


@dataclass
class ImagePose:
    image_id: int
    name: str
    camera_id: int
    width: int
    height: int
    model: str
    params: np.ndarray
    Rcw: np.ndarray
    tcw: np.ndarray
    center_colmap: np.ndarray
    center_da3: np.ndarray
    image_path: Path


@dataclass
class Da3View:
    name: str
    depth: np.ndarray
    conf: np.ndarray | None
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    normal_world: np.ndarray | None = None
    normal_valid: np.ndarray | None = None


@dataclass
class Da3DepthCalibration:
    mode: str
    scale: float
    shift: float
    median_abs_error: float
    p80_abs_error: float
    samples: int


def identity_similarity() -> Similarity:
    return Similarity(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0.0)


def to_4x4(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat.copy()
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = mat
        return out
    raise ValueError(f"Unsupported camera matrix shape: {mat.shape}")


def da3_w2c_matrix(raw_extrinsic: np.ndarray) -> np.ndarray:
    """Return DA3 numeric extrinsics as a world-to-camera matrix.

    The current DA3 numeric exports used for this project store w2c matrices.
    Older v90-era code treated them as c2w and inverted them, which put the
    texture projection on the wrong side of the reconstructed room.
    """
    return to_4x4(raw_extrinsic)


def da3_c2w_matrix(raw_extrinsic: np.ndarray) -> np.ndarray:
    return np.linalg.inv(da3_w2c_matrix(raw_extrinsic))


def da3_w2c_camera_center(raw_extrinsic: np.ndarray) -> np.ndarray:
    raw = da3_w2c_matrix(raw_extrinsic)
    r = raw[:3, :3]
    t = raw[:3, 3]
    return (-r.T @ t).astype(np.float64)


def _find_key_recursive(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key_recursive(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_key_recursive(value, key)
            if found is not None:
                return found
    return None


def load_hf_alignment(scene_glb: Path) -> np.ndarray:
    """Read DA3's exported scene.glb hf_alignment matrix.

    DA3 writes point geometry after applying this alignment. Numeric extrinsics
    remain in DA3's raw world frame, so projection into the exported GLB shell
    must compose W2C_raw with inverse(hf_alignment).
    """
    from pygltflib import GLTF2

    gltf = GLTF2().load_binary(str(scene_glb))
    data = json.loads(gltf.to_json())
    raw = _find_key_recursive(data, "hf_alignment")
    if raw is None:
        raise ValueError(f"No hf_alignment metadata found in {scene_glb}")
    mat = np.asarray(raw, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 hf_alignment in {scene_glb}, got {mat.shape}")
    return mat


def da3_image_names(da3_dir: Path, n: int) -> list[str]:
    meta_path = da3_dir / "meta.json"
    names: list[str] = []
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw_names = meta.get("image_names") or meta.get("images") or meta.get("image_paths") or []
        names = [Path(str(item)).name for item in raw_names]
    if len(names) != n:
        names = [f"view_{idx:06d}" for idx in range(n)]
    return names


def scaled_da3_intrinsics_for_image(k: np.ndarray, depth_shape_hw: tuple[int, int], image_size_wh: tuple[int, int]) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64).copy()
    if k.shape == (4,):
        fx, fy, cx, cy = k
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"Unsupported DA3 intrinsic shape: {k.shape}")
    depth_h, depth_w = depth_shape_hw
    image_w, image_h = image_size_wh
    out = k.copy()
    # DA3 numeric intrinsics are normally in processed-image coordinates.
    # If the principal point is already outside the processed frame, treat the
    # matrix as original-image scale to avoid double scaling.
    if abs(k[0, 2]) <= depth_w * 1.25 and abs(k[1, 2]) <= depth_h * 1.25:
        out[0, :] *= float(image_w) / max(float(depth_w), 1.0)
        out[1, :] *= float(image_h) / max(float(depth_h), 1.0)
    return out


def load_da3_hfalign_poses(dataset_dir: Path, da3_dir: Path) -> tuple[list[ImagePose], Similarity, np.ndarray]:
    required = [da3_dir / name for name in ("depth.npy", "extrinsics.npy", "intrinsics.npy", "meta.json")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"DA3 hf-aligned projection requires missing files: {missing}")
    depth = np.load(da3_dir / "depth.npy")
    extrinsics = np.load(da3_dir / "extrinsics.npy").astype(np.float64)
    intrinsics = np.load(da3_dir / "intrinsics.npy").astype(np.float64)
    if depth.ndim != 3:
        raise ValueError(f"Expected DA3 depth shape (N,H,W), got {depth.shape}")
    if extrinsics.shape[0] != depth.shape[0] or intrinsics.shape[0] != depth.shape[0]:
        raise ValueError("DA3 depth/extrinsics/intrinsics have different view counts")
    hf_scene = da3_dir / "scene.glb"
    if not hf_scene.exists():
        hf_scene = dataset_dir / "scene.glb"
    hf_alignment = load_hf_alignment(hf_scene)
    inv_hf = np.linalg.inv(hf_alignment)
    names = da3_image_names(da3_dir, depth.shape[0])
    image_dir = dataset_dir / "input_images"
    poses: list[ImagePose] = []
    for idx, name in enumerate(names):
        image_path = image_dir / Path(name).name
        if not image_path.exists():
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        # DA3 stores raw world-to-camera matrices. The exported GLB points are
        # x_glb = hf_alignment * x_raw_world, so x_cam =
        # E_raw_w2c * inv(hf_alignment) * x_glb.
        e_glb = da3_w2c_matrix(extrinsics[idx]) @ inv_hf
        r = e_glb[:3, :3]
        t = e_glb[:3, 3]
        center = (-r.T @ t).astype(np.float64)
        k_img = scaled_da3_intrinsics_for_image(intrinsics[idx], tuple(depth.shape[1:3]), (width, height))
        poses.append(
            ImagePose(
                image_id=int(idx),
                name=Path(name).name,
                camera_id=int(idx),
                width=int(width),
                height=int(height),
                model="PINHOLE",
                params=np.array([k_img[0, 0], k_img[1, 1], k_img[0, 2], k_img[1, 2]], dtype=np.float64),
                Rcw=r.astype(np.float64),
                tcw=t.astype(np.float64),
                center_colmap=center,
                center_da3=center,
                image_path=image_path,
            )
        )
    if not poses:
        raise RuntimeError(f"No DA3 hf-aligned poses matched images under {image_dir}")
    poses.sort(key=lambda p: p.name)
    return poses, identity_similarity(), hf_alignment


def load_da3_raw_poses(dataset_dir: Path, da3_dir: Path) -> tuple[list[ImagePose], Similarity]:
    required = [da3_dir / name for name in ("depth.npy", "extrinsics.npy", "intrinsics.npy", "meta.json")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"DA3 raw projection requires missing files: {missing}")
    depth = np.load(da3_dir / "depth.npy")
    extrinsics = np.load(da3_dir / "extrinsics.npy").astype(np.float64)
    intrinsics = np.load(da3_dir / "intrinsics.npy").astype(np.float64)
    names = da3_image_names(da3_dir, depth.shape[0])
    image_dir = dataset_dir / "input_images"
    poses: list[ImagePose] = []
    for idx, name in enumerate(names):
        image_path = image_dir / Path(name).name
        if not image_path.exists():
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        e_raw = da3_w2c_matrix(extrinsics[idx])
        r = e_raw[:3, :3]
        t = e_raw[:3, 3]
        center = (-r.T @ t).astype(np.float64)
        k_img = scaled_da3_intrinsics_for_image(intrinsics[idx], tuple(depth.shape[1:3]), (width, height))
        poses.append(
            ImagePose(
                image_id=int(idx),
                name=Path(name).name,
                camera_id=int(idx),
                width=int(width),
                height=int(height),
                model="PINHOLE",
                params=np.array([k_img[0, 0], k_img[1, 1], k_img[0, 2], k_img[1, 2]], dtype=np.float64),
                Rcw=r.astype(np.float64),
                tcw=t.astype(np.float64),
                center_colmap=center,
                center_da3=center,
                image_path=image_path,
            )
        )
    if not poses:
        raise RuntimeError(f"No DA3 raw poses matched images under {image_dir}")
    poses.sort(key=lambda p: p.name)
    return poses, identity_similarity()


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mu = np.mean(points, axis=0)
    x = points - mu[None, :]
    cov = (x.T @ x) / max(1, len(points))
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    basis = vecs[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1.0
    rms = float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
    return mu.astype(np.float64), basis.astype(np.float64), max(rms, 1e-8)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Similarity:
    src_mu = src.mean(axis=0)
    dst_mu = dst.mean(axis=0)
    src_c = src - src_mu[None, :]
    dst_c = dst - dst_mu[None, :]
    cov = (dst_c.T @ src_c) / max(1, src.shape[0])
    u, svals, vt = np.linalg.svd(cov)
    d = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    var = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.sum(svals * d) / max(var, 1e-10))
    trans = dst_mu - scale * (rot @ src_mu)
    return Similarity(scale, rot, trans, 0.0)


def subsample(points: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= limit:
        return points.astype(np.float64)
    idx = rng.choice(points.shape[0], limit, replace=False)
    return points[idx].astype(np.float64)


def align_colmap_to_da3(colmap_points: np.ndarray, da3_points: np.ndarray, args: argparse.Namespace) -> Similarity:
    rng = np.random.default_rng(args.seed)
    src = subsample(colmap_points, args.icp_points_colmap, rng)
    dst = subsample(da3_points, args.icp_points_da3, rng)
    dst_tree = cKDTree(dst)
    src_mu, src_basis, src_rms = pca_basis(src)
    dst_mu, dst_basis, dst_rms = pca_basis(dst)

    candidates: list[Similarity] = []
    for signs in itertools.product([-1.0, 1.0], repeat=3):
        sign = np.diag(signs)
        if np.linalg.det(sign) < 0:
            continue
        rot = dst_basis @ sign @ src_basis.T
        scale = dst_rms / src_rms
        trans = dst_mu - scale * (rot @ src_mu)
        candidates.append(Similarity(scale, rot, trans, float("inf")))

    best: Similarity | None = None
    for init in candidates:
        sim = init
        for _ in range(args.icp_iters):
            moved = sim.colmap_to_da3(src)
            dist, nn = dst_tree.query(moved, k=1, workers=-1)
            trim = float(np.percentile(dist, args.icp_trim_percentile))
            keep = np.isfinite(dist) & (dist <= max(trim, 1e-8))
            if np.count_nonzero(keep) < 32:
                break
            sim = umeyama_similarity(src[keep], dst[nn[keep]])
        moved = sim.colmap_to_da3(src)
        dist, _ = dst_tree.query(moved, k=1, workers=-1)
        score = float(np.percentile(dist, 65.0))
        sim.score = score
        if best is None or score < best.score:
            best = sim
    if best is None:
        raise RuntimeError("Could not align COLMAP sparse cloud to DA3 point cloud")
    return best


def pycolmap_reconstruction(model_dir: Path):
    import pycolmap

    return pycolmap.Reconstruction(str(model_dir))


def rigid_matrix_from_image(image) -> np.ndarray:
    pose_attr = getattr(image, "cam_from_world", None)
    pose = pose_attr() if callable(pose_attr) else pose_attr
    if hasattr(pose, "matrix"):
        mat = np.asarray(pose.matrix(), dtype=np.float64)
        if mat.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :4] = mat
            return out
        return mat
    qvec = np.asarray(getattr(image, "qvec"), dtype=np.float64)
    tvec = np.asarray(getattr(image, "tvec"), dtype=np.float64)
    qw, qx, qy, qz = qvec
    rot = np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = tvec
    return out


def camera_model_name(camera) -> str:
    name = str(camera.model)
    if "." in name:
        name = name.split(".")[-1]
    return name.upper()


def load_image_poses(rec, dataset_dir: Path, sim: Similarity) -> list[ImagePose]:
    image_dir = dataset_dir / "input_images"
    poses: list[ImagePose] = []
    for image_id, image in rec.images.items():
        has_pose_attr = getattr(image, "has_pose", True)
        has_pose = has_pose_attr() if callable(has_pose_attr) else bool(has_pose_attr)
        if not has_pose:
            continue
        camera = rec.cameras[image.camera_id]
        path = image_dir / image.name
        if not path.exists():
            alt = image_dir / Path(image.name).name
            path = alt
        if not path.exists():
            continue
        mat = rigid_matrix_from_image(image)
        Rcw = np.asarray(mat[:3, :3], dtype=np.float64)
        tcw = np.asarray(mat[:3, 3], dtype=np.float64)
        center = -Rcw.T @ tcw
        poses.append(
            ImagePose(
                image_id=int(image_id),
                name=str(image.name),
                camera_id=int(image.camera_id),
                width=int(camera.width),
                height=int(camera.height),
                model=camera_model_name(camera),
                params=np.asarray(camera.params, dtype=np.float64),
                Rcw=Rcw,
                tcw=tcw,
                center_colmap=center,
                center_da3=sim.colmap_to_da3(center[None, :])[0],
                image_path=path,
            )
        )
    poses.sort(key=lambda p: p.name)
    if not poses:
        raise RuntimeError("No registered COLMAP images with matching input image files")
    return poses


def project_points(points_colmap: np.ndarray, pose: ImagePose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = points_colmap @ pose.Rcw.T + pose.tcw[None, :]
    z = cam[:, 2]
    x = cam[:, 0] / np.maximum(z, 1e-8)
    y = cam[:, 1] / np.maximum(z, 1e-8)
    p = pose.params
    model = pose.model
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        u = f * x + cx
        v = f * y + cy
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        u = fx * x + cx
        v = fy * y + cy
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = p[:4]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2
        u = f * radial * x + cx
        v = f * radial * y + cy
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = p[:5]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        u = f * radial * x + cx
        v = f * radial * y + cy
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        u = fx * x_d + cx
        v = fy * y_d + cy
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)


def backproject_camera_z_to_world(u: np.ndarray, v: np.ndarray, z: np.ndarray, pose: ImagePose) -> np.ndarray | None:
    p = pose.params
    if pose.model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        fx = fy = float(f)
    elif pose.model == "PINHOLE":
        fx, fy, cx, cy = map(float, p[:4])
    else:
        # Distorted COLMAP models need iterative undistortion. The DA3-hfalign
        # path uses PINHOLE, so keep legacy COLMAP projection unchanged.
        return None
    zz = np.asarray(z, dtype=np.float64)
    x = (np.asarray(u, dtype=np.float64) - float(cx)) / max(fx, 1e-8) * zz
    y = (np.asarray(v, dtype=np.float64) - float(cy)) / max(fy, 1e-8) * zz
    cam = np.stack([x, y, zz], axis=1)
    world = (cam - pose.tcw[None, :]) @ pose.Rcw
    return world.astype(np.float64)


def backproject_da3_raw_depth_to_room(
    du: np.ndarray,
    dv: np.ndarray,
    raw_depth: np.ndarray,
    view: Da3View,
    raw_to_room_matrix: np.ndarray | None = None,
    raw_to_room_similarity: Similarity | None = None,
) -> np.ndarray | None:
    k = np.asarray(view.intrinsic, dtype=np.float64)
    if k.shape == (4,):
        fx, fy, cx, cy = map(float, k)
        skew = 0.0
    elif k.shape == (3, 3):
        fx, skew, cx = map(float, k[0, :3])
        fy, cy = float(k[1, 1]), float(k[1, 2])
    else:
        return None
    zz = np.asarray(raw_depth, dtype=np.float64)
    valid = np.isfinite(zz) & (zz > 1e-8)
    if not np.any(valid):
        return None
    y = (np.asarray(dv, dtype=np.float64) - cy) / max(fy, 1e-8) * zz
    x = (np.asarray(du, dtype=np.float64) - cx - skew * (y / np.maximum(zz, 1e-8))) / max(fx, 1e-8) * zz
    cam = np.stack([x, y, zz], axis=1)
    r = view.extrinsic[:3, :3]
    t = view.extrinsic[:3, 3]
    raw_world = cam @ r.T + t[None, :]
    if raw_to_room_matrix is not None:
        homog = np.concatenate([raw_world, np.ones((raw_world.shape[0], 1), dtype=np.float64)], axis=1)
        room = homog @ raw_to_room_matrix.T
        w = np.maximum(np.abs(room[:, 3:4]), 1e-8)
        return (room[:, :3] / w).astype(np.float64)
    if raw_to_room_similarity is not None:
        return raw_to_room_similarity.colmap_to_da3(raw_world).astype(np.float64)
    return raw_world.astype(np.float64)


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    dx = (u - x0).astype(np.float32)
    dy = (v - y0).astype(np.float32)
    c00 = image[y0, x0]
    c10 = image[y0, x1]
    c01 = image[y1, x0]
    c11 = image[y1, x1]
    return (
        c00 * (1.0 - dx)[:, None] * (1.0 - dy)[:, None]
        + c10 * dx[:, None] * (1.0 - dy)[:, None]
        + c01 * (1.0 - dx)[:, None] * dy[:, None]
        + c11 * dx[:, None] * dy[:, None]
    )


def sample_float_map(image: np.ndarray, u: np.ndarray, v: np.ndarray, border_value: float = 0.0) -> np.ndarray:
    h, w = image.shape[:2]
    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    out = np.full(u.shape, float(border_value), dtype=np.float32)
    if not np.any(inside):
        return out
    uu = u[inside]
    vv = v[inside]
    x0 = np.floor(uu).astype(np.int32)
    y0 = np.floor(vv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    dx = (uu - x0).astype(np.float32)
    dy = (vv - y0).astype(np.float32)
    im = image.astype(np.float32)
    vals = (
        im[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + im[y0, x1] * dx * (1.0 - dy)
        + im[y1, x0] * (1.0 - dx) * dy
        + im[y1, x1] * dx * dy
    )
    out[inside] = vals.astype(np.float32)
    return out


def sample_nearest_map(image: np.ndarray, u: np.ndarray, v: np.ndarray, border_value: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    out = np.full(u.shape, int(border_value), dtype=np.uint8)
    if not np.any(inside):
        return out
    x = np.clip(np.round(u[inside]).astype(np.int32), 0, w - 1)
    y = np.clip(np.round(v[inside]).astype(np.int32), 0, h - 1)
    out[inside] = image.astype(np.uint8)[y, x]
    return out


def depth_normal_map_world(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    max_relative_depth_jump: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a robust local surface normal from a DA3 camera-z depth map."""
    z = np.asarray(depth, dtype=np.float64)
    h, w = z.shape
    k = np.asarray(intrinsic, dtype=np.float64)
    if k.shape == (4,):
        fx, fy, cx, cy = map(float, k)
        skew = 0.0
    elif k.shape == (3, 3):
        fx, skew, cx = map(float, k[0, :3])
        fy, cy = float(k[1, 1]), float(k[1, 2])
    else:
        return np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    yy, xx = np.mgrid[:h, :w]
    y_cam = (yy.astype(np.float64) - cy) / max(fy, 1e-8) * z
    x_cam = (
        xx.astype(np.float64)
        - cx
        - skew * (y_cam / np.maximum(z, 1e-8))
    ) / max(fx, 1e-8) * z
    points = np.stack([x_cam, y_cam, z], axis=2)

    tangent_x = points[1:-1, 2:] - points[1:-1, :-2]
    tangent_y = points[2:, 1:-1] - points[:-2, 1:-1]
    normals_camera = np.cross(tangent_x, tangent_y)
    norm = np.linalg.norm(normals_camera, axis=2)
    center_depth = z[1:-1, 1:-1]
    relative_jump = np.maximum(
        np.abs(z[1:-1, 2:] - z[1:-1, :-2]),
        np.abs(z[2:, 1:-1] - z[:-2, 1:-1]),
    ) / np.maximum(center_depth, 1e-6)
    valid = (
        np.isfinite(center_depth)
        & (center_depth > 1e-6)
        & np.isfinite(norm)
        & (norm > 1e-10)
        & np.isfinite(relative_jump)
        & (relative_jump <= float(max_relative_depth_jump))
    )
    normals_camera = normals_camera / np.maximum(norm[..., None], 1e-10)
    rotation = np.asarray(camera_to_world, dtype=np.float64)[:3, :3]
    normals_world_inner = normals_camera @ rotation.T
    normals_world_inner /= np.maximum(
        np.linalg.norm(normals_world_inner, axis=2, keepdims=True),
        1e-10,
    )

    normals_world = np.zeros((h, w, 3), dtype=np.float32)
    normal_valid = np.zeros((h, w), dtype=np.float32)
    normals_world[1:-1, 1:-1] = normals_world_inner.astype(np.float32)
    normal_valid[1:-1, 1:-1] = valid.astype(np.float32)
    normals_world[normal_valid < 0.5] = 0.0
    return normals_world, normal_valid


def sampled_da3_normal_cos(
    view: Da3View | None,
    u: np.ndarray,
    v: np.ndarray,
    face_normal_world: np.ndarray,
    raw_to_room_matrix: np.ndarray | None = None,
    raw_to_room_similarity: Similarity | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample DA3 normals and compare them with a structural face normal."""
    if view is None or view.normal_world is None or view.normal_valid is None:
        return np.zeros(np.asarray(u).shape, dtype=np.float32), np.zeros(np.asarray(u).shape, dtype=bool)
    components = [
        sample_float_map(view.normal_world[..., axis], u, v, border_value=0.0)
        for axis in range(3)
    ]
    normals = np.stack(components, axis=1).astype(np.float64)
    validity = sample_float_map(view.normal_valid, u, v, border_value=0.0) >= 0.75

    if raw_to_room_matrix is not None:
        linear = np.asarray(raw_to_room_matrix, dtype=np.float64)[:3, :3]
        normals = normals @ linear.T
    elif raw_to_room_similarity is not None:
        normals = normals @ np.asarray(raw_to_room_similarity.rotation, dtype=np.float64).T
    normal_length = np.linalg.norm(normals, axis=1)
    validity &= np.isfinite(normal_length) & (normal_length > 1e-6)
    normals /= np.maximum(normal_length[:, None], 1e-8)
    target = np.asarray(face_normal_world, dtype=np.float64)
    target /= max(float(np.linalg.norm(target)), 1e-8)
    cosine = np.abs(normals @ target).astype(np.float32)
    cosine[~validity] = 0.0
    return cosine, validity


def load_da3_views(da3_dir: Path | None, poses: list[ImagePose]) -> dict[int, Da3View]:
    if da3_dir is None:
        return {}
    required = [da3_dir / name for name in ("depth.npy", "extrinsics.npy", "intrinsics.npy")]
    if not all(path.exists() for path in required):
        return {}

    depth = np.load(da3_dir / "depth.npy").astype(np.float32)
    conf_path = da3_dir / "conf.npy"
    conf = np.load(conf_path).astype(np.float32) if conf_path.exists() else None
    extrinsics = np.load(da3_dir / "extrinsics.npy").astype(np.float64)
    intrinsics = np.load(da3_dir / "intrinsics.npy").astype(np.float64)

    if depth.ndim != 3:
        raise ValueError(f"Expected DA3 depth shape (N,H,W), got {depth.shape}")
    if conf is not None and conf.shape != depth.shape:
        raise ValueError(f"DA3 conf shape {conf.shape} does not match depth shape {depth.shape}")
    if extrinsics.shape[0] != depth.shape[0] or intrinsics.shape[0] != depth.shape[0]:
        raise ValueError("DA3 depth/extrinsics/intrinsics have different view counts")

    names: list[str] = []
    meta_path = da3_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        raw_names = meta.get("images") or meta.get("image_paths") or meta.get("image_names") or []
        names = [Path(str(item)).name for item in raw_names]
    if len(names) != depth.shape[0]:
        names = [f"view_{i:06d}" for i in range(depth.shape[0])]

    by_key: dict[str, int] = {}
    for idx, name in enumerate(names):
        p = Path(name)
        for key in {p.name, p.stem, f"view_{idx:03d}", f"view_{idx:06d}"}:
            by_key[key] = idx

    out: dict[int, Da3View] = {}
    for order, pose in enumerate(poses):
        pose_path = Path(pose.name)
        candidates = [
            pose_path.name,
            pose_path.stem,
            pose.image_path.name,
            pose.image_path.stem,
            f"view_{pose.image_id:03d}",
            f"view_{pose.image_id:06d}",
        ]
        idx = next((by_key[key] for key in candidates if key in by_key), None)
        if idx is None and order < depth.shape[0]:
            idx = order
        if idx is None:
            continue
        ex = extrinsics[idx]
        if ex.shape not in {(3, 4), (4, 4)}:
            raise ValueError(f"Unsupported DA3 extrinsic shape for view {idx}: {ex.shape}")
        c2w = da3_c2w_matrix(ex)
        normals_world, normal_valid = depth_normal_map_world(depth[idx], intrinsics[idx], c2w)
        out[pose.image_id] = Da3View(
            name=names[idx],
            depth=depth[idx],
            conf=conf[idx] if conf is not None else None,
            extrinsic=c2w,
            intrinsic=intrinsics[idx],
            normal_world=normals_world,
            normal_valid=normal_valid,
        )
    return out


def project_da3_points(points_da3: np.ndarray, view: Da3View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = view.extrinsic[:3, :3]
    t = view.extrinsic[:3, 3]
    cam = (points_da3 - t[None, :]) @ r
    z = cam[:, 2]
    x = cam[:, 0] / np.maximum(z, 1e-8)
    y = cam[:, 1] / np.maximum(z, 1e-8)
    k = view.intrinsic
    u = k[0, 0] * x + k[0, 1] * y + k[0, 2]
    v = k[1, 0] * x + k[1, 1] * y + k[1, 2]
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)


def da3_uv_from_colmap_pixels(
    u: np.ndarray,
    v: np.ndarray,
    pose: ImagePose,
    view: Da3View,
) -> tuple[np.ndarray, np.ndarray]:
    dh, dw = view.depth.shape[:2]
    sx = float(dw) / max(float(pose.width), 1.0)
    sy = float(dh) / max(float(pose.height), 1.0)
    return (u.astype(np.float32) * sx), (v.astype(np.float32) * sy)


def fit_depth_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    a = np.column_stack([x.astype(np.float64), np.ones_like(x, dtype=np.float64)])
    scale, shift = np.linalg.lstsq(a, y.astype(np.float64), rcond=None)[0]
    pred = scale * x.astype(np.float64) + shift
    err = np.abs(pred - y.astype(np.float64))
    return float(scale), float(shift), err


def calibrate_da3_depth_to_colmap_zbuffer(
    view: Da3View,
    pose: ImagePose,
    zbuf: np.ndarray,
    stride: int = 8,
    max_samples: int = 70000,
) -> Da3DepthCalibration | None:
    yy, xx = np.mgrid[0 : pose.height : stride, 0 : pose.width : stride]
    z = zbuf[yy, xx].astype(np.float32).reshape(-1)
    x = xx.astype(np.float32).reshape(-1)
    y = yy.astype(np.float32).reshape(-1)
    du, dv = da3_uv_from_colmap_pixels(x, y, pose, view)
    d = sample_float_map(view.depth, du, dv, border_value=np.nan)
    valid = np.isfinite(z) & np.isfinite(d) & (z > 1e-6) & (d > 1e-6)
    if int(np.count_nonzero(valid)) < 256:
        return None
    z = z[valid].astype(np.float64)
    d = d[valid].astype(np.float64)
    if z.size > max_samples:
        pick = np.linspace(0, z.size - 1, max_samples).astype(np.int64)
        z = z[pick]
        d = d[pick]

    candidates: list[tuple[str, float, float, np.ndarray]] = []
    for mode, xvals in (
        ("linear", d),
        ("inverse", 1.0 / np.maximum(d, 1e-6)),
    ):
        scale, shift, err = fit_depth_affine(xvals, z)
        if not np.isfinite(scale) or not np.isfinite(shift):
            continue
        keep = err <= np.percentile(err, 80.0)
        if int(np.count_nonzero(keep)) >= 128:
            scale, shift, err_refit = fit_depth_affine(xvals[keep], z[keep])
            pred = scale * xvals + shift
            err = np.abs(pred - z)
        candidates.append((mode, scale, shift, err))
    if not candidates:
        return None
    mode, scale, shift, err = min(candidates, key=lambda item: float(np.median(item[3])))
    if not np.isfinite(scale) or not np.isfinite(shift):
        return None
    return Da3DepthCalibration(
        mode=mode,
        scale=float(scale),
        shift=float(shift),
        median_abs_error=float(np.median(err)),
        p80_abs_error=float(np.percentile(err, 80.0)),
        samples=int(z.size),
    )


def apply_depth_calibration(raw_depth: np.ndarray, calib: Da3DepthCalibration) -> np.ndarray:
    if calib.mode == "inverse":
        x = 1.0 / np.maximum(raw_depth, 1e-6)
    else:
        x = raw_depth
    return (calib.scale * x + calib.shift).astype(np.float32)


def da3_camera_center(view: Da3View) -> np.ndarray:
    return view.extrinsic[:3, 3].astype(np.float64)


def align_da3_numeric_world_to_room(da3_views: dict[int, Da3View], poses: list[ImagePose]) -> Similarity | None:
    numeric_centers = []
    room_centers = []
    for pose in poses:
        view = da3_views.get(pose.image_id)
        if view is None:
            continue
        numeric_centers.append(da3_camera_center(view))
        room_centers.append(pose.center_da3.astype(np.float64))
    if len(numeric_centers) < 3:
        return None
    src = np.asarray(numeric_centers, dtype=np.float64)
    dst = np.asarray(room_centers, dtype=np.float64)
    sim = umeyama_similarity(src, dst)
    pred = sim.colmap_to_da3(src)
    err = np.linalg.norm(pred - dst, axis=1)
    sim.score = float(np.percentile(err, 65.0))
    return sim


def candidate_mask_paths(mask_dir: Path, image_name: str, image_id: int) -> list[Path]:
    stem = Path(image_name).stem
    return [
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{stem}_object_mask.png",
        mask_dir / f"view_{image_id:03d}_object_mask.png",
        mask_dir / f"view_{image_id:06d}_object_mask.png",
        mask_dir / f"{image_id:06d}.png",
        mask_dir / f"{image_id:03d}.png",
    ]


def load_view_reject_maps(pose: ImagePose, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty_object = np.zeros((pose.height, pose.width), dtype=np.uint8)
    empty_risk = np.zeros((pose.height, pose.width), dtype=np.float32)
    full_boundary = np.ones((pose.height, pose.width), dtype=np.float32)
    if args.object_mask_dir is None:
        return empty_object, empty_risk, full_boundary
    path = next((p for p in candidate_mask_paths(args.object_mask_dir, pose.name, pose.image_id) if p.exists()), None)
    if path is None:
        return empty_object, empty_risk, full_boundary
    mask = np.asarray(Image.open(path).convert("L").resize((pose.width, pose.height), RESAMPLE_NEAREST), dtype=np.uint8)
    mask = (mask > 0).astype(np.uint8)
    if args.object_mask_dilate_px > 0 and np.any(mask):
        k = 2 * int(args.object_mask_dilate_px) + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1)
    risk = mask.astype(np.float32)
    if args.object_risk_dilate_px > 0 and np.any(mask):
        k = 2 * int(args.object_risk_dilate_px) + 1
        risk = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=1).astype(np.float32)
    blur_k = int(args.object_risk_blur_px)
    if blur_k > 1 and np.any(risk):
        if blur_k % 2 == 0:
            blur_k += 1
        risk = cv2.GaussianBlur(risk, (blur_k, blur_k), 0)
        risk = np.maximum(risk, mask.astype(np.float32))
        risk = np.clip(risk, 0.0, 1.0)
    if args.mask_boundary_safe_px > 0 and np.any(mask):
        free = (mask == 0).astype(np.uint8)
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 5).astype(np.float32)
        boundary = np.clip(dist / max(float(args.mask_boundary_safe_px), 1e-6), 0.0, 1.0)
        if args.mask_boundary_power > 0.0 and args.mask_boundary_power != 1.0:
            boundary = np.power(boundary, float(args.mask_boundary_power))
        boundary[mask > 0] = 0.0
    else:
        boundary = full_boundary
    return mask.astype(np.uint8), risk.astype(np.float32), boundary.astype(np.float32)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def face_names(manifest: dict, faces_arg: str | None) -> list[str]:
    if faces_arg:
        return [x.strip() for x in faces_arg.split(",") if x.strip()]
    names = [meta["face"] for meta in manifest["faces"]]
    ordered = [f for f in ("floor", "ceiling") if f in names]
    ordered.extend(sorted(f for f in names if f.startswith("wall_")))
    seen = set(ordered)
    ordered.extend([f for f in names if f not in seen])
    return ordered


def face_meta_map(manifest: dict) -> dict[str, dict]:
    return {meta["face"]: meta for meta in manifest["faces"]}


def face_texture_size(source_dir: Path, meta: dict) -> tuple[int, int]:
    path = source_dir / "textures" / f"{meta['face']}.png"
    if path.exists():
        with Image.open(path) as im:
            return im.size
    return tuple(map(int, meta["texture_size"]))


def has_room_basis(manifest: dict) -> bool:
    return "world_from_room_matrix" in manifest


def world_from_room_points(local_xyz: np.ndarray, manifest: dict) -> np.ndarray:
    mat = np.asarray(manifest["world_from_room_matrix"], dtype=np.float64)
    homog = np.concatenate(
        [local_xyz.astype(np.float64), np.ones((local_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    out = homog @ mat.T
    w = np.maximum(np.abs(out[:, 3:4]), 1e-8)
    return out[:, :3] / w


def room_from_world_points(world_xyz: np.ndarray, manifest: dict) -> np.ndarray:
    inv = np.asarray(manifest.get("room_from_world_matrix"), dtype=np.float64)
    if inv.shape != (4, 4):
        inv = np.linalg.inv(np.asarray(manifest["world_from_room_matrix"], dtype=np.float64))
    homog = np.concatenate(
        [world_xyz.astype(np.float64), np.ones((world_xyz.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    out = homog @ inv.T
    w = np.maximum(np.abs(out[:, 3:4]), 1e-8)
    return out[:, :3] / w


def world_from_room_vectors(local_vecs: np.ndarray, manifest: dict) -> np.ndarray:
    rot = np.asarray(manifest["world_from_room_matrix"], dtype=np.float64)[:3, :3]
    out = local_vecs.astype(np.float64) @ rot.T
    denom = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out / denom


def original_from_rotated(rot_xy: np.ndarray, height: np.ndarray, manifest: dict) -> np.ndarray:
    up_axis = int(manifest["up_axis"])
    h_axes = [int(x) for x in manifest["horizontal_axes"]]
    theta = float(manifest["manhattan_rotation_theta_rad"])
    horiz0 = rotate2(rot_xy.astype(np.float32), -theta).astype(np.float64)
    out = np.zeros((rot_xy.shape[0], 3), dtype=np.float64)
    out[:, h_axes[0]] = horiz0[:, 0]
    out[:, h_axes[1]] = horiz0[:, 1]
    out[:, up_axis] = height
    return out


def face_points_for_indices(face: str, rows: np.ndarray, cols: np.ndarray, size: tuple[int, int], manifest: dict, metas: dict) -> np.ndarray:
    w, h = size
    if face in {"floor", "ceiling"}:
        bounds = np.asarray(manifest["bounds_uv"], dtype=np.float64)
        span = np.maximum(bounds[1] - bounds[0], 1e-8)
        u = (cols.astype(np.float64) + 0.5) / float(w)
        v = (rows.astype(np.float64) + 0.5) / float(h)
        rot_xy = np.stack([bounds[0, 0] + u * span[0], bounds[0, 1] + v * span[1]], axis=1)
        y = np.full(len(rows), float(manifest["floor_y"] if face == "floor" else manifest["ceiling_y"]), dtype=np.float64)
        if has_room_basis(manifest):
            local = np.column_stack([rot_xy[:, 0], y, rot_xy[:, 1]])
            return world_from_room_points(local, manifest)
        return original_from_rotated(rot_xy, y, manifest)
    meta = metas[face]
    a = np.asarray(meta["edge_start"], dtype=np.float64)
    b = np.asarray(meta["edge_end"], dtype=np.float64)
    edge = b - a
    length = max(float(np.linalg.norm(edge)), 1e-8)
    edir = edge / length
    room_h = float(manifest["height"])
    along = (cols.astype(np.float64) + 0.5) / float(w) * length
    height = float(manifest["floor_y"]) + (1.0 - (rows.astype(np.float64) + 0.5) / float(h)) * room_h
    rot_xy = a[None, :] + along[:, None] * edir[None, :]
    if has_room_basis(manifest):
        local = np.column_stack([rot_xy[:, 0], height, rot_xy[:, 1]])
        return world_from_room_points(local, manifest)
    return original_from_rotated(rot_xy, height, manifest)


def face_normal(face: str, manifest: dict, metas: dict) -> np.ndarray:
    if has_room_basis(manifest):
        if face == "floor":
            return world_from_room_vectors(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), manifest)[0]
        if face == "ceiling":
            return world_from_room_vectors(np.array([[0.0, -1.0, 0.0]], dtype=np.float64), manifest)[0]
        meta = metas[face]
        a = np.asarray(meta["edge_start"], dtype=np.float64)
        b = np.asarray(meta["edge_end"], dtype=np.float64)
        e = b - a
        e = e / max(float(np.linalg.norm(e)), 1e-8)
        n2 = np.array([-e[1], e[0]], dtype=np.float64)
        return world_from_room_vectors(np.array([[n2[0], 0.0, n2[1]]], dtype=np.float64), manifest)[0]
    up_axis = int(manifest["up_axis"])
    h_axes = [int(x) for x in manifest["horizontal_axes"]]
    if face == "floor":
        n = np.zeros(3, dtype=np.float64)
        n[up_axis] = 1.0
        return n
    if face == "ceiling":
        n = np.zeros(3, dtype=np.float64)
        n[up_axis] = -1.0
        return n
    meta = metas[face]
    a = np.asarray(meta["edge_start"], dtype=np.float64)
    b = np.asarray(meta["edge_end"], dtype=np.float64)
    e = b - a
    e = e / max(float(np.linalg.norm(e)), 1e-8)
    n2 = np.array([-e[1], e[0]], dtype=np.float32)[None, :]
    theta = float(manifest["manhattan_rotation_theta_rad"])
    n2_orig = rotate2(n2, -theta)[0].astype(np.float64)
    n = np.zeros(3, dtype=np.float64)
    n[h_axes[0]] = n2_orig[0]
    n[h_axes[1]] = n2_orig[1]
    return n / max(float(np.linalg.norm(n)), 1e-8)


def rotate_points_to_floorplan(points: np.ndarray, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    if has_room_basis(manifest):
        local = room_from_world_points(points, manifest)
        return local[:, [0, 2]].astype(np.float64), local[:, 1].astype(np.float64)
    up_axis = int(manifest["up_axis"])
    h_axes = [int(x) for x in manifest["horizontal_axes"]]
    theta = float(manifest["manhattan_rotation_theta_rad"])
    horiz0 = points[:, h_axes].astype(np.float32)
    horiz = rotate2(horiz0, theta).astype(np.float64)
    heights = points[:, up_axis].astype(np.float64)
    return horiz, heights


def face_surface_distance(points_da3: np.ndarray, face: str, manifest: dict, metas: dict) -> np.ndarray:
    if points_da3.size == 0:
        return np.zeros(0, dtype=np.float32)
    horiz, heights = rotate_points_to_floorplan(points_da3, manifest)
    if face == "floor":
        return np.abs(heights - float(manifest["floor_y"])).astype(np.float32)
    if face == "ceiling":
        return np.abs(heights - float(manifest["ceiling_y"])).astype(np.float32)
    meta = metas[face]
    a = np.asarray(meta["edge_start"], dtype=np.float64)
    b = np.asarray(meta["edge_end"], dtype=np.float64)
    edge = b - a
    length = max(float(np.linalg.norm(edge)), 1e-8)
    e = edge / length
    rel = horiz - a[None, :]
    perp = np.abs(rel[:, 0] * e[1] - rel[:, 1] * e[0])
    return perp.astype(np.float32)


def valid_indices_for_face(source_dir: Path, face: str, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    mask_path = source_dir / "debug" / f"{face}_valid_mask.png"
    if mask_path.exists():
        mask = np.asarray(Image.open(mask_path).convert("L").resize((w, h), RESAMPLE_NEAREST)) > 0
    else:
        mask = np.ones((h, w), dtype=bool)
    return np.flatnonzero(mask.reshape(-1))


def choose_views_for_face(face: str, size: tuple[int, int], poses: list[ImagePose], sim: Similarity, manifest: dict, metas: dict, args: argparse.Namespace) -> list[ImagePose]:
    if args.views_per_face <= 0:
        return poses
    w, h = size
    sample_rows = np.array([h * 0.25, h * 0.50, h * 0.75], dtype=np.int32)
    sample_cols = np.array([w * 0.25, w * 0.50, w * 0.75], dtype=np.int32)
    rr, cc = np.meshgrid(np.clip(sample_rows, 0, h - 1), np.clip(sample_cols, 0, w - 1), indexing="ij")
    pts_da3 = face_points_for_indices(face, rr.reshape(-1), cc.reshape(-1), size, manifest, metas)
    pts_col = sim.da3_to_colmap(pts_da3)
    n = face_normal(face, manifest, metas)
    scored = []
    for pose in poses:
        u, v, z = project_points(pts_col, pose)
        inside = (z > 0.05) & (u >= 8) & (u < pose.width - 8) & (v >= 8) & (v < pose.height - 8)
        if np.count_nonzero(inside) == 0:
            continue
        view = pose.center_da3[None, :] - pts_da3
        dist = np.linalg.norm(view, axis=1)
        view = view / np.maximum(dist[:, None], 1e-8)
        cos = np.abs(view @ n)
        score = float(
            np.mean(cos[inside])
            * np.mean(inside.astype(np.float32))
            / (1.0 + np.mean(dist[inside]) / max(args.distance_weight_scale, 1e-6))
        )
        if score > 0:
            scored.append((score, pose))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[: args.views_per_face]]


def build_zbuffer(
    pose: ImagePose,
    faces: list[str],
    source_dir: Path,
    sim: Similarity,
    manifest: dict,
    metas: dict,
    stride: int,
) -> np.ndarray:
    zbuf = np.full((pose.height, pose.width), np.inf, dtype=np.float32)
    for face in faces:
        size = face_texture_size(source_dir, metas[face])
        w, h = size
        rows = np.arange(0, h, max(1, stride), dtype=np.int32)
        cols = np.arange(0, w, max(1, stride), dtype=np.int32)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        flat_rows = rr.reshape(-1)
        flat_cols = cc.reshape(-1)
        if face in {"floor", "ceiling"}:
            valid = valid_indices_for_face(source_dir, face, size)
            valid_mask = np.zeros(h * w, dtype=bool)
            valid_mask[valid] = True
            keep = valid_mask[(flat_rows * w + flat_cols).astype(np.int64)]
            flat_rows = flat_rows[keep]
            flat_cols = flat_cols[keep]
        for start in range(0, len(flat_rows), 220000):
            rows_c = flat_rows[start : start + 220000]
            cols_c = flat_cols[start : start + 220000]
            pts_da3 = face_points_for_indices(face, rows_c, cols_c, size, manifest, metas)
            pts_col = sim.da3_to_colmap(pts_da3)
            u, v, z = project_points(pts_col, pose)
            ok = (z > 0.05) & (u >= 0) & (u < pose.width) & (v >= 0) & (v < pose.height)
            if not np.any(ok):
                continue
            px = np.clip(np.round(u[ok]).astype(np.int32), 0, pose.width - 1)
            py = np.clip(np.round(v[ok]).astype(np.int32), 0, pose.height - 1)
            pix = py * pose.width + px
            np.minimum.at(zbuf.reshape(-1), pix, z[ok])
    large = np.float32(1e9)
    finite = np.isfinite(zbuf)
    filled = zbuf.copy()
    filled[~finite] = large
    # A small min-filter fills sparse raster gaps while preserving nearest room surface.
    k = max(3, int(stride * 2 + 1))
    zmin = cv2.erode(filled, np.ones((k, k), np.uint8), iterations=1)
    zmin[zmin >= large * 0.5] = np.inf
    return zmin


def build_face_id_and_zbuffer(
    pose: ImagePose,
    faces: list[str],
    source_dir: Path,
    sim: Similarity,
    manifest: dict,
    metas: dict,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    zbuf = np.full((pose.height, pose.width), np.inf, dtype=np.float32)
    face_id = np.full((pose.height, pose.width), 255, dtype=np.uint8)
    flat_z = zbuf.reshape(-1)
    flat_f = face_id.reshape(-1)
    for face_idx, face_name in enumerate(faces):
        size = face_texture_size(source_dir, metas[face_name])
        w, h = size
        valid = valid_indices_for_face(source_dir, face_name, size)
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
            pts_da3 = face_points_for_indices(face_name, rows, cols, size, manifest, metas)
            pts_col = sim.da3_to_colmap(pts_da3)
            u, v, z = project_points(pts_col, pose)
            ok = (z > 0.05) & (u >= 0.0) & (u < pose.width) & (v >= 0.0) & (v < pose.height)
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
            mask = face_id == idx
            if not np.any(mask):
                continue
            grown = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            fill = unknown & grown
            face_out[fill] = idx
            unknown &= ~fill
    zmin[zmin >= large * 0.5] = np.inf
    return face_out, zmin


def save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        img = np.zeros(arr.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, [1, 99])
        img = np.clip((arr - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def complete_preview(raw: np.ndarray, observed: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = raw.copy()
    valid = valid_mask > 0
    if np.count_nonzero(observed & valid) == 0:
        out[valid] = np.array([180, 180, 180], dtype=np.uint8)
        return out
    med = np.median(raw[observed & valid], axis=0).astype(np.uint8)
    init = raw.copy()
    init[valid & ~observed] = med
    hole = (valid & ~observed).astype(np.uint8) * 255
    if np.count_nonzero(hole):
        out = cv2.inpaint(init, hole, 3.0, cv2.INPAINT_TELEA)
    else:
        out = init
    out[~valid] = 0
    return out


def update_nearest_visible_winners(
    nearest_rgb: np.ndarray,
    nearest_view_id: np.ndarray,
    nearest_distance: np.ndarray,
    nearest_weight: np.ndarray,
    flat_indices: np.ndarray,
    colors: np.ndarray,
    view_id: int,
    distances: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Update a diagnostic winner sidecar without touching weighted accumulators."""

    flat_indices_raw = np.asarray(flat_indices)
    if not np.issubdtype(flat_indices_raw.dtype, np.integer):
        raise ValueError("nearest-visible winner update requires integer atlas texel indices")
    flat_indices = flat_indices_raw.astype(np.int64, copy=False)
    colors = np.asarray(colors, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    texel_count = int(nearest_view_id.size)
    if not (
        flat_indices.ndim == 1
        and colors.shape == (flat_indices.size, 3)
        and distances.shape == flat_indices.shape
        and weights.shape == flat_indices.shape
        and nearest_rgb.shape == (texel_count, 3)
        and nearest_view_id.shape == (texel_count,)
        and np.issubdtype(nearest_view_id.dtype, np.integer)
        and nearest_distance.shape == (texel_count,)
        and nearest_weight.shape == (texel_count,)
    ):
        raise ValueError("nearest-visible winner update received incompatible shapes")
    if not isinstance(view_id, (int, np.integer)) or int(view_id) < 0:
        raise ValueError("nearest-visible winner update requires a non-negative integer view ID")
    if np.any(flat_indices < 0) or np.any(flat_indices >= texel_count):
        raise ValueError("nearest-visible winner update received an out-of-range atlas texel")
    if np.unique(flat_indices).size != flat_indices.size:
        raise ValueError("nearest-visible winner update requires unique atlas texels per call")
    if (
        not np.all(np.isfinite(colors))
        or np.any(colors < -NEAREST_RGB_RANGE_ATOL)
        or np.any(colors > 1.0 + NEAREST_RGB_RANGE_ATOL)
    ):
        raise ValueError("nearest-visible winner RGB must be finite and normalized to [0,1]")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("nearest-visible camera distances must be finite and non-negative")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 1.0e-8):
        raise ValueError("nearest-visible projection weights must be finite and positive")
    current_distance = nearest_distance[flat_indices]
    current_weight = nearest_weight[flat_indices]
    better = (distances < current_distance - NEAREST_DISTANCE_TIE_ATOL) | (
        (np.abs(distances - current_distance) <= NEAREST_DISTANCE_TIE_ATOL)
        & (weights > current_weight)
    )
    if not np.any(better):
        return
    chosen = flat_indices[better]
    nearest_rgb[chosen] = colors[better]
    nearest_view_id[chosen] = int(view_id)
    nearest_distance[chosen] = distances[better]
    nearest_weight[chosen] = weights[better]


def save_nearest_visible_face_evidence(
    out_dir: Path,
    face: str,
    nearest_rgb: np.ndarray,
    nearest_view_id: np.ndarray,
    nearest_distance: np.ndarray,
    nearest_weight: np.ndarray,
    final_keep_map: np.ndarray,
    selected_poses: list[ImagePose],
) -> dict:
    """Write per-face winner evidence masked to the authoritative final_keep domain."""

    final_keep_map = np.asarray(final_keep_map)
    if final_keep_map.ndim != 2:
        raise ValueError(f"nearest-visible final_keep must be 2-D for {face}")
    h, w = final_keep_map.shape
    texel_count = h * w
    nearest_rgb = np.asarray(nearest_rgb, dtype=np.float64)
    nearest_view_id = np.asarray(nearest_view_id)
    nearest_distance = np.asarray(nearest_distance, dtype=np.float64)
    nearest_weight = np.asarray(nearest_weight, dtype=np.float64)
    if not (
        nearest_rgb.shape == (texel_count, 3)
        and nearest_view_id.shape == (texel_count,)
        and np.issubdtype(nearest_view_id.dtype, np.integer)
        and nearest_distance.shape == (texel_count,)
        and nearest_weight.shape == (texel_count,)
    ):
        raise ValueError(f"nearest-visible face evidence has incompatible shapes for {face}")
    keep = final_keep_map.astype(bool, copy=False).reshape(-1)
    if not np.any(keep):
        raise RuntimeError(f"nearest-visible sidecar has no final_keep evidence for {face}")
    ids = np.full(h * w, -1, dtype=np.int32)
    distances = np.full(h * w, np.inf, dtype=np.float32)
    weights = np.zeros(h * w, dtype=np.float32)
    rgb_float = np.zeros((h * w, 3), dtype=np.float64)
    ids[keep] = nearest_view_id[keep].astype(np.int32)
    distances[keep] = nearest_distance[keep].astype(np.float32)
    weights[keep] = nearest_weight[keep].astype(np.float32)
    rgb_float[keep] = nearest_rgb[keep]
    if np.any(ids[keep] < 0):
        raise RuntimeError(f"nearest-visible sidecar has no winning view inside final_keep: {face}")
    if np.any(~np.isfinite(distances[keep])) or np.any(distances[keep] < 0.0):
        raise RuntimeError(f"nearest-visible sidecar has invalid distance inside final_keep: {face}")
    if np.any(~np.isfinite(weights[keep])) or np.any(weights[keep] <= 1.0e-8):
        raise RuntimeError(f"nearest-visible sidecar has invalid weight inside final_keep: {face}")
    if (
        np.any(~np.isfinite(rgb_float[keep]))
        or np.any(rgb_float[keep] < -NEAREST_RGB_RANGE_ATOL)
        or np.any(rgb_float[keep] > 1.0 + NEAREST_RGB_RANGE_ATOL)
    ):
        raise RuntimeError(f"nearest-visible sidecar has invalid RGB inside final_keep: {face}")

    pose_names: dict[int, str] = {}
    for pose in selected_poses:
        pose_id = int(pose.image_id)
        pose_name = str(pose.name)
        if pose_id < 0 or not pose_name:
            raise RuntimeError(f"nearest-visible sidecar has an invalid selected pose for {face}")
        if pose_id in pose_names:
            raise RuntimeError(f"nearest-visible sidecar has duplicate selected view ID {pose_id} for {face}")
        pose_names[pose_id] = pose_name
    chosen_ids = sorted(int(item) for item in np.unique(ids[keep]))
    missing_ids = sorted(set(chosen_ids) - set(pose_names))
    if missing_ids:
        raise RuntimeError(f"nearest-visible sidecar has unmapped view IDs for {face}: {missing_ids}")

    evidence_dir = out_dir / "nearest_visible_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = evidence_dir / f"{face}_winning_rgb.png"
    view_id_path = evidence_dir / f"{face}_view_id.npy"
    distance_path = evidence_dir / f"{face}_camera_distance.npy"
    weight_path = evidence_dir / f"{face}_projection_weight.npy"
    rgb_u8 = np.clip(rgb_float.reshape(h, w, 3) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_u8).save(rgb_path)
    np.save(view_id_path, ids.reshape(h, w))
    np.save(distance_path, distances.reshape(h, w))
    np.save(weight_path, weights.reshape(h, w))

    def record(path: Path) -> dict:
        data = path.read_bytes()
        return {
            "path": path.relative_to(out_dir).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    rgb_record = record(rgb_path)
    id_record = record(view_id_path)
    distance_record = record(distance_path)
    weight_record = record(weight_path)
    return {
        "rgb_path": rgb_record["path"],
        "rgb_bytes": rgb_record["bytes"],
        "rgb_sha256": rgb_record["sha256"],
        "view_id_path": id_record["path"],
        "view_id_bytes": id_record["bytes"],
        "view_id_sha256": id_record["sha256"],
        "distance_path": distance_record["path"],
        "distance_bytes": distance_record["bytes"],
        "distance_sha256": distance_record["sha256"],
        "winning_weight_path": weight_record["path"],
        "winning_weight_bytes": weight_record["bytes"],
        "winning_weight_sha256": weight_record["sha256"],
        "view_id_to_name": {str(view_id): pose_names[view_id] for view_id in chosen_ids},
        "winner_texels": int(np.count_nonzero(keep)),
    }


def projection_contract(manifest: dict) -> dict:
    """Canonical producer/gate contract shared with the baseline consumer."""

    face_contract = []
    for record in manifest.get("faces", []):
        face_contract.append(
            {
                "face": record.get("face"),
                "texture_size": record.get("texture_size"),
                "selected_views": record.get("selected_views"),
            }
        )
    return {
        "dataset_dir": manifest.get("dataset_dir"),
        "polygon_source_dir": manifest.get("polygon_source_dir"),
        "colmap_model_dir": manifest.get("colmap_model_dir"),
        "pose_source": manifest.get("pose_source"),
        "da3_dir": manifest.get("da3_dir"),
        "registered_images": manifest.get("registered_images"),
        "alignment_colmap_to_da3": manifest.get("alignment_colmap_to_da3"),
        "hf_alignment_scene_glb": manifest.get("hf_alignment_scene_glb"),
        "alignment_da3_numeric_to_room_da3": manifest.get(
            "alignment_da3_numeric_to_room_da3"
        ),
        "da3_depth_calibration": manifest.get("da3_depth_calibration"),
        "weight_terms": manifest.get("weight_terms"),
        "replicated_v60_parameters": manifest.get("replicated_v60_parameters"),
        "faces": face_contract,
    }


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    """Commit an authoritative manifest as one filesystem replacement."""

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_nearest_visible_sidecar(out_dir: Path, photo_manifest: dict) -> Path:
    photo_manifest_path = out_dir / "photo_source_manifest.json"
    if not photo_manifest_path.is_file():
        raise FileNotFoundError(photo_manifest_path)
    disk_manifest = json.loads(photo_manifest_path.read_text(encoding="utf-8"))
    if disk_manifest != photo_manifest:
        raise RuntimeError("nearest-visible sidecar manifest object differs from the on-disk manifest")
    faces = {}
    for record in photo_manifest.get("faces", []):
        face = str(record.get("face", ""))
        evidence = record.get("nearest_visible_evidence")
        if not face or not isinstance(evidence, dict):
            raise RuntimeError(f"nearest-visible face evidence is missing from manifest: {face!r}")
        if face in faces:
            raise RuntimeError(f"nearest-visible manifest contains duplicate face {face!r}")
        faces[face] = evidence
    if not faces:
        raise RuntimeError("nearest-visible manifest contains no face evidence")
    sidecar = {
        "schema_version": NEAREST_EVIDENCE_SCHEMA,
        "selection_policy": NEAREST_SELECTION_POLICY,
        "projection_manifest_path": str(photo_manifest_path.resolve()),
        "projection_manifest_sha256": hashlib.sha256(
            photo_manifest_path.read_bytes()
        ).hexdigest(),
        "projection_contract_sha256": sha256_json(projection_contract(photo_manifest)),
        "distance_definition": (
            "Euclidean camera-center-to-surface-texel distance in the polygon room coordinate frame"
        ),
        "distance_tie_absolute_tolerance": NEAREST_DISTANCE_TIE_ATOL,
        "rgb_sampling_and_quantization": (
            "same bilinear source-image sample as weighted fusion; clip(sample*255,0,255) then uint8 truncation"
        ),
        "winner_domain": (
            "same accepted strict projection samples with original projection weight > 1e-8, "
            "then masked to final_keep"
        ),
        "faces": faces,
    }
    path = out_dir / "nearest_visible_evidence.json"
    write_text_atomic(path, json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
    return path


def process_face(
    face: str,
    source_dir: Path,
    out_dir: Path,
    poses: list[ImagePose],
    selected_poses: list[ImagePose],
    sim: Similarity,
    manifest: dict,
    metas: dict,
    all_faces: list[str],
    args: argparse.Namespace,
    zbuffer_cache: dict[int, np.ndarray] | None = None,
    face_id_cache: dict[int, np.ndarray] | None = None,
    view_reject_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    da3_views: dict[int, Da3View] | None = None,
    da3_numeric_to_room: Similarity | None = None,
    da3_raw_to_room_matrix: np.ndarray | None = None,
    da3_depth_calibration_cache: dict[int, Da3DepthCalibration | None] | None = None,
) -> dict:
    size = face_texture_size(source_dir, metas[face])
    w, h = size
    valid_flat = valid_indices_for_face(source_dir, face, size)
    valid_mask = np.zeros(h * w, dtype=np.uint8)
    valid_mask[valid_flat] = 255
    valid_mask = valid_mask.reshape(h, w)

    sum_rgb = np.zeros((h * w, 3), dtype=np.float64)
    sum_rgb2 = np.zeros((h * w, 3), dtype=np.float64)
    weight_sum = np.zeros(h * w, dtype=np.float64)
    if args.emit_nearest_visible_evidence:
        nearest_rgb = np.zeros((h * w, 3), dtype=np.float64)
        nearest_view_id = np.full(h * w, -1, dtype=np.int32)
        nearest_distance = np.full(h * w, np.inf, dtype=np.float64)
        nearest_weight = np.zeros(h * w, dtype=np.float64)
    else:
        nearest_rgb = nearest_view_id = nearest_distance = nearest_weight = None
    accum_depth_residual = np.zeros(h * w, dtype=np.float64)
    accum_surface_distance = np.zeros(h * w, dtype=np.float64)
    accum_object_risk = np.zeros(h * w, dtype=np.float64)
    accum_mask_boundary_trust = np.zeros(h * w, dtype=np.float64)
    accum_footprint_area = np.zeros(h * w, dtype=np.float64)
    valid_count = np.zeros(h * w, dtype=np.float32)
    candidate_count = np.zeros(h * w, dtype=np.float32)
    n = face_normal(face, manifest, metas)

    zbuffers = zbuffer_cache if zbuffer_cache is not None else {}
    face_ids = face_id_cache if face_id_cache is not None else {}
    view_reject_maps = view_reject_cache if view_reject_cache is not None else {}
    da3_views = da3_views or {}
    da3_depth_calibration_cache = da3_depth_calibration_cache if da3_depth_calibration_cache is not None else {}
    face_idx = all_faces.index(face)
    for pose in selected_poses:
        if pose.image_id not in zbuffers or pose.image_id not in face_ids:
            face_id_map, zbuf = build_face_id_and_zbuffer(
                pose,
                all_faces,
                source_dir,
                sim,
                manifest,
                metas,
                args.zbuffer_stride,
            )
            face_ids[pose.image_id] = face_id_map
            zbuffers[pose.image_id] = zbuf
        if pose.image_id not in view_reject_maps:
            view_reject_maps[pose.image_id] = load_view_reject_maps(pose, args)
        if pose.image_id not in da3_depth_calibration_cache and pose.image_id in da3_views:
            da3_depth_calibration_cache[pose.image_id] = calibrate_da3_depth_to_colmap_zbuffer(
                da3_views[pose.image_id],
                pose,
                zbuffers[pose.image_id],
            )
    if da3_views:
        calibs = [
            da3_depth_calibration_cache.get(pose.image_id)
            for pose in selected_poses
            if pose.image_id in da3_views
        ]
        usable = [c for c in calibs if c is not None]
        if usable:
            med = float(np.median([c.median_abs_error for c in usable]))
            p80 = float(np.median([c.p80_abs_error for c in usable]))
            modes = {mode: sum(1 for c in usable if c.mode == mode) for mode in {"linear", "inverse"}}
            print(
                f"[da3-depth-calib] {face}: usable={len(usable)}/{len(calibs)} "
                f"median_abs={med:.5f} median_p80={p80:.5f} modes={modes}",
                flush=True,
            )
        else:
            print(f"[da3-depth-calib] {face}: usable=0/{len(calibs)}", flush=True)
    for pose in selected_poses:
        image = np.asarray(Image.open(pose.image_path).convert("RGB"), dtype=np.float32) / 255.0
        zbuf = zbuffers[pose.image_id]
        face_id_map = face_ids[pose.image_id]
        object_mask, object_risk_map_view, boundary_trust_map_view = view_reject_maps[pose.image_id]
        for start in range(0, len(valid_flat), args.chunk_size):
            flat = valid_flat[start : start + args.chunk_size]
            rows = (flat // w).astype(np.int32)
            cols = (flat % w).astype(np.int32)
            pts_da3 = face_points_for_indices(face, rows, cols, size, manifest, metas)
            pts_col = sim.da3_to_colmap(pts_da3)
            u, v, z = project_points(pts_col, pose)
            border = np.minimum.reduce([u, v, pose.width - 1 - u, pose.height - 1 - v])
            view = pose.center_da3[None, :] - pts_da3
            dist = np.linalg.norm(view, axis=1)
            view_unit = view / np.maximum(dist[:, None], 1e-8)
            cos = np.abs(view_unit @ n)
            in_frame = (
                (z > 1e-6)
                & (u >= 0.0)
                & (v >= 0.0)
                & (u <= pose.width - 1.0)
                & (v <= pose.height - 1.0)
                & (cos >= args.min_view_cos)
            )
            candidate_count[flat] += in_frame.astype(np.float32)
            if not np.any(in_frame):
                continue

            idx = np.flatnonzero(in_frame)
            px = np.clip(np.round(u[idx]).astype(np.int32), 0, pose.width - 1)
            py = np.clip(np.round(v[idx]).astype(np.int32), 0, pose.height - 1)
            sampled_conf = np.ones(idx.shape, dtype=np.float32)
            da3_view = da3_views.get(pose.image_id)
            da3_depth_calib = da3_depth_calibration_cache.get(pose.image_id)
            if da3_view is not None and da3_depth_calib is not None:
                du, dv = da3_uv_from_colmap_pixels(u[idx], v[idx], pose, da3_view)
                sampled_depth_raw = sample_float_map(da3_view.depth, du, dv, border_value=np.nan)
                sampled_depth = apply_depth_calibration(sampled_depth_raw, da3_depth_calib)
                if da3_view.conf is not None:
                    sampled_conf = sample_float_map(da3_view.conf, du, dv, border_value=0.0)
                has_depth = np.isfinite(sampled_depth) & (sampled_depth > 1e-6)
                camera_depth = sampled_depth
                projected_depth = z[idx]
                if args.surface_distance_tol > 0.0:
                    sampled_world = backproject_camera_z_to_world(u[idx], v[idx], camera_depth, pose)
                    if sampled_world is not None:
                        sampled_world = sim.colmap_to_da3(sampled_world)
                    else:
                        sampled_world = backproject_da3_raw_depth_to_room(
                            du,
                            dv,
                            sampled_depth_raw,
                            da3_view,
                            raw_to_room_matrix=da3_raw_to_room_matrix,
                            raw_to_room_similarity=da3_numeric_to_room,
                        )
                    if sampled_world is None:
                        surface_distance = np.full(idx.shape, np.inf, dtype=np.float32)
                    else:
                        surface_distance = face_surface_distance(sampled_world, face, manifest, metas)
                else:
                    surface_distance = np.zeros(idx.shape, dtype=np.float32)
            elif da3_view is not None:
                pts_for_da3_numeric = pts_da3[idx]
                if da3_numeric_to_room is not None:
                    pts_for_da3_numeric = da3_numeric_to_room.da3_to_colmap(pts_for_da3_numeric)
                du, dv, dz = project_da3_points(pts_for_da3_numeric, da3_view)
                sampled_depth = sample_float_map(da3_view.depth, du, dv, border_value=np.nan)
                if da3_view.conf is not None:
                    sampled_conf = sample_float_map(da3_view.conf, du, dv, border_value=0.0)
                has_depth = (dz > 1e-6) & np.isfinite(sampled_depth)
                camera_depth = sampled_depth
                projected_depth = dz
                if args.surface_distance_tol > 0.0:
                    sampled_world = backproject_da3_raw_depth_to_room(
                        du,
                        dv,
                        sampled_depth,
                        da3_view,
                        raw_to_room_similarity=da3_numeric_to_room,
                    )
                    if sampled_world is None:
                        surface_distance = np.full(idx.shape, np.inf, dtype=np.float32)
                    else:
                        surface_distance = face_surface_distance(sampled_world, face, manifest, metas)
                else:
                    surface_distance = np.zeros(idx.shape, dtype=np.float32)
            else:
                z_shell = zbuf[py, px]
                has_depth = np.isfinite(z_shell)
                camera_depth = z_shell
                projected_depth = z[idx]
                surface_distance = np.zeros(idx.shape, dtype=np.float32)
            depth_tol = args.depth_abs_tol + args.depth_rel_tol * np.maximum(camera_depth, 0.0)
            depth_diff = np.full(idx.shape, np.inf, dtype=np.float32)
            depth_diff[has_depth] = np.abs(projected_depth[has_depth] - camera_depth[has_depth])
            depth_residual = np.clip(depth_diff / np.maximum(depth_tol, 1e-6), 0.0, 1.0)
            if args.surface_distance_tol > 0.0 and args.surface_distance_hard_gate:
                surface_ok = np.isfinite(surface_distance) & (surface_distance <= args.surface_distance_tol)
            else:
                surface_ok = np.ones(idx.shape, dtype=bool)
            if args.surface_normal_min_cos > 0.0 and da3_view is not None:
                sampled_normal_cos, sampled_normal_valid = sampled_da3_normal_cos(
                    da3_view,
                    du,
                    dv,
                    n,
                    raw_to_room_matrix=da3_raw_to_room_matrix,
                    raw_to_room_similarity=da3_numeric_to_room,
                )
                normal_ok = sampled_normal_valid & (
                    sampled_normal_cos >= float(args.surface_normal_min_cos)
                )
            elif args.surface_normal_min_cos > 0.0:
                normal_ok = np.zeros(idx.shape, dtype=bool)
            else:
                normal_ok = np.ones(idx.shape, dtype=bool)

            sampled_object = sample_nearest_map(object_mask, u[idx], v[idx], border_value=0)
            sampled_face = sample_nearest_map(face_id_map, u[idx], v[idx], border_value=255)
            sampled_object_risk = sample_float_map(object_risk_map_view, u[idx], v[idx], border_value=0.0)
            sampled_mask_boundary_trust = sample_float_map(boundary_trust_map_view, u[idx], v[idx], border_value=1.0)

            valid = (
                has_depth
                & np.isfinite(depth_diff)
                & (sampled_conf >= args.min_conf)
                & (depth_diff <= depth_tol)
                & (sampled_face == face_idx)
                & (sampled_object == 0)
                & surface_ok
                & normal_ok
            )
            if args.object_risk_hard_thresh <= 1.0:
                valid &= sampled_object_risk <= float(args.object_risk_hard_thresh)
            if args.min_mask_boundary_trust > 0.0:
                valid &= sampled_mask_boundary_trust >= float(args.min_mask_boundary_trust)
            if not np.any(valid):
                continue
            idx = idx[valid]
            depth_diff = depth_diff[valid]
            depth_tol = depth_tol[valid]
            depth_residual = depth_residual[valid]
            surface_distance = surface_distance[valid]
            sampled_conf = sampled_conf[valid]
            sampled_object_risk = sampled_object_risk[valid]
            sampled_mask_boundary_trust = sampled_mask_boundary_trust[valid]

            rows_sel = rows[idx]
            cols_sel = cols[idx]
            cols_x = np.where(cols_sel < w - 1, cols_sel + 1, np.maximum(cols_sel - 1, 0)).astype(np.int32)
            rows_y = np.where(rows_sel < h - 1, rows_sel + 1, np.maximum(rows_sel - 1, 0)).astype(np.int32)
            dx_sign = np.where(cols_sel < w - 1, 1.0, -1.0).astype(np.float32)
            dy_sign = np.where(rows_sel < h - 1, 1.0, -1.0).astype(np.float32)
            pts_x = face_points_for_indices(face, rows_sel, cols_x, size, manifest, metas)
            pts_y = face_points_for_indices(face, rows_y, cols_sel, size, manifest, metas)
            u_x, v_x, _ = project_points(sim.da3_to_colmap(pts_x), pose)
            u_y, v_y, _ = project_points(sim.da3_to_colmap(pts_y), pose)
            du_dx = (u_x - u[idx]) / dx_sign
            dv_dx = (v_x - v[idx]) / dx_sign
            du_dy = (u_y - u[idx]) / dy_sign
            dv_dy = (v_y - v[idx]) / dy_sign
            footprint_area = np.abs(du_dx * dv_dy - du_dy * dv_dx).astype(np.float32)
            footprint_area = np.where(np.isfinite(footprint_area), footprint_area, 0.0)
            if args.footprint_min_area > 0.0:
                footprint_weight = np.clip(
                    footprint_area / max(float(args.footprint_min_area), 1e-6),
                    0.0,
                    1.0,
                )
                if args.footprint_power > 0.0 and args.footprint_power != 1.0:
                    footprint_weight = np.power(footprint_weight, args.footprint_power)
            else:
                footprint_weight = np.ones(idx.shape, dtype=np.float32)
                footprint_area = np.ones(idx.shape, dtype=np.float32)

            angle_w = np.clip(cos[idx], 0.0, 1.0) ** 2
            if args.distance_weight_power > 0.0:
                distance_weight = 1.0 / (
                    1.0
                    + dist[idx]
                    / max(args.distance_weight_scale, 1e-6)
                ) ** args.distance_weight_power
            else:
                distance_weight = np.ones(idx.shape, dtype=np.float32)
            depth_weight = 1.0 / (1.0 + depth_diff / np.maximum(depth_tol, 1e-6))
            if args.surface_distance_tol > 0.0:
                surface_weight = 1.0 / (
                    1.0 + surface_distance / max(float(args.surface_distance_tol), 1e-6)
                )
                if args.surface_distance_power > 0.0 and args.surface_distance_power != 1.0:
                    surface_weight = np.power(surface_weight, float(args.surface_distance_power))
            else:
                surface_weight = np.ones(idx.shape, dtype=np.float32)
            boundary_weight = np.clip(sampled_mask_boundary_trust, 0.0, 1.0)
            weights = (
                sampled_conf
                * angle_w
                * distance_weight
                * depth_weight
                * surface_weight
                * boundary_weight
                * footprint_weight
            ).astype(np.float64)
            weights[~np.isfinite(weights)] = 0.0
            nonzero = weights > 1e-8
            if args.min_sample_weight > 0.0:
                nonzero &= weights >= float(args.min_sample_weight)
            if not np.any(nonzero):
                continue
            idx = idx[nonzero]
            weights = weights[nonzero]
            colors = bilinear_sample(image, u[idx], v[idx]).astype(np.float64)
            flat_sel = flat[idx]
            if args.emit_nearest_visible_evidence:
                assert nearest_rgb is not None
                assert nearest_view_id is not None
                assert nearest_distance is not None
                assert nearest_weight is not None
                update_nearest_visible_winners(
                    nearest_rgb,
                    nearest_view_id,
                    nearest_distance,
                    nearest_weight,
                    flat_sel,
                    colors,
                    pose.image_id,
                    dist[idx],
                    weights,
                )
            sum_rgb[flat_sel] += weights[:, None] * colors
            sum_rgb2[flat_sel] += weights[:, None] * (colors * colors)
            weight_sum[flat_sel] += weights
            valid_count[flat_sel] += 1.0
            accum_depth_residual[flat_sel] += weights * depth_residual[nonzero]
            accum_surface_distance[flat_sel] += weights * surface_distance[nonzero]
            accum_object_risk[flat_sel] += weights * sampled_object_risk[nonzero]
            accum_mask_boundary_trust[flat_sel] += weights * boundary_weight[nonzero]
            accum_footprint_area[flat_sel] += weights * footprint_area[nonzero]

    observed = weight_sum > 1e-8
    reliable = observed & (valid_count >= args.min_valid_views)
    raw = np.zeros((h * w, 3), dtype=np.float32)
    if np.any(observed):
        raw[observed] = (sum_rgb[observed] / weight_sum[observed, None]).astype(np.float32)
    else:
        raw[:, :] = np.array([0.55, 0.55, 0.55], dtype=np.float32)
    raw = np.clip(raw.reshape(h, w, 3), 0.0, 1.0)
    raw_u8 = np.clip(raw * 255.0, 0, 255).astype(np.uint8)
    observed_map = observed.reshape(h, w)
    reliable_map = reliable.reshape(h, w)
    hole_mask = (~reliable_map).astype(np.uint8) * 255
    if args.hole_dilate_px > 0:
        kernel_size = 2 * int(args.hole_dilate_px) + 1
        hole_mask = cv2.dilate(hole_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    if np.any(hole_mask > 0) and np.any(hole_mask == 0):
        completed = cv2.cvtColor(
            cv2.inpaint(
                cv2.cvtColor(raw_u8, cv2.COLOR_RGB2BGR),
                hole_mask,
                args.inpaint_radius,
                cv2.INPAINT_TELEA,
            ),
            cv2.COLOR_BGR2RGB,
        )
    else:
        completed = raw_u8
    completed[valid_mask == 0] = 0

    count_map = valid_count.reshape(h, w).astype(np.float32)
    candidate_map = candidate_count.reshape(h, w).astype(np.float32)
    weight_map = weight_sum.reshape(h, w).astype(np.float32)
    positive_weight = weight_map[weight_map > 1e-8]
    if positive_weight.size:
        weight_scale = float(np.percentile(positive_weight, 90.0))
        if weight_scale <= 1e-8:
            weight_scale = float(np.max(positive_weight))
    else:
        weight_scale = 1.0
    count_reliability = np.clip(count_map / max(1, len(selected_poses)), 0.0, 1.0)
    weight_reliability = np.clip(weight_map / max(weight_scale, 1e-8), 0.0, 1.0)
    reliability = np.sqrt(count_reliability * weight_reliability).astype(np.float32)
    reliability[~observed_map] = 0.0

    mean_rgb = np.zeros_like(sum_rgb, dtype=np.float64)
    mean_rgb2 = np.zeros_like(sum_rgb2, dtype=np.float64)
    mean_depth_residual = np.ones(h * w, dtype=np.float64)
    mean_surface_distance = np.ones(h * w, dtype=np.float64)
    mean_object_risk = np.ones(h * w, dtype=np.float64)
    mean_mask_boundary_trust = np.zeros(h * w, dtype=np.float64)
    mean_footprint_area = np.zeros(h * w, dtype=np.float64)
    mean_rgb[observed] = sum_rgb[observed] / weight_sum[observed, None]
    mean_rgb2[observed] = sum_rgb2[observed] / weight_sum[observed, None]
    mean_depth_residual[observed] = accum_depth_residual[observed] / weight_sum[observed]
    mean_surface_distance[observed] = accum_surface_distance[observed] / weight_sum[observed]
    mean_object_risk[observed] = accum_object_risk[observed] / weight_sum[observed]
    mean_mask_boundary_trust[observed] = accum_mask_boundary_trust[observed] / weight_sum[observed]
    mean_footprint_area[observed] = accum_footprint_area[observed] / weight_sum[observed]
    color_var = np.maximum(mean_rgb2 - mean_rgb * mean_rgb, 0.0)
    color_std = np.sqrt(np.mean(color_var, axis=1)).reshape(h, w).astype(np.float32)
    depth_map = mean_depth_residual.reshape(h, w).astype(np.float32)
    surface_distance_map = mean_surface_distance.reshape(h, w).astype(np.float32)
    object_risk_map = mean_object_risk.reshape(h, w).astype(np.float32)
    mask_boundary_trust_map = mean_mask_boundary_trust.reshape(h, w).astype(np.float32)
    footprint_map = mean_footprint_area.reshape(h, w).astype(np.float32)
    valid_ratio = np.zeros((h, w), dtype=np.float32)
    np.divide(count_map, np.maximum(candidate_map, 1.0), out=valid_ratio, where=candidate_map > 0)

    object_penalty = np.clip((object_risk_map - 0.55) / 0.45, 0.0, 1.0)
    mask_boundary_penalty = np.clip((0.55 - mask_boundary_trust_map) / 0.55, 0.0, 1.0) * 0.65
    effective_footprint_min_area = float(args.footprint_min_area)
    adaptive_footprint_applied = False
    adaptive_horizontal_footprint_applied = False
    wall_length = None
    median_wall_length = None
    observed_footprint_median = None
    horizontal_reliable_footprint_median = None
    if args.adaptive_horizontal_footprint and face in {"floor", "ceiling"}:
        positive_reliable_footprint = footprint_map[
            reliable_map & np.isfinite(footprint_map) & (footprint_map > 0.0)
        ]
        if positive_reliable_footprint.size:
            horizontal_reliable_footprint_median = float(
                np.median(positive_reliable_footprint)
            )
            adaptive_reference = max(
                float(args.horizontal_footprint_min_area),
                horizontal_reliable_footprint_median
                * float(args.horizontal_footprint_median_multiplier),
            )
            effective_footprint_min_area = min(
                effective_footprint_min_area,
                adaptive_reference,
            )
            adaptive_horizontal_footprint_applied = (
                effective_footprint_min_area < float(args.footprint_min_area)
            )
    if args.adaptive_short_face_footprint and face.startswith("wall_"):
        wall_lengths = [
            float(item.get("length", 0.0))
            for name, item in metas.items()
            if name.startswith("wall_") and float(item.get("length", 0.0)) > 0.0
        ]
        wall_length = float(metas[face].get("length", 0.0))
        if wall_lengths:
            median_wall_length = float(np.median(np.asarray(wall_lengths, dtype=np.float64)))
        positive_footprint = footprint_map[observed_map & np.isfinite(footprint_map) & (footprint_map > 0.0)]
        if positive_footprint.size:
            observed_footprint_median = float(np.median(positive_footprint))
        is_short = bool(
            median_wall_length is not None
            and wall_length > 0.0
            and wall_length <= float(args.short_face_length_median_frac) * median_wall_length
        )
        if is_short and observed_footprint_median is not None:
            adaptive_reference = max(
                float(args.short_face_footprint_min_area),
                observed_footprint_median * float(args.short_face_footprint_median_multiplier),
            )
            effective_footprint_min_area = min(effective_footprint_min_area, adaptive_reference)
            adaptive_footprint_applied = effective_footprint_min_area < float(args.footprint_min_area)
    if effective_footprint_min_area > 0.0:
        footprint_penalty = (
            np.clip(
                (effective_footprint_min_area - footprint_map)
                / max(effective_footprint_min_area, 1e-6),
                0.0,
                1.0,
            )
            * 0.55
        )
    else:
        footprint_penalty = np.zeros((h, w), dtype=np.float32)
    depth_penalty = np.clip((depth_map - 0.60) / 0.40, 0.0, 1.0) * 0.80
    surface_distance_clean_tol = (
        float(args.surface_distance_clean_tol)
        if args.surface_distance_clean_tol > 0.0
        else float(args.surface_distance_tol)
    )
    if surface_distance_clean_tol > 0.0:
        surface_penalty = (
            np.clip(
                surface_distance_map / max(surface_distance_clean_tol, 1e-6),
                0.0,
                1.0,
            )
            * 0.60
        ).astype(np.float32)
    else:
        surface_penalty = np.zeros((h, w), dtype=np.float32)
    color_penalty = (
        np.clip(
            (color_std - args.color_std_clean_tol) / max(args.color_std_clean_tol, 1e-6),
            0.0,
            1.0,
        )
        * 0.70
    )
    low_valid_ratio_penalty = np.clip((0.25 - valid_ratio) / 0.25, 0.0, 1.0) * args.valid_ratio_penalty
    contamination_score = np.maximum.reduce(
        [
            object_penalty,
            mask_boundary_penalty,
            footprint_penalty,
            depth_penalty,
            surface_penalty,
            color_penalty,
            low_valid_ratio_penalty,
        ]
    ).astype(np.float32)
    clean_score = np.clip(1.0 - contamination_score, 0.0, 1.0).astype(np.float32)
    clean_score[~observed_map] = 0.0
    contamination_score[~observed_map] = 1.0

    final_keep_map = reliable_map.copy()
    if args.min_output_reliability > 0.0:
        final_keep_map &= reliability >= float(args.min_output_reliability)
    if args.min_output_clean_score > 0.0:
        final_keep_map &= clean_score >= float(args.min_output_clean_score)
    if args.max_output_contamination_score <= 1.0:
        final_keep_map &= contamination_score <= float(args.max_output_contamination_score)
    if args.max_output_object_risk <= 1.0:
        final_keep_map &= object_risk_map <= float(args.max_output_object_risk)
    final_keep_map &= valid_mask > 0
    output_u8 = completed.copy()
    if args.strict_empty_low_quality:
        output_u8[~final_keep_map] = 0

    tex_dir = out_dir / "textures"
    dbg_dir = out_dir / "debug"
    tex_dir.mkdir(parents=True, exist_ok=True)
    dbg_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_u8).save(tex_dir / f"{face}.png")
    Image.fromarray(raw_u8).save(dbg_dir / f"{face}_raw_projected.png")
    Image.fromarray(valid_mask).save(dbg_dir / f"{face}_valid_mask.png")
    Image.fromarray(hole_mask).save(dbg_dir / f"{face}_inpaint_mask.png")
    Image.fromarray(np.clip(weight_reliability * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_confidence.png")
    Image.fromarray(np.clip(observed_map.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_observed_mask.png")
    Image.fromarray(np.clip(reliable_map.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_reliable_mask.png")
    Image.fromarray(np.clip(reliability * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_reliability.png")
    Image.fromarray(np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_depth_residual.png")
    Image.fromarray(np.clip(color_std / max(args.color_std_clean_tol, 1e-6) * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_color_std.png")
    Image.fromarray(np.clip(object_risk_map * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_object_risk.png")
    Image.fromarray(np.clip(mask_boundary_trust_map * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_mask_boundary_trust.png")
    Image.fromarray(
        np.clip(footprint_map / max(effective_footprint_min_area, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    ).save(dbg_dir / f"{face}_footprint_area.png")
    Image.fromarray(np.clip(valid_ratio * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_valid_ratio.png")
    Image.fromarray(np.clip(clean_score * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_clean_score.png")
    Image.fromarray(np.clip(contamination_score * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_contamination_score.png")
    Image.fromarray(np.clip(final_keep_map.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)).save(dbg_dir / f"{face}_final_keep_mask.png")
    np.save(dbg_dir / f"{face}_valid_count.npy", count_map.astype(np.float32))
    np.save(dbg_dir / f"{face}_candidate_count.npy", candidate_map.astype(np.float32))
    np.save(dbg_dir / f"{face}_weight_sum.npy", weight_map.astype(np.float32))
    np.save(dbg_dir / f"{face}_color_std.npy", color_std)
    np.save(dbg_dir / f"{face}_object_risk.npy", object_risk_map.astype(np.float32))
    np.save(dbg_dir / f"{face}_depth_residual.npy", depth_map.astype(np.float32))
    np.save(dbg_dir / f"{face}_surface_distance.npy", surface_distance_map.astype(np.float32))
    np.save(dbg_dir / f"{face}_footprint_area.npy", footprint_map.astype(np.float32))
    np.save(dbg_dir / f"{face}_mask_boundary_trust.npy", mask_boundary_trust_map.astype(np.float32))
    np.save(dbg_dir / f"{face}_valid_ratio.npy", valid_ratio.astype(np.float32))
    np.save(dbg_dir / f"{face}_clean_score.npy", clean_score.astype(np.float32))
    np.save(dbg_dir / f"{face}_contamination_score.npy", contamination_score.astype(np.float32))

    nearest_record = None
    if args.emit_nearest_visible_evidence:
        assert nearest_rgb is not None
        assert nearest_view_id is not None
        assert nearest_distance is not None
        assert nearest_weight is not None
        nearest_record = save_nearest_visible_face_evidence(
            out_dir,
            face,
            nearest_rgb,
            nearest_view_id,
            nearest_distance,
            nearest_weight,
            final_keep_map,
            selected_poses,
        )

    return {
        "face": face,
        "texture_size": [w, h],
        "selected_views": [p.name for p in selected_poses],
        "observed_texels": int(np.count_nonzero(observed_map)),
        "reliable_texels": int(np.count_nonzero(reliable_map)),
        "final_kept_texels": int(np.count_nonzero(final_keep_map)),
        "valid_texels": int(np.count_nonzero(valid_mask > 0)),
        "mean_valid_count": float(count_map[observed_map].mean()) if np.any(observed_map) else 0.0,
        "mean_clean_score": float(clean_score[observed_map].mean()) if np.any(observed_map) else 0.0,
        "mean_reliability": float(reliability[observed_map].mean()) if np.any(observed_map) else 0.0,
        "footprint_min_area_global": float(args.footprint_min_area),
        "footprint_min_area_effective": effective_footprint_min_area,
        "adaptive_short_face_footprint_applied": adaptive_footprint_applied,
        "adaptive_horizontal_footprint_applied": adaptive_horizontal_footprint_applied,
        "horizontal_reliable_footprint_median": horizontal_reliable_footprint_median,
        "wall_length": wall_length,
        "median_wall_length": median_wall_length,
        "observed_footprint_median": observed_footprint_median,
        "nearest_visible_evidence": nearest_record,
    }


def write_preview(out_dir: Path, faces: list[str]) -> None:
    tiles = []
    for face in faces:
        path = out_dir / "textures" / f"{face}.png"
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        im.thumbnail((440, 280))
        tile = Image.new("RGB", (460, 320), (22, 22, 22))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 8), face, fill=(245, 245, 245))
        tile.paste(im, ((460 - im.width) // 2, 34))
        tiles.append(tile)
    if not tiles:
        return
    cols = 2 if len(tiles) <= 8 else 3
    rows = int(math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 460, rows * 320), (18, 18, 18))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 460, (i // cols) * 320))
    sheet.save(out_dir / "structured_texture_preview.jpg", quality=92)


def copy_mesh_files(source_dir: Path, out_dir: Path, scene_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_candidates = [source_dir / f"{scene_name}.obj", *sorted(source_dir.glob("*.obj"))]
    mtl_candidates = [source_dir / f"{scene_name}.mtl", *sorted(source_dir.glob("*.mtl"))]
    obj = next((p for p in obj_candidates if p.exists()), None)
    mtl = next((p for p in mtl_candidates if p.exists()), None)
    if obj:
        text = obj.read_text(encoding="utf-8")
        text = text.replace(obj.with_suffix(".mtl").name, f"{scene_name}.mtl")
        (out_dir / f"{scene_name}.obj").write_text(text, encoding="utf-8")
    if mtl:
        shutil.copy2(mtl, out_dir / f"{scene_name}.mtl")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave a previous authoritative manifest/sidecar
    # looking current. Per-face files may be diagnostic, but consumers require
    # these two newly committed records (and the enclosing strict success marker).
    (args.out_dir / "photo_source_manifest.json").unlink(missing_ok=True)
    (args.out_dir / "nearest_visible_evidence.json").unlink(missing_ok=True)
    manifest_path = args.polygon_source_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = args.polygon_source_dir / "metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metas = face_meta_map(manifest)
    faces = face_names(manifest, args.faces)

    hf_alignment = None
    pose_source = "colmap_icp" if args.pose_source == "colmap" else args.pose_source
    if pose_source == "da3_hfalign":
        if args.da3_dir is None:
            raise ValueError("--pose-source da3_hfalign requires --da3-dir")
        poses, sim, hf_alignment = load_da3_hfalign_poses(args.dataset_dir, args.da3_dir)
        print(
            f"[info] Using DA3 hf-aligned poses in exported scene.glb coordinates: views={len(poses)}",
            flush=True,
        )
    elif pose_source == "da3_raw":
        if args.da3_dir is None:
            raise ValueError("--pose-source da3_raw requires --da3-dir")
        poses, sim = load_da3_raw_poses(args.dataset_dir, args.da3_dir)
        print(
            f"[info] Using DA3 raw numeric poses without scene.glb hf_alignment: views={len(poses)}",
            flush=True,
        )
    else:
        rec = pycolmap_reconstruction(args.colmap_model_dir)
        colmap_points = np.asarray([p.xyz for p in rec.points3D.values()], dtype=np.float64)
        da3_points, _ = load_point_cloud_glb(args.dataset_dir / "scene.glb")
        sim = align_colmap_to_da3(colmap_points, da3_points.astype(np.float64), args)
        poses = load_image_poses(rec, args.dataset_dir, sim)
    da3_views = load_da3_views(args.da3_dir, poses)
    da3_numeric_to_room = None if pose_source in {"da3_hfalign", "da3_raw"} else (align_da3_numeric_world_to_room(da3_views, poses) if da3_views else None)
    da3_raw_to_room_matrix = hf_alignment if pose_source == "da3_hfalign" else None
    if args.da3_dir is not None and not da3_views:
        print(f"[warn] DA3 numeric files were requested but not usable: {args.da3_dir}", flush=True)
    elif da3_views:
        print(f"[info] Loaded DA3 numeric depth/conf for {len(da3_views)}/{len(poses)} registered views", flush=True)
        if da3_numeric_to_room is not None:
            print(
                "[info] Aligned DA3 numeric camera world to room DA3 world: "
                f"scale={da3_numeric_to_room.scale:.6f}, p65_camera_error={da3_numeric_to_room.score:.6f}",
                flush=True,
            )
        else:
            print("[warn] Could not align DA3 numeric camera world to room DA3 world", flush=True)
    copy_mesh_files(args.polygon_source_dir, args.out_dir, args.scene_name)
    if manifest_path.exists():
        shutil.copy2(manifest_path, args.out_dir / "metadata.json")

    stats = []
    zbuffer_cache: dict[int, np.ndarray] = {}
    face_id_cache: dict[int, np.ndarray] = {}
    view_reject_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    da3_depth_calibration_cache: dict[int, Da3DepthCalibration | None] = {}
    for face in faces:
        size = face_texture_size(args.polygon_source_dir, metas[face])
        selected = choose_views_for_face(face, size, poses, sim, manifest, metas, args)
        if not selected:
            selected = poses[: min(len(poses), args.views_per_face)]
        print(f"[face] {face}: {len(selected)} selected views", flush=True)
        stats.append(
            process_face(
                face,
                args.polygon_source_dir,
                args.out_dir,
                poses,
                selected,
                sim,
                manifest,
                metas,
                faces,
                args,
                zbuffer_cache,
                face_id_cache,
                view_reject_cache,
                da3_views,
                da3_numeric_to_room,
                da3_raw_to_room_matrix,
                da3_depth_calibration_cache,
            )
        )

    write_preview(args.out_dir, faces)
    out_manifest = {
        "method": "polygon_photo_source_from_colmap_v53_v60_weight_replicated_v2",
        "dataset_dir": str(args.dataset_dir),
        "polygon_source_dir": str(args.polygon_source_dir),
        "colmap_model_dir": str(args.colmap_model_dir),
        "pose_source": pose_source,
        "da3_dir": str(args.da3_dir) if args.da3_dir else None,
        "da3_numeric_views": len(da3_views),
        "registered_images": len(poses),
        "faces": stats,
        "alignment_colmap_to_da3": {
            "scale": float(sim.scale),
            "rotation": sim.rotation.tolist(),
            "translation": sim.translation.tolist(),
            "score_p65": float(sim.score),
        },
        "hf_alignment_scene_glb": hf_alignment.tolist() if hf_alignment is not None else None,
        "alignment_da3_numeric_to_room_da3": (
            {
                "scale": float(da3_numeric_to_room.scale),
                "rotation": da3_numeric_to_room.rotation.tolist(),
                "translation": da3_numeric_to_room.translation.tolist(),
                "camera_error_p65": float(da3_numeric_to_room.score),
                "matched_views": len(da3_views),
            }
            if da3_numeric_to_room is not None
            else None
        ),
        "da3_depth_calibration": {
            "method": "per-view robust affine calibration from DA3 depth map to COLMAP room-shell z-buffer",
            "usable_views": int(sum(1 for c in da3_depth_calibration_cache.values() if c is not None)),
            "total_views": int(len(da3_depth_calibration_cache)),
            "views": {
                str(image_id): (
                    {
                        "mode": c.mode,
                        "scale": c.scale,
                        "shift": c.shift,
                        "median_abs_error": c.median_abs_error,
                        "p80_abs_error": c.p80_abs_error,
                        "samples": c.samples,
                    }
                    if c is not None
                    else None
                )
                for image_id, c in sorted(da3_depth_calibration_cache.items())
            },
        },
        "weight_terms": {
            "formula": (
                "w_i = da3_conf_i * max(cos_i,0)^2 * distance_weight_i * "
                "depth_weight_i * surface_weight_i * boundary_weight_i * footprint_weight_i"
            ),
            "da3_conf": (
                "sampled from DA3 conf.npy in DA3 camera coordinates"
                if da3_views
                else "constant 1.0 fallback because no usable DA3 numeric confidence file was present"
            ),
            "angle": "max(abs(dot(face_normal, camera_direction)),0)^2",
            "distance": "1 / (1 + view_distance / distance_weight_scale)^distance_weight_power",
            "depth_tol": "depth_abs_tol + depth_rel_tol * camera_depth",
            "depth_weight": "1 / (1 + depth_diff / depth_tol)",
            "depth_source": (
                "DA3 depth.npy sampled at COLMAP image pixels and per-view calibrated to the COLMAP room-shell z-buffer before v60 depth gating"
                if da3_views
                else "COLMAP-pose projection against DA3 polygon room-shell z-buffer fallback"
            ),
            "surface_distance": (
                "DA3 depth point calibrated to camera z, back-projected to room coordinates, "
                "and compared against the current structural face plane"
            ),
            "surface_weight": "1 / (1 + surface_distance / surface_distance_tol)",
            "surface_clean_penalty": (
                "0.60 * clamp(surface_distance / surface_distance_clean_tol, 0, 1); "
                "this scoring scale does not change the hard geometry gate"
            ),
            "surface_hard_gate": (
                "surface_distance <= surface_distance_tol"
                if args.surface_distance_hard_gate and args.surface_distance_tol > 0.0
                else "disabled"
            ),
            "surface_normal_gate": (
                f"abs(dot(DA3_depth_normal, face_normal)) >= {args.surface_normal_min_cos}"
                if args.surface_normal_min_cos > 0.0
                else "disabled"
            ),
            "same_face_gate": "sampled nearest visible polygon face id must equal the current texture face, matching v53 sampled_face == face_id",
            "boundary_weight": "sampled mask-boundary trust from optional per-view reject masks; defaults to 1.0 when masks are absent",
            "object_risk_hard_gate": (
                f"sampled dilated object risk <= {args.object_risk_hard_thresh}"
                if args.object_risk_hard_thresh <= 1.0
                else "disabled"
            ),
            "mask_boundary_hard_gate": (
                f"sampled mask-boundary trust >= {args.min_mask_boundary_trust}"
                if args.min_mask_boundary_trust > 0.0
                else "disabled"
            ),
            "footprint": "clamp(projected source-image pixel area per atlas texel / footprint_min_area,0,1)^footprint_power",
        },
        "replicated_v60_parameters": {
            "depth_abs_tol": args.depth_abs_tol,
            "depth_rel_tol": args.depth_rel_tol,
            "distance_weight_scale": args.distance_weight_scale,
            "distance_weight_power": args.distance_weight_power,
            "min_view_cos": args.min_view_cos,
            "min_conf": args.min_conf,
            "mask_boundary_safe_px": args.mask_boundary_safe_px,
            "mask_boundary_power": args.mask_boundary_power,
            "min_mask_boundary_trust": args.min_mask_boundary_trust,
            "object_risk_hard_thresh": args.object_risk_hard_thresh,
            "footprint_min_area": args.footprint_min_area,
            "footprint_power": args.footprint_power,
            "adaptive_short_face_footprint": args.adaptive_short_face_footprint,
            "short_face_length_median_frac": args.short_face_length_median_frac,
            "short_face_footprint_median_multiplier": args.short_face_footprint_median_multiplier,
            "short_face_footprint_min_area": args.short_face_footprint_min_area,
            "surface_distance_tol": args.surface_distance_tol,
            "surface_distance_clean_tol": (
                args.surface_distance_clean_tol
                if args.surface_distance_clean_tol > 0.0
                else args.surface_distance_tol
            ),
            "surface_distance_power": args.surface_distance_power,
            "surface_distance_hard_gate": args.surface_distance_hard_gate,
            "color_std_clean_tol": args.color_std_clean_tol,
            "valid_ratio_penalty": args.valid_ratio_penalty,
            "min_sample_weight": args.min_sample_weight,
            "min_output_reliability": args.min_output_reliability,
            "min_output_clean_score": args.min_output_clean_score,
            "max_output_contamination_score": args.max_output_contamination_score,
            "max_output_object_risk": args.max_output_object_risk,
            "strict_empty_low_quality": args.strict_empty_low_quality,
            "emit_nearest_visible_evidence": args.emit_nearest_visible_evidence,
            "nearest_visible_selection": (
                {
                    "policy": NEAREST_SELECTION_POLICY,
                    "distance_definition": (
                        "Euclidean camera-center-to-surface-texel distance in the polygon room coordinate frame"
                    ),
                    "distance_tie_absolute_tolerance": NEAREST_DISTANCE_TIE_ATOL,
                    "rgb_sampling": "the same bilinear source-image sample used by weighted fusion",
                    "uint8_quantization": "clip(sample*255,0,255) then uint8 truncation",
                    "winner_domain": (
                        "same strict accepted samples with original projection weight > 1e-8, then final_keep"
                    ),
                }
                if args.emit_nearest_visible_evidence
                else None
            ),
            "min_valid_views": args.min_valid_views,
            "object_mask_dir": str(args.object_mask_dir) if args.object_mask_dir else None,
            "views_per_face": args.views_per_face,
        },
    }
    write_text_atomic(
        args.out_dir / "photo_source_manifest.json",
        json.dumps(out_manifest, indent=2, ensure_ascii=False),
    )
    if args.emit_nearest_visible_evidence:
        write_nearest_visible_sidecar(args.out_dir, out_manifest)
    print(json.dumps({"out_dir": str(args.out_dir), "preview": str(args.out_dir / "structured_texture_preview.jpg")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
