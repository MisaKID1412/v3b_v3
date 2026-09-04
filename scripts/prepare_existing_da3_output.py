#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = ("depth", "conf", "extrinsics", "intrinsics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an existing DA3 NPZ/GLB export into the numeric directory contract used by v3b."
    )
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--scene-glb", type=Path, required=True)
    parser.add_argument("--camera-poses-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="existing_da3_export")
    return parser.parse_args()


def load_image_names(path: Path, count: int) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = data.get("poses", [])
    names = [Path(str(item.get("image_file", ""))).name for item in poses]
    if len(names) != count or any(not name for name in names):
        raise ValueError(f"Expected {count} ordered image_file entries in {path}, found {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate image names in {path}")
    return names


def main() -> int:
    args = parse_args()
    for path in (args.npz, args.scene_glb, args.camera_poses_json):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(args.npz, allow_pickle=False) as payload:
        missing = [name for name in REQUIRED_ARRAYS if name not in payload]
        if missing:
            raise KeyError(f"Missing DA3 arrays in {args.npz}: {missing}")
        arrays = {name: np.asarray(payload[name]) for name in REQUIRED_ARRAYS}

    count = int(arrays["depth"].shape[0])
    if arrays["depth"].ndim != 3 or arrays["conf"].shape != arrays["depth"].shape:
        raise ValueError("depth/conf must have the same [view, height, width] shape")
    if arrays["extrinsics"].shape not in {(count, 3, 4), (count, 4, 4)}:
        raise ValueError(f"Unexpected extrinsics shape: {arrays['extrinsics'].shape}")
    if arrays["intrinsics"].shape not in {(count, 3, 3), (count, 4)}:
        raise ValueError(f"Unexpected intrinsics shape: {arrays['intrinsics'].shape}")

    image_names = load_image_names(args.camera_poses_json, count)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(args.out_dir / f"{name}.npy", array)
    shutil.copy2(args.scene_glb, args.out_dir / "scene.glb")
    shutil.copy2(args.camera_poses_json, args.out_dir / "camera_poses.json")

    meta = {
        "source": "adapted_existing_da3_export",
        "model_dir": args.model_name,
        "source_npz": str(args.npz),
        "source_scene_glb": str(args.scene_glb),
        "source_camera_poses_json": str(args.camera_poses_json),
        "image_names": image_names,
        "num_images": count,
        "process_res": int(max(arrays["depth"].shape[1:])),
        "process_res_method": "recorded_by_existing_export",
        "ref_view_strategy": "recorded_by_existing_export",
        "export_format": "glb",
        "extrinsics_convention": "w2c",
        "intrinsics_coordinate_space": "processed_images",
        "depth_shape": list(arrays["depth"].shape),
        "conf_shape": list(arrays["conf"].shape),
        "extrinsics_shape": list(arrays["extrinsics"].shape),
        "intrinsics_shape": list(arrays["intrinsics"].shape),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "views": count, "image_names": image_names}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
