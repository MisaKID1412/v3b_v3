#!/usr/bin/env python3
"""Replay the single generalized material-proposal policy on staged v3b data.

This utility exists for regression against historical rooms.  A fresh
image-to-Unity run applies the same values at its material-proposal stage; no
room profile is selected here.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path


PATH_KEYS = {
    "source_dir",
    "polygon_source_dir",
    "dataset_dir",
    "colmap_model_dir",
    "da3_dir",
    "object_mask_dir",
    "out_dir",
    "chord_output_dir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--generator-script", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--material-max-per-face", type=int, default=8)
    parser.add_argument("--material-cluster-components", type=int, default=4)
    parser.add_argument("--material-cluster-min-fraction", type=float, default=0.01)
    parser.add_argument("--material-cluster-min-region-size", type=int, default=32)
    return parser.parse_args()


def load_generator(path: Path):
    scripts_dir = str(path.resolve().parent)
    sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("v3b_unified_region_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parser_defaults(generator, old: dict, out_dir: Path) -> argparse.Namespace:
    argv = [str(generator.__file__), "--stage", "prepare"]
    for key in (
        "source_dir",
        "polygon_source_dir",
        "dataset_dir",
        "colmap_model_dir",
        "da3_dir",
        "object_mask_dir",
    ):
        argv.extend([f"--{key.replace('_', '-')}", str(old[key])])
    argv.extend(["--out-dir", str(out_dir)])
    saved = sys.argv
    try:
        sys.argv = argv
        values = vars(generator.parse_args())
    finally:
        sys.argv = saved
    values.update(copy.deepcopy(old))
    for key in PATH_KEYS:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    values["out_dir"] = out_dir
    return argparse.Namespace(**values)


def apply_unified_policy(
    args: argparse.Namespace, cli: argparse.Namespace
) -> dict:
    policy = {
        # Four appearance components are enough to expose distinct materials
        # while avoiding the unstable illumination shards produced by K=8.
        # MatSeg, not this component count, decides final material identity.
        "material_cluster_components": int(cli.material_cluster_components),
        "material_cluster_min_fraction": float(cli.material_cluster_min_fraction),
        "material_cluster_min_region_size": int(cli.material_cluster_min_region_size),
        "max_priors_per_face": int(cli.material_max_per_face),
        "single_material_faces": None,
        "face_max_materials": None,
        # Lab creates deliberately fine spatial proposals only. It is not
        # allowed to decide material identity before MatSeg.
        "material_cluster_chroma_merge_threshold": 0.0,
        "discover_persistent_wall_bands": True,
        "wall_band_min_texture_delta": 0.18,
        "wall_band_max_count": 4,
        # Keep a detected structural strip even when the first strict 512px
        # rectification cannot cover it.  The later weight-ordered trace-back
        # stage retries with a window derived from the strip geometry.
        "wall_band_min_traceable_views": 0,
        # Do not delete a face's only usable proposal at this early stage.
        # Cross-face projection contamination is evaluated geometrically after
        # MatSeg identity groups have been formed.
        "reject_cross_face_edge_singletons": False,
        "remove_tiny_material_islands": False,
        # The trace-back patch must come only from strict observed support.
        # Geometry-adaptive window sizing handles narrow territories without
        # growing them by colour similarity.
        # This was enabled in all three accepted regressions.  It changes only
        # how a narrow spatial proposal obtains an original-image observation;
        # MatSeg still makes the same/different-material decision later.
        "thin_territory_source_adaptation": True,
    }
    for key, value in policy.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return policy


def main() -> None:
    cli = parse_args()
    source = json.loads(cli.source_metadata.read_text(encoding="utf-8"))
    generator = load_generator(cli.generator_script)
    generator_args = parser_defaults(generator, source["params"], cli.out_dir)
    policy = apply_unified_policy(generator_args, cli)
    cli.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "method": "unified_v5_single_generalized_material_proposal_policy",
        "source_metadata": str(cli.source_metadata),
        "generator_script": str(cli.generator_script),
        "room_profile_or_route_selection_used": False,
        "policy": policy,
    }
    (cli.out_dir / "unified_proposal_policy.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    raise SystemExit(generator.prepare_stage(generator_args))


if __name__ == "__main__":
    main()
