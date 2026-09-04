#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-metadata", type=Path, required=True)
    parser.add_argument("--material-dir", type=Path, required=True)
    parser.add_argument("--completed-observed-dir", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace = json.loads(args.trace_metadata.read_text(encoding="utf-8"))
    params = trace["params"]
    source_package = Path(params["polygon_source_dir"])
    strict_projection = Path(params["source_dir"])
    completed = args.completed_observed_dir
    data = {
        "status": "generated_for_unified_matseg_traceback_backend",
        "experiment_root": str(args.experiment_root),
        "dataset_dir": str(params["dataset_dir"]),
        "da3_dir": str(params["da3_dir"]),
        "colmap_model_dir": str(params["colmap_model_dir"]),
        "mesh_source_dir": str(source_package),
        "strict_observed_projection_dir": str(strict_projection),
        "completed_observed_lama_dir": str(completed),
        "completed_observed_dir": str(completed / "completed_observed"),
        "completed_observed_weight_dir": str(completed / "weights"),
        "chord_material_dir": str(args.material_dir),
        "mesh_manifest": str(source_package / "manifest.json"),
        "mesh_obj": str(source_package / "room_empty.obj"),
        "projection_metadata": str(strict_projection / "metadata.json"),
        "chord_inputs_metadata": str(
            args.material_dir / "metadata_view_contributor_chord_inputs.json"
        ),
        "chord_materials_metadata": str(
            args.material_dir / "metadata_view_contributor_chord_materials.json"
        ),
        "completed_observed_lama_metadata": str(
            completed / "metadata_completed_observed_lama.json"
        ),
        "frozen_projection_gates": {
            "surface_distance_tol": float(params.get("surface_distance_tol", 0.055)),
            "surface_distance_hard_gate": bool(
                params.get("surface_distance_hard_gate", True)
            ),
            "object_reject_version": "inherited_from_traceback_projection_metadata",
        },
        "selection_contract": {
            "trace_metadata": str(args.trace_metadata),
            "trace_method": trace.get("method"),
            "selection_parameters_changed": False,
            "room_specific_route_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
