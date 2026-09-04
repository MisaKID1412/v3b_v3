#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from regenerate_matseg_material_chord_inputs_by_traceback import (
    load_generator_module,
    namespace_from_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compose CHORD material territories from a locked unified trace-back "
            "selection without changing any stored projection or selection parameter."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--chord-output-dir", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    args.package_dir.mkdir(parents=True, exist_ok=True)

    # compose_stage deliberately reads this canonical filename.  Copying the
    # locked file keeps the compose package self-contained and leaves the
    # selected metadata untouched.
    canonical = args.package_dir / "metadata_view_contributor_chord_inputs.json"
    shutil.copy2(args.metadata, canonical)

    generator = load_generator_module(args.generator, "unified_trace_compose_generator")
    compose_args = namespace_from_metadata(metadata, args.package_dir, generator)
    compose_args.stage = "compose"
    compose_args.out_dir = args.package_dir
    compose_args.chord_output_dir = args.chord_output_dir
    compose_args.basecolor_key = "basecolor"
    compose_args.pbr_keys = "basecolor,normal,roughness,metallic"

    result = int(generator.compose_stage(compose_args))
    receipt = {
        "status": "complete" if result == 0 else "failed",
        "method": "unified_matseg_traceback_to_original_v3b_compose",
        "selected_metadata": str(args.metadata),
        "canonical_metadata_copy": str(canonical),
        "chord_output_dir": str(args.chord_output_dir),
        "generator": str(args.generator),
        "selection_parameters_changed": False,
        "compose_return_code": result,
    }
    (args.package_dir / "unified_compose_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
