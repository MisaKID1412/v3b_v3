#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt repeated spatial exemplars of one locked material identity to the "
            "unique-record interface expected by the v1 material-layout backend."
        )
    )
    parser.add_argument("--locked-metadata", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--chord-output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.package_dir / "metadata_view_contributor_chord_inputs.json"
    material_path = args.package_dir / "metadata_view_contributor_chord_materials.json"
    inputs = json.loads(args.locked_metadata.read_text(encoding="utf-8"))

    locked_copy = args.package_dir / "metadata_view_contributor_chord_inputs.locked.json"
    shutil.copy2(args.locked_metadata, locked_copy)
    locked_material_copy = args.package_dir / "metadata_view_contributor_chord_materials.locked.json"
    legacy_locked_material_copy = (
        args.package_dir / "metadata_view_contributor_chord_materials.locked_v5.json"
    )
    if locked_material_copy.exists():
        material_source = locked_material_copy
    elif legacy_locked_material_copy.exists():
        # One-time migration for work directories created before the neutral
        # naming.  The legacy file is still the unadapted source of truth.
        shutil.copy2(legacy_locked_material_copy, locked_material_copy)
        material_source = locked_material_copy
    else:
        shutil.copy2(material_path, locked_material_copy)
        material_source = locked_material_copy
    materials = json.loads(material_source.read_text(encoding="utf-8"))

    stem_map: dict[tuple[str, int, str], str] = {}
    copied_masks: list[str] = []
    for face_info in inputs["stats"]:
        face = str(face_info["face"])
        for region in face_info.get("regions", []):
            region_index = int(region["region"])
            for candidate in region.get("view_candidates", []):
                old_stem = str(candidate["stem"])
                new_stem = f"{old_stem}__e{region_index:02d}"
                stem_map[(face, region_index, old_stem)] = new_stem
                candidate["locked_material_stem"] = old_stem
                candidate["stem"] = new_stem
                if args.chord_output_dir is not None:
                    source_output = args.chord_output_dir / old_stem
                    alias_output = args.chord_output_dir / new_stem
                    if not source_output.is_dir():
                        raise FileNotFoundError(source_output)
                    if alias_output.is_symlink():
                        alias_output.unlink()
                    elif alias_output.exists():
                        raise RuntimeError(f"refusing to replace non-symlink {alias_output}")
                    os.symlink(source_output.name, alias_output, target_is_directory=True)

            # MatSeg has already merged spatial exemplars into one material
            # identity.  Its strict merged support is the correct v1 seed: it is
            # disjoint identity evidence.  The broader proposal territories can
            # overlap and therefore must not be treated as hard ownership here;
            # the v1 structured layout expands these supports over the face.
            material_id = int(region.get("material_id", region_index))
            source_mask = (
                args.package_dir
                / "debug"
                / f"{face}_material_{material_id:02d}_merged_support.png"
            )
            if not source_mask.exists():
                target_tile = Path(region["target_tile"])
                proposal_debug = target_tile.parent.parent / "debug"
                source_mask = (
                    proposal_debug / f"{face}_region_{region_index:02d}_material_mask.png"
                )
            target_mask = args.package_dir / "debug" / source_mask.name
            target_mask = (
                args.package_dir
                / "debug"
                / f"{face}_region_{region_index:02d}_material_mask.png"
            )
            if source_mask.exists():
                target_mask.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_mask, target_mask)
                copied_masks.append(str(target_mask))

    for face_info in materials.get("stats", []):
        face = str(face_info["face"])
        for record in face_info.get("regions", []):
            region_index = int(record["region"])
            old_stem = str(record["chosen_stem"])
            key = (face, region_index, old_stem)
            if key not in stem_map:
                raise RuntimeError(f"missing locked candidate for {key}")
            record["locked_material_stem"] = old_stem
            record["chosen_stem"] = stem_map[key]
            for score in record.get("candidate_scores", []):
                score_stem = str(score.get("stem", ""))
                mapped = stem_map.get((face, region_index, score_stem))
                if mapped is not None:
                    score["locked_material_stem"] = score_stem
                    score["stem"] = mapped

    inputs["v1_backend_adapter"] = {
        "method": "unique_exemplar_ids_for_repeated_locked_material_evidence",
        "material_identity_or_candidate_changed": False,
        "chord_output_aliases_created": len(stem_map)
        if args.chord_output_dir is not None
        else 0,
        "reason": (
            "The v1 layout backend accepts several exemplar regions per material but "
            "requires each exemplar record to have a unique identifier."
        ),
        "locked_metadata": str(args.locked_metadata),
    }
    materials["v1_backend_adapter"] = inputs["v1_backend_adapter"]
    input_path.write_text(json.dumps(inputs, indent=2, ensure_ascii=False), encoding="utf-8")
    material_path.write_text(
        json.dumps(materials, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    receipt = {
        "status": "complete",
        "locked_metadata": str(args.locked_metadata),
        "adapted_inputs_metadata": str(input_path),
        "adapted_materials_metadata": str(material_path),
        "locked_materials_metadata": str(locked_material_copy),
        "unique_exemplar_ids": len(stem_map),
        "copied_discovery_material_masks": len(copied_masks),
        "material_identity_or_candidate_changed": False,
        "copied_masks": copied_masks,
    }
    (args.package_dir / "v1_backend_adapter_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "copied_masks"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
