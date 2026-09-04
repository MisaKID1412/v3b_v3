#!/usr/bin/env python3
"""Audit channel completeness, normalization, provenance, and BaseColor drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


CHANNELS = ("basecolor", "normal", "roughness", "metallic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--basecolor-only-chord-dir", type=Path)
    parser.add_argument("--reference-basecolor-dir", type=Path)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, mode: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0


def main() -> int:
    args = parse_args()
    metadata_path = args.run_dir / "metadata_pbr_placement.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    warnings: list[str] = []
    contract = metadata.get("channel_contract", {})
    adaptive_records = metadata.get("faces", [])
    if not contract and metadata.get("generalization_contract") and adaptive_records:
        # The accepted adaptive backend records the joint mapping on every
        # material rather than repeating a legacy top-level channel_contract.
        # Derive the summary only from those explicit per-material assertions;
        # the file/shape/normal checks below still verify the emitted channels.
        adaptive_materials = [
            material
            for face_record in adaptive_records
            for material in face_record.get("materials", [])
        ]
        shared = bool(adaptive_materials) and all(
            str(material.get("pbr_spatial_mapping", "")).startswith("shared_")
            for material in adaptive_materials
        )
        no_independent_quilting = bool(adaptive_materials) and all(
            material.get("contains_fixed_tile_grid") is False
            and material.get("contains_image_patch_cut_and_paste") is False
            for material in adaptive_materials
        )
        contract = {
            "source": "derived_from_adaptive_per_material_contracts",
            "shared_spatial_mapping": shared,
            "independent_channel_quilting": not no_independent_quilting,
            "channels": list(CHANNELS),
        }
    if not contract.get("shared_spatial_mapping"):
        failures.append("metadata does not assert shared PBR spatial mapping")
    if contract.get("independent_channel_quilting") is not False:
        failures.append("independent channel quilting is not explicitly disabled")
    if contract.get("channels") != list(CHANNELS):
        failures.append(f"unexpected channel contract: {contract.get('channels')}")

    face_reports: dict[str, dict[str, object]] = {}
    seen_faces: set[str] = set()
    for record in metadata.get("faces", []):
        face = str(record["face"])
        seen_faces.add(face)
        sizes: dict[str, list[int]] = {}
        arrays: dict[str, np.ndarray] = {}
        for channel in CHANNELS:
            path = args.run_dir / "pbr_textures" / channel / f"{face}.png"
            if not path.exists():
                failures.append(f"missing {face}/{channel}")
                continue
            mode = "RGB" if channel in ("basecolor", "normal") else "L"
            arrays[channel] = load(path, mode)
            sizes[channel] = [int(arrays[channel].shape[1]), int(arrays[channel].shape[0])]
        if len(set(tuple(size) for size in sizes.values())) > 1:
            failures.append(f"channel sizes disagree for {face}: {sizes}")
        if len(arrays) != len(CHANNELS):
            continue
        normal = arrays["normal"] * 2.0 - 1.0
        normal_length = np.linalg.norm(normal, axis=2)
        normal_mean_error = float(np.mean(np.abs(normal_length - 1.0)))
        normal_p99_error = float(np.percentile(np.abs(normal_length - 1.0), 99))
        if normal_mean_error > 0.012 or normal_p99_error > 0.03:
            failures.append(
                f"normal normalization failed for {face}: mean={normal_mean_error:.5f}, p99={normal_p99_error:.5f}"
            )
        labels_path = args.run_dir / "labels_npy" / f"{face}.npy"
        if not labels_path.exists():
            failures.append(f"missing hard territory labels for {face}")
            material_count = 0
        else:
            labels = np.load(labels_path)
            if labels.shape != arrays["basecolor"].shape[:2]:
                failures.append(
                    f"label/texture shape mismatch for {face}: {labels.shape} vs {arrays['basecolor'].shape[:2]}"
                )
            material_count = int(len(np.unique(labels)))
        material_mappings = [str(item.get("pbr_spatial_mapping", "")) for item in record.get("materials", [])]
        if any(not mapping.startswith("shared_") for mapping in material_mappings):
            failures.append(f"non-shared spatial mapping recorded for {face}: {material_mappings}")
        if any(item.get("contains_fixed_tile_grid") is not False for item in record.get("materials", [])):
            failures.append(f"fixed tile grid was not explicitly disabled for {face}")
        if any(
            item.get("contains_image_patch_cut_and_paste") is not False
            for item in record.get("materials", [])
        ):
            failures.append(f"image patch cut-and-paste was not explicitly disabled for {face}")

        reference_report = None
        if args.reference_basecolor_dir is not None:
            reference_path = args.reference_basecolor_dir / f"{face}.png"
            if reference_path.exists():
                reference = load(reference_path, "RGB")
                current = arrays["basecolor"]
                if reference.shape == current.shape:
                    delta = np.abs(current - reference)
                    reference_report = {
                        "mean_absolute_8bit_delta": float(np.mean(delta) * 255.0),
                        "p99_absolute_8bit_delta": float(np.percentile(delta, 99) * 255.0),
                        "max_absolute_8bit_delta": float(np.max(delta) * 255.0),
                        "pixel_identical": bool(np.array_equal(current, reference)),
                    }
                else:
                    warnings.append(f"reference size differs for {face}")
        face_reports[face] = {
            "sizes_wh": sizes,
            "material_count_from_labels": material_count,
            "normal_mean_unit_length_error": normal_mean_error,
            "normal_p99_unit_length_error": normal_p99_error,
            "roughness_min_max": [float(np.min(arrays["roughness"])), float(np.max(arrays["roughness"]))],
            "metallic_min_max": [float(np.min(arrays["metallic"])), float(np.max(arrays["metallic"]))],
            "spatial_mappings": material_mappings,
            "reference_basecolor_difference": reference_report,
        }

    disk_faces = {
        path.stem for path in (args.run_dir / "pbr_textures" / "basecolor").glob("*.png")
    }
    if disk_faces != seen_faces:
        failures.append(
            f"metadata/basecolor face set mismatch: metadata={sorted(seen_faces)}, disk={sorted(disk_faces)}"
        )

    chord_report: dict[str, object] | None = None
    if args.basecolor_only_chord_dir is not None:
        full_root = Path(str(metadata["source_full_chord_outputs"]))
        stems = sorted(path.name for path in full_root.iterdir() if path.is_dir())
        mismatches = []
        missing = []
        for stem in stems:
            full_path = full_root / stem / "basecolor.png"
            old_path = args.basecolor_only_chord_dir / stem / "basecolor.png"
            if not full_path.exists() or not old_path.exists():
                missing.append(stem)
            elif sha256(full_path) != sha256(old_path):
                mismatches.append(stem)
        chord_report = {
            "material_count": len(stems),
            "full_vs_basecolor_only_identical_count": len(stems) - len(missing) - len(mismatches),
            "missing": missing,
            "mismatches": mismatches,
        }
        if missing or mismatches:
            failures.append(f"full CHORD BaseColor drift: missing={missing}, mismatches={mismatches}")

    report = {
        "audit": "wholefield_aligned_pbr_v2",
        "passed": not failures,
        "run_dir": str(args.run_dir),
        "channel_contract": contract,
        "face_count": len(face_reports),
        "faces": face_reports,
        "chord_basecolor_regression": chord_report,
        "warnings": warnings,
        "failures": failures,
    }
    out_path = args.out_json or args.run_dir / "audit_wholefield_pbr.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "face_count": len(face_reports), "failures": failures}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
