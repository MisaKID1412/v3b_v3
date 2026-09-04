#!/usr/bin/env python3
"""Write a compact end-to-end provenance receipt for a completed v3b_v3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path, run_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(run_root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def file_record(path: Path, run_root: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": portable_path(path, run_root),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--identity-contract", type=Path, required=True)
    parser.add_argument("--trace-log", type=Path, required=True)
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--pbr-dir", type=Path, required=True)
    parser.add_argument("--unity-dir", type=Path, required=True)
    parser.add_argument("--unitypackage", type=Path, required=True)
    args = parser.parse_args()

    images = sorted(
        path for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    layout_metadata = args.layout_dir / "metadata_material_placement.json"
    pbr_metadata = args.pbr_dir / "metadata_adaptive_whole_territory_pbr.json"
    unity_manifest = args.unity_dir / "unity_export_manifest.json"
    if not unity_manifest.is_file():
        candidates = sorted(args.unity_dir.glob("*manifest*.json"))
        if not candidates:
            raise FileNotFoundError(unity_manifest)
        unity_manifest = candidates[0]

    receipt = {
        "method": "v3b_v3_raw_images_unified_matseg_traceback_chord_whole_territory_pbr_unity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "image_dir": args.image_dir.name,
            "image_count": len(images),
            "images": [
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in images
            ],
        },
        "contracts": {
            "matseg_scope": "material identity only",
            "patch_selection": "maximum atlas projection-weight support then geometric source-image traceback",
            "chord_channels": ["basecolor", "normal", "roughness", "metallic"],
            "territory_synthesis": "scale-locked whole-field; no fixed tile grid and no patch cut-and-paste",
        },
        "artifacts": {
            "identity_contract": file_record(args.identity_contract, args.run_root),
            "traceback_log": file_record(args.trace_log, args.run_root),
            "layout_metadata": file_record(layout_metadata, args.run_root),
            "pbr_metadata": file_record(pbr_metadata, args.run_root),
            "unity_manifest": file_record(unity_manifest, args.run_root),
            "unitypackage": file_record(args.unitypackage, args.run_root),
        },
    }
    output = args.run_root / "v3b_v3_end_to_end_manifest.json"
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[v3b_v3] wrote end-to-end manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
