#!/usr/bin/env python3
"""Reuse CHORD PBR outputs only when the complete input image hash matches.

This is an inference-cache utility, not a material selector. Missing hashes are
copied to a pending directory and must be processed by the unchanged CHORD
runner.  Cache roots are supplied by the caller so inference profiles (for
example 512 versus 2048) are never mixed implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REQUIRED_MAPS = (
    "input.png",
    "basecolor.png",
    "normal.png",
    "roughness.png",
    "metallic.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pending-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, action="append", default=[])
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def complete_output(directory: Path) -> bool:
    return all((directory / name).is_file() for name in REQUIRED_MAPS)


def build_index(roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for input_path in root.rglob("input.png"):
            directory = input_path.parent
            if complete_output(directory):
                index.setdefault(digest(input_path), directory)
    return index


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The pending directory is a derived work queue.  Rebuild it on every
    # invocation so a completed/resumed run cannot be mistaken for unfinished
    # work merely because an old queue entry is still present.
    if args.pending_dir.exists():
        shutil.rmtree(args.pending_dir)
    args.pending_dir.mkdir(parents=True, exist_ok=True)
    cache = build_index(args.cache_root)
    records = []
    for input_path in sorted(args.input_dir.glob("*.png")):
        input_hash = digest(input_path)
        stem = input_path.stem
        existing = args.output_dir / stem
        if (
            complete_output(existing)
            and (existing / "input.png").is_file()
            and digest(existing / "input.png") == input_hash
        ):
            records.append(
                {
                    "stem": stem,
                    "input_sha256": input_hash,
                    "status": "reused_existing_exact_input_hash",
                    "cache_source": str(existing),
                }
            )
            continue
        cached = cache.get(input_hash)
        if cached is None:
            shutil.copy2(input_path, args.pending_dir / input_path.name)
            records.append(
                {
                    "stem": stem,
                    "input_sha256": input_hash,
                    "status": "pending_unchanged_chord_inference",
                }
            )
            continue
        destination = args.output_dir / stem
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(cached, destination)
        records.append(
            {
                "stem": stem,
                "input_sha256": input_hash,
                "status": "reused_exact_input_hash_cache",
                "cache_source": str(cached),
            }
        )
    receipt = {
        "method": "exact_chord_input_sha256_inference_cache_v1",
        "material_selection_changed": False,
        "cache_roots": [str(path) for path in args.cache_root],
        "reused": sum(row["status"].startswith("reused") for row in records),
        "pending": sum(row["status"].startswith("pending") for row in records),
        "records": records,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
