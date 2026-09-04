#!/usr/bin/env python3
"""Validate the immutable layout/trace-back/CHORD contract before synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from generate_scale_locked_whole_material_fields import candidate_map, face_record_map, load_json


CHANNELS = ("basecolor", "normal", "roughness", "metallic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-dir", type=Path, required=True)
    parser.add_argument("--chord-input-metadata", type=Path, required=True)
    parser.add_argument("--pbr-chord-output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout_path = args.layout_dir / "metadata_material_placement.json"
    if not layout_path.is_file():
        raise FileNotFoundError(layout_path)
    labels_dir = args.layout_dir / "labels_npy"
    if not labels_dir.is_dir():
        raise FileNotFoundError(labels_dir)
    if not args.chord_input_metadata.is_file():
        raise FileNotFoundError(args.chord_input_metadata)

    faces = face_record_map(load_json(layout_path))
    candidates = candidate_map(load_json(args.chord_input_metadata))
    materials = 0
    stems: set[str] = set()
    for face, record in faces.items():
        label_path = labels_dir / f"{face}.npy"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        for material in record["materials"]:
            materials += 1
            stem = str(material["chosen_stem"])
            stems.add(stem)
            if stem not in candidates:
                raise KeyError(f"{face}: chosen_stem {stem!r} is absent from candidate metadata")
            source_dir = args.pbr_chord_output_dir / stem
            for channel in CHANNELS:
                path = source_dir / f"{channel}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as image:
                    if image.size != (512, 512):
                        raise ValueError(f"{path}: expected 512x512, got {image.size}")

    print(
        json.dumps(
            {
                "status": "ok",
                "faces": len(faces),
                "materials": materials,
                "unique_chosen_stems": len(stems),
                "chord_channels": list(CHANNELS),
                "chord_shape": [512, 512],
                "traceback_selection_modified": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

