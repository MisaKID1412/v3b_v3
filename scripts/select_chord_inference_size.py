#!/usr/bin/env python3
"""Choose the accepted CHORD inference profile from source-image resolution."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise RuntimeError(f"no images under {args.image_dir}")
    maximum = 0
    for path in paths:
        with Image.open(path) as image:
            maximum = max(maximum, image.width, image.height)
    # Low-resolution perspective sets (for example Structure3D's 512px views)
    # stay native. High-resolution captures use the accepted 2048 inference
    # profile; CHORD maps are then restored to the unchanged 512px input shape.
    print(512 if maximum <= 768 else 2048)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
