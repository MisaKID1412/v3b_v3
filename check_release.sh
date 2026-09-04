#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

bash -n run_from_images.sh
bash -n restart_fullface_native_pipeline/scripts/run_new_dataset_full_pipeline_2080.sh
python3 -m unittest discover -s tests -p 'test_package_integrity.py' -v

echo "[v3b_v3] source-integrity checks passed"
