#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$PROJECT_DIR/config/v3b.env}"

# The documented resume/stage controls may be supplied by the caller for a
# one-off continuation. Preserve those values across `source`, even when the
# saved config contains its normal full-run defaults.
V3B_CALLER_RESUME_SET="${RESUME+x}"
V3B_CALLER_RESUME="${RESUME-}"
V3B_CALLER_RUN_FROM_SET="${RUN_FROM+x}"
V3B_CALLER_RUN_FROM="${RUN_FROM-}"
V3B_CALLER_RUN_UNTIL_SET="${RUN_UNTIL+x}"
V3B_CALLER_RUN_UNTIL="${RUN_UNTIL-}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[v3b_v3] missing config: $CONFIG_FILE" >&2
  echo "[v3b_v3] copy config/v3b.env.example to config/v3b.env and edit the paths" >&2
  exit 2
fi

set -a
source "$CONFIG_FILE"
set +a

if [[ -n "$V3B_CALLER_RESUME_SET" ]]; then RESUME="$V3B_CALLER_RESUME"; fi
if [[ -n "$V3B_CALLER_RUN_FROM_SET" ]]; then RUN_FROM="$V3B_CALLER_RUN_FROM"; fi
if [[ -n "$V3B_CALLER_RUN_UNTIL_SET" ]]; then RUN_UNTIL="$V3B_CALLER_RUN_UNTIL"; fi

: "${DATASET_DIR:?set DATASET_DIR in the config}"
IMAGE_DIR="${IMAGE_DIR:-$DATASET_DIR/input_images}"
RUN_NAME="${RUN_NAME:-v3b_v3_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/outputs/$RUN_NAME}"
GPU_ID="${GPU_ID:-0}"
RESUME="${RESUME:-0}"
RUN_FROM="${RUN_FROM:-frontend_reconstruction}"
RUN_UNTIL="${RUN_UNTIL:-unitypackage}"
FRONTEND_FORCE_STAGE="${FRONTEND_FORCE_STAGE:-0}"
ATLAS_RESOLUTION_SCALE="${ATLAS_RESOLUTION_SCALE:-2.0}"
MATERIAL_MAX_PER_FACE="${MATERIAL_MAX_PER_FACE:-8}"
MATERIAL_CLUSTER_COMPONENTS="${MATERIAL_CLUSTER_COMPONENTS:-4}"
MATERIAL_CLUSTER_MIN_FRACTION="${MATERIAL_CLUSTER_MIN_FRACTION:-0.01}"
MATERIAL_CLUSTER_MIN_REGION_SIZE="${MATERIAL_CLUSTER_MIN_REGION_SIZE:-32}"
MATSEG_PY="${MATSEG_PY:-$SD_PY}"
IOPAINT_BIN="${IOPAINT_BIN:-iopaint}"
IOPAINT_MODEL_DIR="${IOPAINT_MODEL_DIR:-$PROJECT_DIR/models/iopaint}"
IOPAINT_MODEL="${IOPAINT_MODEL:-lama}"
IOPAINT_DEVICE="${IOPAINT_DEVICE:-cuda}"

if [[ -z "${CAMERA_METADATA_JSON:-}" && -f "$DATASET_DIR/camera_metadata.json" ]]; then
  CAMERA_METADATA_JSON="$DATASET_DIR/camera_metadata.json"
fi
if [[ -z "${EXISTING_DA3_NPZ:-}" && -f "$DATASET_DIR/existing_da3/results.npz" ]]; then
  EXISTING_DA3_NPZ="$DATASET_DIR/existing_da3/results.npz"
  EXISTING_DA3_SCENE_GLB="$DATASET_DIR/existing_da3/scene.glb"
  EXISTING_DA3_CAMERA_POSES="$DATASET_DIR/existing_da3/camera_poses.json"
fi

DA3_DIR="${DA3_DIR:-$RUN_ROOT/da3_large11_full160}"
if [[ -n "${CAMERA_METADATA_JSON:-}" ]]; then
  DA3_SOURCE_DIR="${DA3_SOURCE_DIR:-$RUN_ROOT/da3_numeric_before_known_camera_alignment}"
else
  DA3_SOURCE_DIR="${DA3_SOURCE_DIR:-$DA3_DIR}"
fi
ROOMFORMER_SEARCH_DIR="${ROOMFORMER_SEARCH_DIR:-$RUN_ROOT/roomformer_da3_corner_search}"
PROVISIONAL_SOURCE_DIR="${PROVISIONAL_SOURCE_DIR:-$RUN_ROOT/source_package_provisional}"
VIEW_FACE_DIR="${VIEW_FACE_DIR:-$RUN_ROOT/da3_depth_face_masks_tol055}"
SAM3_MASK_DIR="${SAM3_MASK_DIR:-$RUN_ROOT/sam3_view_masks_v5}"
STRICT_VIEW_MASK_DIR="${STRICT_VIEW_MASK_DIR:-$RUN_ROOT/merged_strict_view_reject_masks_v5}"
REFIT_DIR="${REFIT_DIR:-$RUN_ROOT/observed_refit_structure}"
SOURCE_PACKAGE_DIR="${SOURCE_PACKAGE_DIR:-$RUN_ROOT/source_package}"
STRICT_PROJECTION_DIR="${STRICT_PROJECTION_DIR:-$RUN_ROOT/source_projected_strict}"
COMPLETED_OBSERVED_DIR="${COMPLETED_OBSERVED_DIR:-$RUN_ROOT/completed_observed_lama}"
ORIGINAL_CANDIDATE_DIR="${ORIGINAL_CANDIDATE_DIR:-$RUN_ROOT/view_contributor_regions_original}"
UNIFIED_REGION_DIR="${UNIFIED_REGION_DIR:-$RUN_ROOT/unified_region_proposals}"
UNIFIED_FRONTEND_DIR="${UNIFIED_FRONTEND_DIR:-$RUN_ROOT/unified_material_frontend}"
MATERIAL_PACKAGE_DIR="$UNIFIED_FRONTEND_DIR/final_trace/package"
SELECTED_METADATA="$UNIFIED_FRONTEND_DIR/final_trace/metadata_final_traceback_inputs.json"
TRACE_LOG="$UNIFIED_FRONTEND_DIR/final_trace/material_level_traceback.json"
IDENTITY_CONTRACT="$UNIFIED_FRONTEND_DIR/identity/identity_receipt.json"
CHORD_OUTPUT_DIR="${CHORD_OUTPUT_DIR:-$RUN_ROOT/chord_pbr_same_config}"
CHORD_PENDING_DIR="${CHORD_PENDING_DIR:-$RUN_ROOT/chord_pending}"
MATERIAL_LAYOUT_DIR="${MATERIAL_LAYOUT_DIR:-$RUN_ROOT/material_layout_axis_n}"
REFINED_LAYOUT_DIR="${REFINED_LAYOUT_DIR:-$RUN_ROOT/material_layout_structured_boundaries}"
PBR_OUTPUT_DIR="${PBR_OUTPUT_DIR:-$RUN_ROOT/pbr_full_normalized}"
GEOMETRY_UNITY_DIR="${GEOMETRY_UNITY_DIR:-$RUN_ROOT/unity_geometry}"
UNITY_PROJECT_DIR="${UNITY_PROJECT_DIR:-$RUN_ROOT/unity_project}"
UNITY_PACKAGE_FILE="${UNITY_PACKAGE_FILE:-$RUN_ROOT/v3b_v3.unitypackage}"

stages=(frontend_reconstruction unified_proposals material_identity_traceback chord_pbr material_layout territory_pbr unity_project unitypackage)

stage_index() {
  local target="$1" i
  for i in "${!stages[@]}"; do
    if [[ "${stages[$i]}" == "$target" ]]; then echo "$i"; return 0; fi
  done
  echo "[v3b_v3] unknown stage: $target" >&2
  exit 2
}

from_i="$(stage_index "$RUN_FROM")"
until_i="$(stage_index "$RUN_UNTIL")"
if (( from_i > until_i )); then echo "[v3b_v3] RUN_FROM must not follow RUN_UNTIL" >&2; exit 2; fi
should_run() { local i; i="$(stage_index "$1")"; (( i >= from_i && i <= until_i )); }
marker() { echo "$RUN_ROOT/.v3b_v3_stage_$1.done"; }
stage_done() { [[ -f "$(marker "$1")" ]]; }
mark_done() { date -Is > "$(marker "$1")"; }
require_file() { [[ -f "$1" ]] || { echo "[v3b_v3] missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "[v3b_v3] missing directory: $1" >&2; exit 2; }; }
require_executable() {
  if [[ -x "$1" ]]; then return 0; fi
  command -v "$1" >/dev/null 2>&1 || { echo "[v3b_v3] missing executable: $1" >&2; exit 2; }
}

if [[ -e "$RUN_ROOT" && "$RESUME" != "1" ]]; then
  echo "[v3b_v3] output already exists: $RUN_ROOT" >&2
  echo "[v3b_v3] set RESUME=1 to continue it, or choose another RUN_NAME" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"
touch "$RUN_ROOT/.v3b_v3_initialized"
exec > >(tee -a "$RUN_ROOT/v3b_v3_pipeline.log") 2>&1

echo "[v3b_v3] raw-images-to-Unity pipeline"
echo "[v3b_v3] images=$IMAGE_DIR"
echo "[v3b_v3] run_root=$RUN_ROOT"
echo "[v3b_v3] stages=$RUN_FROM..$RUN_UNTIL"

require_dir "$IMAGE_DIR"
image_count="$(find "$IMAGE_DIR" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')"
if [[ "$image_count" -eq 0 ]]; then echo "[v3b_v3] no input images found under $IMAGE_DIR" >&2; exit 2; fi
for executable in "$DA3_PY" "$ROOMFORMER_PY" "$SAM3_PY" "$SD_PY" "$MATSEG_PY" "$CHORD_PY"; do require_executable "$executable"; done
require_dir "$DA3_MODEL_DIR"
require_dir "$ROOMFORMER_DIR"
require_file "$ROOMFORMER_CKPT_TIGHT"
require_file "$ROOMFORMER_CKPT_BASE"
require_dir "$MATSEG_VENDOR_DIR"
require_file "$MATSEG_CHECKPOINT"
require_dir "$CHORD_REPO"
require_file "$CHORD_CKPT"
require_file "$CHORD_CONFIG"
require_executable "$IOPAINT_BIN"
require_dir "$IOPAINT_MODEL_DIR"
if [[ -n "${SAM3_CHECKPOINT_PATH:-}" ]]; then require_file "$SAM3_CHECKPOINT_PATH"; fi
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then echo "[v3b_v3] preflight passed; no inference was run"; exit 0; fi

export PROJECT_DIR DATASET_DIR IMAGE_DIR RUN_NAME RUN_ROOT GPU_ID
export DA3_DIR DA3_SOURCE_DIR ROOMFORMER_SEARCH_DIR PROVISIONAL_SOURCE_DIR VIEW_FACE_DIR
export SAM3_MASK_DIR STRICT_VIEW_MASK_DIR REFIT_DIR SOURCE_PACKAGE_DIR STRICT_PROJECTION_DIR
export COMPLETED_OBSERVED_DIR CAMERA_METADATA_JSON
export EXISTING_DA3_NPZ="${EXISTING_DA3_NPZ:-}"
export EXISTING_DA3_SCENE_GLB="${EXISTING_DA3_SCENE_GLB:-}"
export EXISTING_DA3_CAMERA_POSES="${EXISTING_DA3_CAMERA_POSES:-}"
export CHORD_SOURCE_DIR="$STRICT_PROJECTION_DIR" CHORD_DATASET_DIR="$DATASET_DIR"
export CHORD_DA3_DIR="$DA3_DIR" CHORD_OBJECT_MASK_DIR="$STRICT_VIEW_MASK_DIR"
export CANDIDATE_CHORD_INPUT_SIZE=512 CANDIDATE_CHORD_SIZE=512
export CHORD_LOCAL_SCRIPT="$PROJECT_DIR/scripts/run_chord_local_inference.py"
export PYTHONPATH="$PROJECT_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"

if should_run frontend_reconstruction && ! stage_done frontend_reconstruction; then
  echo "[stage 1/8] reconstruct room, reject objects, project strict observations"
  CHORD_CANDIDATE_DIR="$ORIGINAL_CANDIDATE_DIR" RUN_FROM="${FRONTEND_RUN_FROM:-preflight}" RUN_UNTIL=chord_inputs FORCE_STAGE="$FRONTEND_FORCE_STAGE" \
    bash "$PROJECT_DIR/restart_fullface_native_pipeline/scripts/run_new_dataset_full_pipeline_2080.sh"
  require_file "$ORIGINAL_CANDIDATE_DIR/metadata_view_contributor_chord_inputs.json"
  mark_done frontend_reconstruction
fi

if should_run unified_proposals && ! stage_done unified_proposals; then
  echo "[stage 2/8] generate one common set of spatial material proposals"
  require_file "$ORIGINAL_CANDIDATE_DIR/metadata_view_contributor_chord_inputs.json"
  "$MATSEG_PY" "$PROJECT_DIR/scripts/frontend_v3/prepare_unified_region_proposals.py" \
    --source-metadata "$ORIGINAL_CANDIDATE_DIR/metadata_view_contributor_chord_inputs.json" \
    --generator-script "$PROJECT_DIR/scripts/frontend_v3/generate_chord_view_contributor_region_priors.py" \
    --out-dir "$UNIFIED_REGION_DIR" --material-max-per-face "$MATERIAL_MAX_PER_FACE" \
    --material-cluster-components "$MATERIAL_CLUSTER_COMPONENTS" \
    --material-cluster-min-fraction "$MATERIAL_CLUSTER_MIN_FRACTION" \
    --material-cluster-min-region-size "$MATERIAL_CLUSTER_MIN_REGION_SIZE"
  require_file "$UNIFIED_REGION_DIR/metadata_view_contributor_chord_inputs.json"
  mark_done unified_proposals
fi

if should_run material_identity_traceback && ! stage_done material_identity_traceback; then
  echo "[stage 3/8] MatSeg identity only, support recovery, native-scale guard, v3b trace-back"
  "$MATSEG_PY" "$PROJECT_DIR/scripts/run_unified_frontend_v3.py" \
    --metadata "$UNIFIED_REGION_DIR/metadata_view_contributor_chord_inputs.json" \
    --region-assets-dir "$UNIFIED_REGION_DIR" --vendor-dir "$MATSEG_VENDOR_DIR" \
    --checkpoint "$MATSEG_CHECKPOINT" --output-dir "$UNIFIED_FRONTEND_DIR" --device "cuda:$GPU_ID"
  require_file "$SELECTED_METADATA"
  require_file "$TRACE_LOG"
  require_file "$IDENTITY_CONTRACT"
  mark_done material_identity_traceback
fi

if should_run chord_pbr && ! stage_done chord_pbr; then
  echo "[stage 4/8] run one unchanged CHORD configuration for every trace-back input"
  require_dir "$MATERIAL_PACKAGE_DIR/chord_inputs"
  cache_args=()
  if [[ -n "${CHORD_CACHE_ROOTS:-}" ]]; then
    IFS=':' read -r -a cache_roots <<< "$CHORD_CACHE_ROOTS"
    for cache_root in "${cache_roots[@]}"; do cache_args+=(--cache-root "$cache_root"); done
  fi
  "$SD_PY" "$PROJECT_DIR/scripts/materialize_chord_pbr_cache_by_input_hash.py" \
    --input-dir "$MATERIAL_PACKAGE_DIR/chord_inputs" --output-dir "$CHORD_OUTPUT_DIR" \
    --pending-dir "$CHORD_PENDING_DIR" "${cache_args[@]}" --receipt "$RUN_ROOT/chord_cache_receipt.json"
  if find "$CHORD_PENDING_DIR" -maxdepth 1 -type f -name '*.png' | grep -q .; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}" \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    "$CHORD_PY" "$PROJECT_DIR/scripts/run_chord_local_inference.py" \
      --chord-repo "$CHORD_REPO" --ckpt "$CHORD_CKPT" --input-dir "$CHORD_PENDING_DIR" \
      --output-dir "$CHORD_OUTPUT_DIR" --config-path "$CHORD_CONFIG" --device cuda --size 0 --pad-multiple 32
  fi
  "$SD_PY" "$PROJECT_DIR/scripts/materialize_chord_pbr_cache_by_input_hash.py" \
    --input-dir "$MATERIAL_PACKAGE_DIR/chord_inputs" --output-dir "$CHORD_OUTPUT_DIR" \
    --pending-dir "$CHORD_PENDING_DIR" --receipt "$RUN_ROOT/chord_cache_receipt.json"
  if find "$CHORD_PENDING_DIR" -maxdepth 1 -type f -name '*.png' | grep -q .; then echo "[v3b_v3] CHORD left incomplete materials" >&2; exit 2; fi
  mark_done chord_pbr
fi

if should_run material_layout && ! stage_done material_layout; then
  echo "[stage 5/8] compose traced materials and infer generalized territories"
  "$SD_PY" "$PROJECT_DIR/scripts/compose_unified_trace_chord_materials.py" \
    --metadata "$SELECTED_METADATA" --package-dir "$MATERIAL_PACKAGE_DIR" \
    --chord-output-dir "$CHORD_OUTPUT_DIR" --generator "$PROJECT_DIR/scripts/frontend_v3/generate_chord_view_contributor_region_priors.py"
  "$SD_PY" "$PROJECT_DIR/scripts/adapt_unified_material_evidence_for_v1_backend.py" \
    --locked-metadata "$SELECTED_METADATA" --package-dir "$MATERIAL_PACKAGE_DIR" --chord-output-dir "$CHORD_OUTPUT_DIR"
  "$SD_PY" "$PROJECT_DIR/scripts/write_unified_backend_freeze_manifest.py" \
    --trace-metadata "$SELECTED_METADATA" --material-dir "$MATERIAL_PACKAGE_DIR" \
    --completed-observed-dir "$COMPLETED_OBSERVED_DIR" --experiment-root "$RUN_ROOT" --output "$RUN_ROOT/freeze_manifest.json"
  "$SD_PY" "$PROJECT_DIR/scripts/compose_material_base_atlas_v1.py" \
    --freeze-manifest "$RUN_ROOT/freeze_manifest.json" --out-dir "$MATERIAL_LAYOUT_DIR" \
    --completed-observed-dir "$COMPLETED_OBSERVED_DIR" --observed-confidence 0.58 \
    --observed-margin 0.12 --placement-min-reliability 0.55 --color-calibration-strength 0.0 \
    --axis-boundary-min-accuracy 0.84 --axis-boundary-min-class-accuracy 0.68 \
    --axis-layer-candidates --axis-layer-min-strict-fraction 0.003 \
    --axis-layer-energy-tie-margin 0.01 \
    --linear-min-accuracy 0.91 --linear-min-class-accuracy 0.84 --linear-min-tangent-span 0.5 \
    --linear-min-strict-fraction 0.055 --axis-curve-max-deviation-frac 0.075 \
    --axis-curve-smooth-frac 0.03 --axis-curve-max-tangent-samples 640 \
    --axis-curve-tangent-edge-ignore-frac 0.12 --data-energy-weight 1.0 \
    --seed-mismatch-weight 5.0 --boundary-complexity-weight 0.025 --soft-material-blend \
    --soft-probability-sigma 5.5 --soft-probability-power 0.82 --soft-confidence-margin 0.55 \
    --soft-boundary-radius-frac 0.08 --soft-region-blur-frac 0.02 \
    --soft-weight-source target_reconstruction --target-reconstruction-blur-sigma 5.5 \
    --target-reconstruction-temperature 0.55 --target-reconstruction-label-prior 0.08 \
    --target-reconstruction-smooth-sigma 1.4 --target-reconstruction-min-weight 0.24 \
    --target-reconstruction-weight-power 1.45 --label-soft-mix-base 0.18 \
    --label-soft-mix-lowconf 0.62 --soft-boundary-only --soft-boundary-width-frac 0.038 \
    --pairwise-boundary-blend --pairwise-target-mix 0.55 --no-use-discovered-material-masks \
    --allow-neutral-ceiling-wall-fallback --merge-small-neutral-ceiling-shading-clusters --no-target-lowfreq-transfer
  require_file "$MATERIAL_LAYOUT_DIR/metadata_material_placement.json"
  mark_done material_layout
fi

if should_run territory_pbr && ! stage_done territory_pbr; then
  echo "[stage 6/8] regularize supported boundaries and synthesize whole-territory PBR"
  "$SD_PY" "$PROJECT_DIR/scripts/refine_structured_material_territories_v2.py" \
    --layout-dir "$MATERIAL_LAYOUT_DIR" --source-package-dir "$SOURCE_PACKAGE_DIR" \
    --out-dir "$REFINED_LAYOUT_DIR" --wall-horizontal-min-tangent-span 0.07
  if [[ -f "$PBR_OUTPUT_DIR/metadata_adaptive_whole_territory_pbr.json" ]]; then
    echo "[stage 6/8] reuse completed whole-territory PBR synthesis; finish pending audits"
  else
    "$SD_PY" "$PROJECT_DIR/scripts/synthesize_adaptive_whole_territory_pbr_generalized_mirror_guard.py" \
      --chord-input-metadata "$MATERIAL_PACKAGE_DIR/metadata_view_contributor_chord_inputs.json" \
      --pbr-chord-output-dir "$CHORD_OUTPUT_DIR" --layout-dir "$REFINED_LAYOUT_DIR" \
      --out-dir "$PBR_OUTPUT_DIR" --atlas-resolution-scale "$ATLAS_RESOLUTION_SCALE" \
      --chord-output-support-mode full_normalized
  fi
  require_file "$PBR_OUTPUT_DIR/pbr_textures/basecolor/floor.png"
  require_file "$PBR_OUTPUT_DIR/pbr_textures/normal/floor.png"
  require_file "$PBR_OUTPUT_DIR/pbr_textures/roughness/floor.png"
  require_file "$PBR_OUTPUT_DIR/pbr_textures/metallic/floor.png"
  cp "$PBR_OUTPUT_DIR/metadata_adaptive_whole_territory_pbr.json" "$PBR_OUTPUT_DIR/metadata_pbr_placement.json"
  "$SD_PY" "$PROJECT_DIR/scripts/audit_scale_and_lattice.py" --run-dir "$PBR_OUTPUT_DIR" --out-json "$PBR_OUTPUT_DIR/audit_scale_and_lattice.json"
  "$SD_PY" "$PROJECT_DIR/scripts/audit_wholefield_pbr.py" --run-dir "$PBR_OUTPUT_DIR" --out-json "$PBR_OUTPUT_DIR/audit_wholefield_pbr.json"
  mark_done territory_pbr
fi

if should_run unity_project && ! stage_done unity_project; then
  echo "[stage 7/8] export OBJ, full PBR textures, and Unity import settings"
  "$SD_PY" "$PROJECT_DIR/scripts/export_unity_from_source_package.py" \
    --source-package-dir "$SOURCE_PACKAGE_DIR" --texture-dir "$PBR_OUTPUT_DIR/pbr_textures/basecolor" \
    --out-dir "$GEOMETRY_UNITY_DIR" --target-height-m "${TARGET_UNITY_HEIGHT_M:-2.7}" --version-name v3b_v3_image_to_unity
  "$SD_PY" "$PROJECT_DIR/scripts/export_unity_aligned_pbr.py" \
    --geometry-unity-dir "$GEOMETRY_UNITY_DIR" --pbr-run-dir "$PBR_OUTPUT_DIR" \
    --editor-script "$PROJECT_DIR/unity/Editor/V3bV3PBRSetup.cs" --out-dir "$UNITY_PROJECT_DIR"
  require_file "$UNITY_PROJECT_DIR/room.obj"
  require_dir "$UNITY_PROJECT_DIR/pbr_textures"
  mark_done unity_project
fi

if should_run unitypackage && ! stage_done unitypackage; then
  echo "[stage 8/8] build importable Unity package and provenance manifest"
  "$SD_PY" "$PROJECT_DIR/scripts/build_unitypackage.py" --source-dir "$UNITY_PROJECT_DIR" \
    --out-file "$UNITY_PACKAGE_FILE" --asset-root Assets/v3b_v3
  "$SD_PY" "$PROJECT_DIR/scripts/write_pipeline_manifest.py" --run-root "$RUN_ROOT" --image-dir "$IMAGE_DIR" \
    --identity-contract "$IDENTITY_CONTRACT" --trace-log "$TRACE_LOG" --layout-dir "$REFINED_LAYOUT_DIR" \
    --pbr-dir "$PBR_OUTPUT_DIR" --unity-dir "$UNITY_PROJECT_DIR" --unitypackage "$UNITY_PACKAGE_FILE"
  mark_done unitypackage
fi

echo "[v3b_v3] complete through stage: $RUN_UNTIL"
if [[ -d "$UNITY_PROJECT_DIR" ]]; then echo "[v3b_v3] Unity project: $UNITY_PROJECT_DIR"; fi
if [[ -f "$UNITY_PACKAGE_FILE" ]]; then echo "[v3b_v3] Unity package: $UNITY_PACKAGE_FILE"; fi
