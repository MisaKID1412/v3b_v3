#!/usr/bin/env bash
set -euo pipefail

# Internal staged raw-image frontend used by v3b_v3. The public entry point is
# run_from_images.sh; this file preserves the accepted reconstruction settings.
#
# The public runner calls it through raw reconstruction and initial trace-back
# proposal discovery, then continues with the unified v3 frontend and backend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:?set DATASET_DIR through config/v3b.env}"
IMAGE_DIR="${IMAGE_DIR:-$DATASET_DIR/input_images}"
COLMAP_MODEL_DIR="${COLMAP_MODEL_DIR:-}"

RUN_NAME="${RUN_NAME:-restart_fullface_native_raw_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/outputs/restart_fullface_native_pipeline/$RUN_NAME}"
RUN_FROM="${RUN_FROM:-preflight}"
RUN_UNTIL="${RUN_UNTIL:-final_export}"
FORCE_STAGE="${FORCE_STAGE:-0}"
case "$RUN_UNTIL" in
  preflight|da3|structure_search|provisional_masks|refit_source|strict_projection|completed_observed|chord_inputs|candidate_chord|chord_compose|material_layout) ;;
  *)
    echo "[frontend] this packaged internal runner stops at material_layout; use run_from_images.sh for PBR and Unity" >&2
    exit 2
    ;;
esac
STRICT_V3B="${STRICT_V3B:-0}"
STRICT_V3B_MATERIAL_PROVENANCE="${STRICT_V3B_MATERIAL_PROVENANCE:-0}"
ROOM_CANDIDATE_SOURCE_POLICY="${ROOM_CANDIDATE_SOURCE_POLICY:-all}"
if [[ "$STRICT_V3B" == "1" ]]; then
  STRICT_V3B_MATERIAL_PROVENANCE=1
  ROOM_CANDIDATE_SOURCE_POLICY=roomformer_only
  ROOM_TARGET_CORNERS=0
fi
STRICT_MATERIAL_ARGS=()
if [[ "$STRICT_V3B_MATERIAL_PROVENANCE" == "1" ]]; then
  STRICT_MATERIAL_ARGS+=(--strict-v3b-material-provenance)
fi

DA3_MODEL_DIR="${DA3_MODEL_DIR:?set DA3_MODEL_DIR through config/v3b.env}"
ROOMFORMER_DIR="${ROOMFORMER_DIR:?set ROOMFORMER_DIR through config/v3b.env}"
ROOMFORMER_CKPT_TIGHT="${ROOMFORMER_CKPT_TIGHT:?set ROOMFORMER_CKPT_TIGHT through config/v3b.env}"
ROOMFORMER_CKPT_BASE="${ROOMFORMER_CKPT_BASE:?set ROOMFORMER_CKPT_BASE through config/v3b.env}"

DA3_PY="${DA3_PY:?set DA3_PY through config/v3b.env}"
ROOMFORMER_PY="${ROOMFORMER_PY:?set ROOMFORMER_PY through config/v3b.env}"
SAM3_PY="${SAM3_PY:?set SAM3_PY through config/v3b.env}"
SD_PY="${SD_PY:?set SD_PY through config/v3b.env}"
CHORD_PY="${CHORD_PY:?set CHORD_PY through config/v3b.env}"
IOPAINT_BIN="${IOPAINT_BIN:-iopaint}"
IOPAINT_MODEL_DIR="${IOPAINT_MODEL_DIR:-$PROJECT_DIR/models/iopaint}"
IOPAINT_MODEL="${IOPAINT_MODEL:-lama}"
IOPAINT_DEVICE="${IOPAINT_DEVICE:-cuda}"

DA3_PROCESS_RES="${DA3_PROCESS_RES:-504}"
DA3_REF_VIEW_STRATEGY="${DA3_REF_VIEW_STRATEGY:-saddle_balanced}"
DA3_NUM_MAX_POINTS="${DA3_NUM_MAX_POINTS:-3000000}"
DA3_SEED="${DA3_SEED:-20260626}"
EXISTING_DA3_NPZ="${EXISTING_DA3_NPZ:-}"
EXISTING_DA3_SCENE_GLB="${EXISTING_DA3_SCENE_GLB:-}"
EXISTING_DA3_CAMERA_POSES="${EXISTING_DA3_CAMERA_POSES:-}"
EXISTING_DA3_MODEL_NAME="${EXISTING_DA3_MODEL_NAME:-existing_da3_export}"
PRECOMPUTED_DA3_OUTPUT_DIR="${PRECOMPUTED_DA3_OUTPUT_DIR:-}"
CAMERA_METADATA_JSON="${CAMERA_METADATA_JSON:-}"
TEXTURE_PPM="${TEXTURE_PPM:-900}"
TARGET_UNITY_HEIGHT_M="${TARGET_UNITY_HEIGHT_M:-2.7}"
GPU_ID="${GPU_ID:-0}"
ROOM_MIN_CORNERS="${ROOM_MIN_CORNERS:-4}"
ROOM_MAX_CORNERS="${ROOM_MAX_CORNERS:-20}"
ROOM_TARGET_CORNERS="${ROOM_TARGET_CORNERS:-0}"
ROOM_AXIS_WEIGHT="${ROOM_AXIS_WEIGHT:-6.5}"
ROOM_RADIAL_MAX_BOUNDS_EXPANSION="${ROOM_RADIAL_MAX_BOUNDS_EXPANSION:-0.65}"
ROOM_RADIAL_STEP_MAX_DEPTH_FRAC="${ROOM_RADIAL_STEP_MAX_DEPTH_FRAC:-0.75}"
ROOM_WALL_EDGE_SUPPORT_TARGET="${ROOM_WALL_EDGE_SUPPORT_TARGET:-0.32}"
ROOM_WALL_EDGE_SUPPORT_HARD_MIN="${ROOM_WALL_EDGE_SUPPORT_HARD_MIN:-0.05}"
if [[ -n "$CAMERA_METADATA_JSON" ]]; then
  ROOM_DISABLE_YAW_ALIGN="${ROOM_DISABLE_YAW_ALIGN:-1}"
else
  ROOM_DISABLE_YAW_ALIGN="${ROOM_DISABLE_YAW_ALIGN:-0}"
fi
MATERIAL_MAX_PER_FACE="${MATERIAL_MAX_PER_FACE:-8}"
# Match the accepted v3b discovery stage. This is the number of initial
# appearance components, not a fixed final material count; the final count is
# still selected from the data and capped by MATERIAL_MAX_PER_FACE.
MATERIAL_CLUSTER_COMPONENTS="${MATERIAL_CLUSTER_COMPONENTS:-8}"
MATERIAL_CLUSTER_MIN_FRACTION="${MATERIAL_CLUSTER_MIN_FRACTION:-0.02}"
MATERIAL_CLUSTER_MIN_REGION_SIZE="${MATERIAL_CLUSTER_MIN_REGION_SIZE:-32}"
MATERIAL_CLUSTER_CHROMA_MERGE_THRESHOLD="${MATERIAL_CLUSTER_CHROMA_MERGE_THRESHOLD:-0.55}"
MATERIAL_WALL_BAND_MAX_HEIGHT_FRAC="${MATERIAL_WALL_BAND_MAX_HEIGHT_FRAC:-0.08}"
MATERIAL_WALL_BAND_MIN_TEXTURE_DELTA="${MATERIAL_WALL_BAND_MIN_TEXTURE_DELTA:-0.30}"
SURFACE_NORMAL_MIN_COS="${SURFACE_NORMAL_MIN_COS:-0.60}"
SURFACE_DISTANCE_CLEAN_TOL="${SURFACE_DISTANCE_CLEAN_TOL:-0.0}"
HORIZONTAL_PLANE_NORMAL_MIN_COS="${HORIZONTAL_PLANE_NORMAL_MIN_COS:-0.90}"
MATERIAL_MIN_VIEW_MASK_PIXELS="${MATERIAL_MIN_VIEW_MASK_PIXELS:-1800}"
MATERIAL_MIN_RECTIFIED_VALID_FRAC="${MATERIAL_MIN_RECTIFIED_VALID_FRAC:-0.42}"
MATERIAL_RECTIFIED_INNER_MIN_SIZE="${MATERIAL_RECTIFIED_INNER_MIN_SIZE:-128}"
MATERIAL_RECTIFIED_INNER_MIN_VALID_FRAC="${MATERIAL_RECTIFIED_INNER_MIN_VALID_FRAC:-0.95}"
MATERIAL_RECTIFIED_INNER_MIN_SAFE_FRAC="${MATERIAL_RECTIFIED_INNER_MIN_SAFE_FRAC:-0.90}"
MATERIAL_RECTIFIED_SEARCH_BOX_SCALE="${MATERIAL_RECTIFIED_SEARCH_BOX_SCALE:-1.0}"
MATERIAL_RECTIFIED_SEARCH_BOX_MIN_SIZE="${MATERIAL_RECTIFIED_SEARCH_BOX_MIN_SIZE:-0}"
MATERIAL_FLOOR_RECTIFIED_SEARCH_BOX_SCALE="${MATERIAL_FLOOR_RECTIFIED_SEARCH_BOX_SCALE:-2.0}"
MATERIAL_FLOOR_RECTIFIED_SEARCH_BOX_MIN_SIZE="${MATERIAL_FLOOR_RECTIFIED_SEARCH_BOX_MIN_SIZE:-1024}"
MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_SIZE="${MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_SIZE:-32}"
MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_VALID_FRAC="${MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_VALID_FRAC:-0.95}"
MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_SAFE_FRAC="${MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_SAFE_FRAC:-0.90}"
MATERIAL_MIN_FINAL_SOURCE_UNIQUE_PIXELS="${MATERIAL_MIN_FINAL_SOURCE_UNIQUE_PIXELS:-0}"
MATERIAL_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX="${MATERIAL_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX:-0}"
# Structure3D RGB views are only 512 px wide.  Keep the source-resolution gate,
# but calibrate its low-resolution floor branch to accept one fully valid,
# non-quilted rectified patch with roughly a 20 px native short side. The retry
# still has to traceback to original pixels and pass every validity/safety gate.
MATERIAL_FLOOR_MIN_FINAL_SOURCE_UNIQUE_PIXELS="${MATERIAL_FLOOR_MIN_FINAL_SOURCE_UNIQUE_PIXELS:-300}"
MATERIAL_FLOOR_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX="${MATERIAL_FLOOR_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX:-20}"
MATERIAL_FLOOR_LOWRES_SOURCE_ADAPTATION="${MATERIAL_FLOOR_LOWRES_SOURCE_ADAPTATION:-1}"
MATERIAL_FLOOR_LOWRES_RETRY_SOURCE_UNIQUE_PIXELS="${MATERIAL_FLOOR_LOWRES_RETRY_SOURCE_UNIQUE_PIXELS:-1800}"
MATERIAL_FLOOR_LOWRES_RETRY_SOURCE_BBOX_SHORT_SIDE_PX="${MATERIAL_FLOOR_LOWRES_RETRY_SOURCE_BBOX_SHORT_SIDE_PX:-56}"
MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_SIZE="${MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_SIZE:-256}"
MATERIAL_FLOOR_LOWRES_RECTIFIED_MAX_SIDE_FRAC="${MATERIAL_FLOOR_LOWRES_RECTIFIED_MAX_SIDE_FRAC:-0.98}"
# A Structure3D perspective can expose only a narrow floor strip.  This gate
# only admits that strip to the low-resolution retry; the retry must still form
# one fully valid rectangle, trace back to enough original pixels, and pass the
# source short-side and planar-colour checks below.
MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_VALID_FRAC="${MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_VALID_FRAC:-0.03}"
MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_SAFE_FRAC="${MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_SAFE_FRAC:-0.03}"
MATERIAL_FLOOR_LOWRES_PLANAR_COLOR_DELTA="${MATERIAL_FLOOR_LOWRES_PLANAR_COLOR_DELTA:-1.80}"
# Keep the normal v3b multi-view rule.  If a low-resolution wall has no normal
# support, allow one clean Atlas observation solely for original-view traceback.
MATERIAL_WALL_LOWRES_SINGLE_VIEW_SOURCE_ADAPTATION="${MATERIAL_WALL_LOWRES_SINGLE_VIEW_SOURCE_ADAPTATION:-1}"
MATERIAL_WALL_LOWRES_ADAPT_TRIGGER_FINAL_KEEP_FRAC="${MATERIAL_WALL_LOWRES_ADAPT_TRIGGER_FINAL_KEEP_FRAC:-0.006}"
MATERIAL_WALL_LOWRES_ADAPT_MIN_SUPPORT_FRAC="${MATERIAL_WALL_LOWRES_ADAPT_MIN_SUPPORT_FRAC:-0.006}"
MATERIAL_WALL_LOWRES_ADAPT_CLEAN_THRESH="${MATERIAL_WALL_LOWRES_ADAPT_CLEAN_THRESH:-0.48}"
MATERIAL_WALL_LOWRES_ADAPT_MIN_VALID_VIEWS="${MATERIAL_WALL_LOWRES_ADAPT_MIN_VALID_VIEWS:-1}"
MATERIAL_WALL_LOWRES_ADAPT_OBJECT_RISK_THRESH="${MATERIAL_WALL_LOWRES_ADAPT_OBJECT_RISK_THRESH:-0.05}"
MATERIAL_WALL_LOWRES_ADAPT_BOUNDARY_TRUST_THRESH="${MATERIAL_WALL_LOWRES_ADAPT_BOUNDARY_TRUST_THRESH:-0.55}"
MATERIAL_WALL_LOWRES_ADAPT_FOOTPRINT_MIN="${MATERIAL_WALL_LOWRES_ADAPT_FOOTPRINT_MIN:-0.008}"
MATERIAL_WALL_LOWRES_MIN_RECTIFIED_VALID_FRAC="${MATERIAL_WALL_LOWRES_MIN_RECTIFIED_VALID_FRAC:-0.012}"
MATERIAL_WALL_LOWRES_RECTIFIED_MIN_SIZE="${MATERIAL_WALL_LOWRES_RECTIFIED_MIN_SIZE:-16}"
MATERIAL_WALL_LOWRES_MIN_FINAL_SOURCE_UNIQUE_PIXELS="${MATERIAL_WALL_LOWRES_MIN_FINAL_SOURCE_UNIQUE_PIXELS:-128}"
MATERIAL_WALL_LOWRES_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX="${MATERIAL_WALL_LOWRES_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX:-8}"
MATERIAL_WALL_LOWRES_RECTIFIED_SEARCH_BOX_SCALE="${MATERIAL_WALL_LOWRES_RECTIFIED_SEARCH_BOX_SCALE:-2.0}"
MATERIAL_WALL_LOWRES_RECTIFIED_SEARCH_BOX_MIN_SIZE="${MATERIAL_WALL_LOWRES_RECTIFIED_SEARCH_BOX_MIN_SIZE:-1024}"
MATERIAL_WALL_LOWRES_PLANAR_COLOR_DELTA="${MATERIAL_WALL_LOWRES_PLANAR_COLOR_DELTA:-1.60}"
MATERIAL_WALL_LOWRES_ADAPT_FLAG="--no-wall-lowres-single-view-source-adaptation"
if [[ "$MATERIAL_WALL_LOWRES_SINGLE_VIEW_SOURCE_ADAPTATION" == "1" ]]; then
  MATERIAL_WALL_LOWRES_ADAPT_FLAG="--wall-lowres-single-view-source-adaptation"
fi
MATERIAL_THIN_TERRITORY_MIN_SPAN_FRAC="${MATERIAL_THIN_TERRITORY_MIN_SPAN_FRAC:-0.30}"
MATERIAL_THIN_TERRITORY_MAX_THICKNESS_FRAC="${MATERIAL_THIN_TERRITORY_MAX_THICKNESS_FRAC:-0.10}"
MATERIAL_THIN_TERRITORY_MIN_SOURCE_UNIQUE_PIXELS="${MATERIAL_THIN_TERRITORY_MIN_SOURCE_UNIQUE_PIXELS:-128}"
MATERIAL_THIN_TERRITORY_MIN_SOURCE_BBOX_SHORT_SIDE_PX="${MATERIAL_THIN_TERRITORY_MIN_SOURCE_BBOX_SHORT_SIDE_PX:-4}"
MATERIAL_THIN_TERRITORY_MIN_RECTIFIED_VALID_FRAC="${MATERIAL_THIN_TERRITORY_MIN_RECTIFIED_VALID_FRAC:-0.08}"
MATERIAL_THIN_TERRITORY_RECTIFIED_SEARCH_MIN_SIZE="${MATERIAL_THIN_TERRITORY_RECTIFIED_SEARCH_MIN_SIZE:-512}"
MATERIAL_FINAL_SOURCE_RESOLUTION_SCORE_WEIGHT="${MATERIAL_FINAL_SOURCE_RESOLUTION_SCORE_WEIGHT:-0.35}"
MATERIAL_FINAL_SOURCE_RESOLUTION_REFERENCE_SIDE_PX="${MATERIAL_FINAL_SOURCE_RESOLUTION_REFERENCE_SIDE_PX:-96}"

DA3_DIR="${DA3_DIR:-$RUN_ROOT/da3_large11_full160}"
if [[ -n "$CAMERA_METADATA_JSON" ]]; then
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
STRICT_PROJECTION_DIR="${STRICT_PROJECTION_DIR:-$RUN_ROOT/source_projected_da3hf_surface055_v5reject_strict_empty_v1}"
COMPLETED_OBSERVED_DIR="${COMPLETED_OBSERVED_DIR:-$RUN_ROOT/completed_observed_lama_v1_legacyraw}"
CHORD_CANDIDATE_DIR="${CHORD_CANDIDATE_DIR:-$RUN_ROOT/view_contributor_chord_v3n_multi_exemplar_materials}"
CHORD_SOURCE_DIR="${CHORD_SOURCE_DIR:-$STRICT_PROJECTION_DIR}"
CHORD_DATASET_DIR="${CHORD_DATASET_DIR:-$DATASET_DIR}"
CHORD_DA3_DIR="${CHORD_DA3_DIR:-$DA3_DIR}"
CHORD_OBJECT_MASK_DIR="${CHORD_OBJECT_MASK_DIR:-$STRICT_VIEW_MASK_DIR}"
CANDIDATE_CHORD_INPUT_SIZE="${CANDIDATE_CHORD_INPUT_SIZE:-512}"
CANDIDATE_CHORD_SIZE="${CANDIDATE_CHORD_SIZE:-512}"
CANDIDATE_CHORD_OUTPUT_DIR="${CANDIDATE_CHORD_OUTPUT_DIR:-$RUN_ROOT/chord_outputs_${CANDIDATE_CHORD_SIZE}_candidates}"
FULLFACE_CHORD_SIZE="${FULLFACE_CHORD_SIZE:-512}"
MATERIAL_LAYOUT_DIR="${MATERIAL_LAYOUT_DIR:-$RUN_ROOT/material_placement_v11_legacyraw_originalweights}"
NOTILE_DIR="${NOTILE_DIR:-$RUN_ROOT/material_placement_v11_legacyraw_nontile_2048_cleanquilt_keepdir_v2}"
UNITY_NOTILE_DIR="${UNITY_NOTILE_DIR:-$RUN_ROOT/room2048_legacyraw_nontile_clean_unity}"
FULLFACE_CHORD_INPUT_DIR="${FULLFACE_CHORD_INPUT_DIR:-$RUN_ROOT/fullface_chord_inputs_legacyraw}"
FULLFACE_CHORD_OUTPUT_DIR="${FULLFACE_CHORD_OUTPUT_DIR:-$RUN_ROOT/fullface_chord_outputs_native_basecolor_legacyraw}"
UNITY_OUT_DIR="${UNITY_OUT_DIR:-$RUN_ROOT/room_fullface_chord_native_unity_legacyraw}"

CHORD_REPO="${CHORD_REPO:?set CHORD_REPO through config/v3b.env}"
CHORD_CKPT="${CHORD_CKPT:-$PROJECT_DIR/pipeline/models/ubisoft-laforge-chord/chord_v1.safetensors}"
CHORD_CONFIG="${CHORD_CONFIG:-$CHORD_REPO/config/chord.yaml}"
CHORD_LOCAL_SCRIPT="${CHORD_LOCAL_SCRIPT:-$PROJECT_DIR/scripts/run_chord_local_inference.py}"
RUN_CANDIDATE_CHORD_ON_2080="${RUN_CANDIDATE_CHORD_ON_2080:-0}"
RUN_FULLFACE_CHORD_ON_2080="${RUN_FULLFACE_CHORD_ON_2080:-0}"

cd "$PROJECT_DIR"
mkdir -p "$RUN_ROOT"
if [[ -z "$COLMAP_MODEL_DIR" ]]; then
  # DA3 hf-aligned poses do not read a COLMAP reconstruction. Some shared CLIs
  # still require the argument, so use an explicit empty placeholder.
  COLMAP_MODEL_DIR="$RUN_ROOT/unused_colmap_model_for_da3_pose"
  mkdir -p "$COLMAP_MODEL_DIR"
fi

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_DIR/.cache}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_DIR/.cache/pip}"
export TMPDIR="${TMPDIR:-$PROJECT_DIR/tmp}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

stages=(
  preflight
  da3
  structure_search
  provisional_masks
  refit_source
  strict_projection
  completed_observed
  chord_inputs
  candidate_chord
  chord_compose
  material_layout
  nontile_atlas
  notile_unity
  fullface_inputs
  fullface_chord
  final_export
)

stage_index() {
  local target="$1"
  local i
  for i in "${!stages[@]}"; do
    if [[ "${stages[$i]}" == "$target" ]]; then
      echo "$i"
      return 0
    fi
  done
  echo "[error] unknown stage: $target" >&2
  exit 2
}

from_i="$(stage_index "$RUN_FROM")"
until_i="$(stage_index "$RUN_UNTIL")"

should_run() {
  local idx
  idx="$(stage_index "$1")"
  [[ "$idx" -ge "$from_i" && "$idx" -le "$until_i" ]]
}

done_marker() {
  echo "$RUN_ROOT/.stage_$1.done"
}

already_done() {
  [[ "$FORCE_STAGE" != "1" && -f "$(done_marker "$1")" ]]
}

mark_done() {
  date -Is > "$(done_marker "$1")"
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "[error] missing required path: $1" >&2
    exit 2
  fi
}

write_run_manifest() {
  "$SD_PY" - "$RUN_ROOT/pipeline_run_manifest.json" <<PY
import json, os, sys
manifest = {
    "method": "restart_fullface_native_raw_images_to_obj_staged_v1",
    "project_dir": os.environ.get("PROJECT_DIR", "$PROJECT_DIR"),
    "dataset_dir": "$DATASET_DIR",
    "image_dir": "$IMAGE_DIR",
    "run_root": "$RUN_ROOT",
    "da3_dir": "$DA3_DIR",
    "da3_source_dir": "$DA3_SOURCE_DIR",
    "camera_metadata_json": "$CAMERA_METADATA_JSON" or None,
    "existing_da3_source": {
        "npz": "$EXISTING_DA3_NPZ" or None,
        "scene_glb": "$EXISTING_DA3_SCENE_GLB" or None,
        "camera_poses": "$EXISTING_DA3_CAMERA_POSES" or None,
        "model_name": "$EXISTING_DA3_MODEL_NAME" if "$EXISTING_DA3_NPZ" else None,
        "precomputed_output_dir": "$PRECOMPUTED_DA3_OUTPUT_DIR" or None,
    },
    "source_package_dir": "$SOURCE_PACKAGE_DIR",
    "strict_projection_dir": "$STRICT_PROJECTION_DIR",
    "chord_candidate_dir": "$CHORD_CANDIDATE_DIR",
    "candidate_chord_output_dir": "$CANDIDATE_CHORD_OUTPUT_DIR",
    "material_layout_dir": "$MATERIAL_LAYOUT_DIR",
    "notile_dir": "$NOTILE_DIR",
    "fullface_chord_input_dir": "$FULLFACE_CHORD_INPUT_DIR",
    "fullface_chord_output_dir": "$FULLFACE_CHORD_OUTPUT_DIR",
    "unity_out_dir": "$UNITY_OUT_DIR",
    "room_corner_policy": {
        "min_corners": int("$ROOM_MIN_CORNERS"),
        "max_corners": int("$ROOM_MAX_CORNERS"),
        "target_corners": int("$ROOM_TARGET_CORNERS"),
        "disable_yaw_alignment": bool(int("$ROOM_DISABLE_YAW_ALIGN")),
        "axis_weight": float("$ROOM_AXIS_WEIGHT"),
        "candidate_source_policy": "$ROOM_CANDIDATE_SOURCE_POLICY",
    },
    "material_discovery": {
        "max_per_face": int("$MATERIAL_MAX_PER_FACE"),
        "cluster_components": int("$MATERIAL_CLUSTER_COMPONENTS"),
        "min_fraction": float("$MATERIAL_CLUSTER_MIN_FRACTION"),
        "min_region_size": int("$MATERIAL_CLUSTER_MIN_REGION_SIZE"),
        "chroma_merge_threshold": float("$MATERIAL_CLUSTER_CHROMA_MERGE_THRESHOLD"),
        "wall_band_max_height_frac": float("$MATERIAL_WALL_BAND_MAX_HEIGHT_FRAC"),
        "wall_band_min_texture_delta": float("$MATERIAL_WALL_BAND_MIN_TEXTURE_DELTA"),
        "thin_territory_min_span_frac": float("$MATERIAL_THIN_TERRITORY_MIN_SPAN_FRAC"),
        "thin_territory_max_thickness_frac": float("$MATERIAL_THIN_TERRITORY_MAX_THICKNESS_FRAC"),
        "thin_territory_min_source_unique_pixels": int("$MATERIAL_THIN_TERRITORY_MIN_SOURCE_UNIQUE_PIXELS"),
        "thin_territory_min_source_bbox_short_side_px": float("$MATERIAL_THIN_TERRITORY_MIN_SOURCE_BBOX_SHORT_SIDE_PX"),
        "thin_territory_min_rectified_valid_frac": float("$MATERIAL_THIN_TERRITORY_MIN_RECTIFIED_VALID_FRAC"),
        "thin_territory_rectified_search_min_size": int("$MATERIAL_THIN_TERRITORY_RECTIFIED_SEARCH_MIN_SIZE"),
        "min_view_mask_pixels": int("$MATERIAL_MIN_VIEW_MASK_PIXELS"),
        "min_rectified_valid_frac": float("$MATERIAL_MIN_RECTIFIED_VALID_FRAC"),
        "rectified_inner_min_size": int("$MATERIAL_RECTIFIED_INNER_MIN_SIZE"),
        "rectified_inner_min_valid_frac": float("$MATERIAL_RECTIFIED_INNER_MIN_VALID_FRAC"),
        "rectified_inner_min_safe_frac": float("$MATERIAL_RECTIFIED_INNER_MIN_SAFE_FRAC"),
        "floor_rectified_search_box_scale": float("$MATERIAL_FLOOR_RECTIFIED_SEARCH_BOX_SCALE"),
        "floor_rectified_search_box_min_size": int("$MATERIAL_FLOOR_RECTIFIED_SEARCH_BOX_MIN_SIZE"),
        "rectified_inner_fallback_min_size": int("$MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_SIZE"),
        "rectified_inner_fallback_min_valid_frac": float("$MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_VALID_FRAC"),
        "rectified_inner_fallback_min_safe_frac": float("$MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_SAFE_FRAC"),
        "floor_min_final_source_unique_pixels": int("$MATERIAL_FLOOR_MIN_FINAL_SOURCE_UNIQUE_PIXELS"),
        "floor_min_final_source_bbox_short_side_px": float("$MATERIAL_FLOOR_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX"),
        "floor_lowres_source_adaptation": bool(int("$MATERIAL_FLOOR_LOWRES_SOURCE_ADAPTATION")),
        "floor_lowres_retry_source_unique_pixels": int("$MATERIAL_FLOOR_LOWRES_RETRY_SOURCE_UNIQUE_PIXELS"),
        "floor_lowres_retry_source_bbox_short_side_px": float("$MATERIAL_FLOOR_LOWRES_RETRY_SOURCE_BBOX_SHORT_SIDE_PX"),
        "floor_lowres_rectified_min_valid_frac": float("$MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_VALID_FRAC"),
        "floor_lowres_rectified_min_safe_frac": float("$MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_SAFE_FRAC"),
        "include_atlas_fallback": False,
    },
    "surface_distance_tol": 0.055,
    "surface_distance_clean_tol": float("$SURFACE_DISTANCE_CLEAN_TOL") if float("$SURFACE_DISTANCE_CLEAN_TOL") > 0.0 else 0.055,
    "surface_distance_hard_gate": True,
    "strict_empty_low_quality": True,
    "strict_projection_evidence": {
        "nearest_visible_enabled": bool(int("$STRICT_V3B")),
        "nearest_visible_sidecar": (
            "$STRICT_PROJECTION_DIR/nearest_visible_evidence.json"
            if int("$STRICT_V3B") else None
        ),
        "selection_policy": (
            "min_camera_distance_then_max_projection_weight"
            if int("$STRICT_V3B") else None
        ),
    },
    "chord_candidate_input_size": int("$CANDIDATE_CHORD_INPUT_SIZE"),
    "chord_candidate_size": int("$CANDIDATE_CHORD_SIZE"),
    "fullface_chord_size": int("$FULLFACE_CHORD_SIZE"),
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(manifest, indent=2, ensure_ascii=False))
PY
}

faces_csv() {
  "$SD_PY" - "$SOURCE_PACKAGE_DIR/metadata.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(f["face"] for f in d["faces"]))
PY
}

faces_array_from_csv() {
  local csv="$1"
  read -r -a face_args <<< "$(printf '%s' "$csv" | tr ',' ' ')"
}

write_freeze_manifest() {
  local manifest="$RUN_ROOT/freeze_manifest_current_paths.json"
  "$SD_PY" - "$manifest" <<PY
import json, sys
data = {
  "status": "generated_for_restart_fullface_native_pipeline",
  "server_project_root": "$PROJECT_DIR",
  "experiment_root": "$RUN_ROOT",
  "dataset_dir": "$DATASET_DIR",
  "da3_dir": "$DA3_DIR",
  "colmap_model_dir": "$COLMAP_MODEL_DIR",
  "mesh_source_dir": "$SOURCE_PACKAGE_DIR",
  "strict_observed_projection_dir": "$STRICT_PROJECTION_DIR",
  "completed_observed_lama_dir": "$COMPLETED_OBSERVED_DIR",
  "completed_observed_dir": "$COMPLETED_OBSERVED_DIR/completed_observed",
  "completed_observed_weight_dir": "$COMPLETED_OBSERVED_DIR/weights",
  "chord_material_dir": "$CHORD_CANDIDATE_DIR",
  "mesh_manifest": "$SOURCE_PACKAGE_DIR/manifest.json",
  "mesh_obj": "$SOURCE_PACKAGE_DIR/room_empty.obj",
  "projection_metadata": "$STRICT_PROJECTION_DIR/metadata.json",
  "chord_inputs_metadata": "$CHORD_CANDIDATE_DIR/metadata_view_contributor_chord_inputs.json",
  "chord_materials_metadata": "$CHORD_CANDIDATE_DIR/metadata_view_contributor_chord_materials.json",
  "completed_observed_lama_metadata": "$COMPLETED_OBSERVED_DIR/metadata_completed_observed_lama.json",
  "accepted_material_base_atlas_dir": "$MATERIAL_LAYOUT_DIR",
  "accepted_material_base_atlas_metadata": "$MATERIAL_LAYOUT_DIR/metadata_material_placement.json",
  "frozen_projection_gates": {
    "surface_distance_tol": 0.055,
    "surface_distance_clean_tol": float("$SURFACE_DISTANCE_CLEAN_TOL") if float("$SURFACE_DISTANCE_CLEAN_TOL") > 0.0 else 0.055,
    "surface_distance_hard_gate": True,
    "object_reject_version": "v5reject_strict_empty_v1"
  }
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(data, indent=2, ensure_ascii=False))
PY
  echo "$manifest"
}

echo "[pipeline] project=$PROJECT_DIR"
echo "[pipeline] dataset=$DATASET_DIR"
echo "[pipeline] images=$IMAGE_DIR"
echo "[pipeline] run_root=$RUN_ROOT"
echo "[pipeline] run_from=$RUN_FROM run_until=$RUN_UNTIL"
write_run_manifest

if should_run preflight; then
  if ! already_done preflight; then
    require_path "$IMAGE_DIR"
    require_path "$COLMAP_MODEL_DIR"
    if [[ -n "$PRECOMPUTED_DA3_OUTPUT_DIR" ]]; then
      require_path "$PRECOMPUTED_DA3_OUTPUT_DIR"
    elif [[ -n "$EXISTING_DA3_NPZ" ]]; then
      require_path "$EXISTING_DA3_NPZ"
      require_path "$EXISTING_DA3_SCENE_GLB"
      require_path "$EXISTING_DA3_CAMERA_POSES"
      require_path "scripts/prepare_existing_da3_output.py"
    else
      require_path "$DA3_MODEL_DIR"
    fi
    if [[ -n "$CAMERA_METADATA_JSON" ]]; then
      require_path "$CAMERA_METADATA_JSON"
      require_path "scripts/align_da3_to_known_camera_metadata.py"
    fi
    require_path "$ROOMFORMER_DIR"
    require_path "$ROOMFORMER_CKPT_TIGHT"
    require_path "$ROOMFORMER_CKPT_BASE"
    require_path "$DA3_PY"
    require_path "$ROOMFORMER_PY"
    require_path "$SAM3_PY"
    require_path "$SD_PY"
    if [[ -n "${SAM3_CHECKPOINT_PATH:-}" ]]; then
      require_path "$SAM3_CHECKPOINT_PATH"
    fi
    image_count="$(find "$IMAGE_DIR" -maxdepth 1 \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')"
    if [[ "$image_count" -eq 0 ]]; then
      echo "[error] no images under $IMAGE_DIR" >&2
      exit 2
    fi
    echo "[preflight] image_count=$image_count"
    mark_done preflight
  fi
fi

if should_run da3; then
  if ! already_done da3; then
    if [[ -n "$PRECOMPUTED_DA3_OUTPUT_DIR" ]]; then
      if [[ "$STRICT_V3B" != "1" ]]; then
        echo "[error] PRECOMPUTED_DA3_OUTPUT_DIR is accepted only by the audited strict profile" >&2
        exit 2
      fi
      for required_da3_file in depth.npy conf.npy extrinsics.npy intrinsics.npy meta.json scene.glb; do
        require_path "$PRECOMPUTED_DA3_OUTPUT_DIR/$required_da3_file"
      done
      rm -rf "$DA3_SOURCE_DIR"
      mkdir -p "$DA3_SOURCE_DIR"
      cp -a "$PRECOMPUTED_DA3_OUTPUT_DIR/." "$DA3_SOURCE_DIR/"
      "$SD_PY" - "$PRECOMPUTED_DA3_OUTPUT_DIR" "$DA3_SOURCE_DIR" "$RUN_ROOT/da3_precomputed_transfer_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
copied = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3])
required = ("depth.npy", "conf.npy", "extrinsics.npy", "intrinsics.npy", "meta.json", "scene.glb")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

records = []
for name in required:
    source_path = source / name
    copied_path = copied / name
    source_hash = sha256(source_path)
    copied_hash = sha256(copied_path)
    if source_hash != copied_hash:
        raise RuntimeError(f"DA3 transfer hash mismatch: {name}")
    records.append({
        "name": name,
        "source": str(source_path),
        "copied": str(copied_path),
        "size_bytes": source_path.stat().st_size,
        "sha256": source_hash,
    })
manifest_path.write_text(json.dumps({
    "method": "strict_precomputed_da3_byte_exact_transfer_v1",
    "source_dir": str(source),
    "copied_dir": str(copied),
    "files": records,
}, indent=2) + "\n", encoding="utf-8")
PY
    elif [[ -n "$EXISTING_DA3_NPZ" ]]; then
      "$SD_PY" scripts/prepare_existing_da3_output.py \
        --npz "$EXISTING_DA3_NPZ" \
        --scene-glb "$EXISTING_DA3_SCENE_GLB" \
        --camera-poses-json "$EXISTING_DA3_CAMERA_POSES" \
        --model-name "$EXISTING_DA3_MODEL_NAME" \
        --out-dir "$DA3_SOURCE_DIR"
    else
      export CUDA_VISIBLE_DEVICES="$GPU_ID"
      "$DA3_PY" scripts/run_da3_numeric_inference.py \
        --image-dir "$IMAGE_DIR" \
        --out-dir "$DA3_SOURCE_DIR" \
        --model-dir "$DA3_MODEL_DIR" \
        --local-files-only \
        --process-res "$DA3_PROCESS_RES" \
        --process-res-method upper_bound_resize \
        --ref-view-strategy "$DA3_REF_VIEW_STRATEGY" \
        --num-max-points "$DA3_NUM_MAX_POINTS" \
        --seed "$DA3_SEED" \
        --export-format glb \
        --device cuda
    fi
    if [[ -n "$CAMERA_METADATA_JSON" ]]; then
      "$SD_PY" scripts/align_da3_to_known_camera_metadata.py \
        --da3-dir "$DA3_SOURCE_DIR" \
        --camera-metadata-json "$CAMERA_METADATA_JSON" \
        --image-dir "$IMAGE_DIR" \
        --out-dir "$DA3_DIR" \
        --max-points "$DA3_NUM_MAX_POINTS" \
        --seed "$DA3_SEED"
    fi
    require_path "$DA3_DIR/depth.npy"
    require_path "$DA3_DIR/conf.npy"
    require_path "$DA3_DIR/extrinsics.npy"
    require_path "$DA3_DIR/intrinsics.npy"
    require_path "$DA3_DIR/meta.json"
    require_path "$DA3_DIR/scene.glb"
    mark_done da3
  fi
fi

if should_run structure_search; then
  if ! already_done structure_search; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    room_pose_args=()
    if [[ "$ROOM_DISABLE_YAW_ALIGN" == "1" ]]; then
      room_pose_args+=(--disable-yaw-align)
    fi
    if [[ -n "$CAMERA_METADATA_JSON" ]]; then
      # The known-camera adapter defines a y-up DA3 world.  Preserve that
      # trusted gravity instead of re-estimating it from monocular depth.
      room_pose_args+=(--forced-up-axis-world 0 1 0)
    fi
    if [[ "${ROOM_OPENING_AWARE_ENCLOSURE_OVERRIDE:-0}" == "1" ]]; then
      room_pose_args+=(
        --opening-aware-enclosure-override
        --wall-enclosure-override-min-support
        "${ROOM_WALL_ENCLOSURE_OVERRIDE_MIN_SUPPORT:-0.55}"
        --wall-enclosure-override-mean-support
        "${ROOM_WALL_ENCLOSURE_OVERRIDE_MEAN_SUPPORT:-0.75}"
      )
    fi
    "$ROOMFORMER_PY" scripts/run_roomformer_da3_corner_search.py \
      --roomformer-dir "$ROOMFORMER_DIR" \
      --scene-glb "$DA3_DIR/scene.glb" \
      --out-dir "$ROOMFORMER_SEARCH_DIR" \
      --checkpoints "$ROOMFORMER_CKPT_TIGHT" "$ROOMFORMER_CKPT_BASE" \
      --min-corners "$ROOM_MIN_CORNERS" \
      --max-corners "$ROOM_MAX_CORNERS" \
      --target-corners "$ROOM_TARGET_CORNERS" \
      --candidate-source-policy "$ROOM_CANDIDATE_SOURCE_POLICY" \
      --axis-weight "$ROOM_AXIS_WEIGHT" \
      --radial-max-bounds-expansion "$ROOM_RADIAL_MAX_BOUNDS_EXPANSION" \
      --radial-step-max-depth-frac "$ROOM_RADIAL_STEP_MAX_DEPTH_FRAC" \
      --wall-edge-support-target "$ROOM_WALL_EDGE_SUPPORT_TARGET" \
      --wall-edge-support-hard-min "$ROOM_WALL_EDGE_SUPPORT_HARD_MIN" \
      "${room_pose_args[@]}" \
      --device cuda
    require_path "$ROOMFORMER_SEARCH_DIR/structure_roomformer_da3_polygon.json"
    "$SD_PY" scripts/create_polygon_source_from_structure_json.py \
      --structure-json "$ROOMFORMER_SEARCH_DIR/structure_roomformer_da3_polygon.json" \
      --out-dir "$PROVISIONAL_SOURCE_DIR" \
      --scene-name room_empty \
      --texture-ppm "$TEXTURE_PPM" \
      --max-texture-size 4096 \
      --min-texture-size 512 \
      --min-wall-texture-width "${MIN_WALL_TEXTURE_WIDTH:-64}" \
      --copy-debug-image "$ROOMFORMER_SEARCH_DIR/corner_search_world_summary.png"
    mark_done structure_search
  fi
fi

if should_run provisional_masks; then
  if ! already_done provisional_masks; then
    faces="$("$SD_PY" - "$PROVISIONAL_SOURCE_DIR/metadata.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join(f["face"] for f in d["faces"]))
PY
)"
    faces_array_from_csv "$faces"
    "$SD_PY" scripts/generate_polygon_view_face_masks_from_da3_depth.py \
      --dataset-dir "$DATASET_DIR" \
      --source-dir "$PROVISIONAL_SOURCE_DIR" \
      --da3-dir "$DA3_DIR" \
      --out-dir "$VIEW_FACE_DIR" \
      --faces "$faces" \
      --coordinate-space hfalign \
      --surface-distance-tol 0.055 \
      --min-conf 1.0 \
      --close-px 2 \
      --dilate-px 1 \
      --calibrate-depth-to-polygon-zbuffer \
      --require-depth-calibration \
      --zbuffer-stride 2

    checkpoint_args=()
    if [[ -n "${SAM3_CHECKPOINT_PATH:-}" && -f "${SAM3_CHECKPOINT_PATH:-}" ]]; then
      checkpoint_args=(--sam3-checkpoint-path "$SAM3_CHECKPOINT_PATH")
    fi
    cutout_prompt_args=()
    if [[ "${SAM3_DISABLE_CUTOUT_PROMPTS:-0}" == "1" ]]; then
      cutout_prompt_args=(--disable-cutout-prompts)
    fi
    object_prompt_args=()
    if [[ "${SAM3_DISABLE_OBJECT_PROMPTS:-0}" == "1" ]]; then
      object_prompt_args=(--disable-object-prompts)
    fi
    surface_prompt_args=()
    if [[ "${SAM3_DISABLE_SURFACE_PROMPTS:-0}" == "1" ]]; then
      surface_prompt_args=(--disable-surface-prompts)
    fi
    # Large/floor-to-ceiling windows can legitimately occupy most of a
    # Structure3D perspective. The relaxed area cap below applies only to
    # explicit SAM3 door/window cutout prompts; movable objects retain the
    # stricter object cap.
    "$SAM3_PY" scripts/generate_polygon_sam3_view_masks.py \
      --dataset-dir "$DATASET_DIR" \
      --source-dir "$PROVISIONAL_SOURCE_DIR" \
      --colmap-model-dir "$COLMAP_MODEL_DIR" \
      --out-dir "$SAM3_MASK_DIR" \
      --view-face-mask-dir "$VIEW_FACE_DIR/view_face_masks" \
      --faces "$faces" \
      --device cuda \
      --score-thresh "${SAM3_SCORE_THRESH:-0.10}" \
      --object-score-thresh "${SAM3_OBJECT_SCORE_THRESH:-0.12}" \
      --surface-score-thresh "${SAM3_SURFACE_SCORE_THRESH:-0.12}" \
      --cutout-score-thresh "${SAM3_CUTOUT_SCORE_THRESH:-0.22}" \
      --min-mask-area "${SAM3_MIN_MASK_AREA:-48}" \
      --max-mask-area-ratio "${SAM3_MAX_MASK_AREA_RATIO:-0.56}" \
      --object-max-mask-area-ratio "${SAM3_OBJECT_MAX_MASK_AREA_RATIO:-0.42}" \
      --surface-max-mask-area-ratio "${SAM3_SURFACE_MAX_MASK_AREA_RATIO:-0.88}" \
      --cutout-max-mask-area-ratio "${SAM3_CUTOUT_MAX_MASK_AREA_RATIO:-0.85}" \
      --min-structure-overlap "${SAM3_MIN_STRUCTURE_OVERLAP:-0.10}" \
      --surface-min-structure-overlap "${SAM3_SURFACE_MIN_STRUCTURE_OVERLAP:-0.04}" \
      --cutout-min-structure-overlap "${SAM3_CUTOUT_MIN_STRUCTURE_OVERLAP:-0.18}" \
      --min-face-overlap "${SAM3_MIN_FACE_OVERLAP:-0.05}" \
      --surface-min-face-overlap "${SAM3_SURFACE_MIN_FACE_OVERLAP:-0.02}" \
      --cutout-min-face-overlap "${SAM3_CUTOUT_MIN_FACE_OVERLAP:-0.08}" \
      --dilate-object-px "${SAM3_DILATE_OBJECT_PX:-4}" \
      --surface-color-mode "${SAM3_SURFACE_COLOR_MODE:-dark_green_board}" \
      --surface-color-close-px "${SAM3_SURFACE_COLOR_CLOSE_PX:-7}" \
      --surface-color-dilate-px "${SAM3_SURFACE_COLOR_DILATE_PX:-5}" \
      "${object_prompt_args[@]}" \
      "${surface_prompt_args[@]}" \
      --object-prompt "desk" --object-prompt "table" --object-prompt "chair" \
      --object-prompt "cabinet" --object-prompt "filing cabinet" --object-prompt "drawer cabinet" \
      --object-prompt "bookshelf" --object-prompt "bookcase" --object-prompt "shelf" \
      --object-prompt "computer monitor" --object-prompt "keyboard" --object-prompt "mouse" \
      --object-prompt "printer" --object-prompt "office printer" --object-prompt "photocopier" \
      --object-prompt "air conditioner" --object-prompt "heater" --object-prompt "radiator" \
      --object-prompt "trash bin" --object-prompt "box" \
      --object-prompt "sofa" --object-prompt "couch" --object-prompt "armchair" \
      --object-prompt "area rug" --object-prompt "floor rug" --object-prompt "rug" \
      --object-prompt "television" --object-prompt "flat-screen television" \
      --object-prompt "television screen" --object-prompt "TV screen" --object-prompt "screen displaying image" \
      --object-prompt "potted plant" --object-prompt "indoor plant" \
      --object-prompt "curtain" --object-prompt "drape" \
      --object-prompt "floor lamp" --object-prompt "table lamp" \
      --object-prompt "ceiling light" --object-prompt "ceiling lamp" \
      --object-prompt "light fixture" --object-prompt "chandelier" --object-prompt "pendant light" \
      --object-prompt "curtain rod" --object-prompt "curtain track" --object-prompt "curtain rail" \
      --object-prompt "picture frame" --object-prompt "wall art" \
      --object-prompt "bed" --object-prompt "wardrobe" --object-prompt "nightstand" \
      --surface-prompt "blackboard" --surface-prompt "chalkboard" --surface-prompt "green board" \
      --surface-prompt "large green chalkboard" --surface-prompt "writing board" \
      "${cutout_prompt_args[@]}" \
      "${checkpoint_args[@]}"

    "$SD_PY" scripts/merge_polygon_sam3_view_masks_strict.py \
      --input-dir "$SAM3_MASK_DIR" \
      --out-dir "$STRICT_VIEW_MASK_DIR" \
      --close-px 2 \
      --dilate-px 2 \
      --min-area 24
    mark_done provisional_masks
  fi
fi

if should_run refit_source; then
  if ! already_done refit_source; then
    refit_generalization_args=(
      --normal-guided-vertical-bounds
      --horizontal-normal-min-cos "$HORIZONTAL_PLANE_NORMAL_MIN_COS"
      --max-local-wall-shift-ratio "${MAX_LOCAL_WALL_SHIFT_RATIO:-0.12}"
    )
    if [[ -n "$CAMERA_METADATA_JSON" ]]; then
      # Same-center calibrated camera metadata already fixes gravity and gives
      # RoomFormer a stable vertical proportion. Monocular horizontal-normal
      # modes can mistake a table-height plane for the floor; retain the seed
      # bounds and let the normal robust refit make only its bounded update.
      refit_generalization_args=(
        --no-normal-guided-vertical-bounds
        --max-local-wall-shift-ratio "${MAX_LOCAL_WALL_SHIFT_RATIO:-0.12}"
      )
    fi
    if [[ "$STRICT_V3B" == "1" ]]; then
      refit_generalization_args=(
        --no-normal-guided-vertical-bounds
        --max-local-wall-shift-ratio 0
      )
    fi
    "$SD_PY" scripts/refine_manhattan_l_structure_from_da3_observed_surfaces.py \
      --dataset-dir "$DATASET_DIR" \
      --da3-dir "$DA3_DIR" \
      --structure-json "$ROOMFORMER_SEARCH_DIR/structure_roomformer_da3_polygon.json" \
      --reject-mask-dir "$STRICT_VIEW_MASK_DIR" \
      --out-dir "$REFIT_DIR" \
      "${refit_generalization_args[@]}"
    require_path "$REFIT_DIR/structure_da3_manhattan_polygon_observed_refit.json"
    "$SD_PY" scripts/create_polygon_source_from_structure_json.py \
      --structure-json "$REFIT_DIR/structure_da3_manhattan_polygon_observed_refit.json" \
      --out-dir "$SOURCE_PACKAGE_DIR" \
      --scene-name room_empty \
      --texture-ppm "$TEXTURE_PPM" \
      --max-texture-size 4096 \
      --min-texture-size 512 \
      --min-wall-texture-width "${MIN_WALL_TEXTURE_WIDTH:-64}" \
      --copy-debug-image "$REFIT_DIR/structure_da3_manhattan_polygon_observed_refit_comparison.png"
    mark_done refit_source
  fi
fi

if should_run strict_projection; then
  if ! already_done strict_projection; then
    faces="$(faces_csv)"
    faces_array_from_csv "$faces"
    strict_projection_evidence_args=()
    if [[ "$STRICT_V3B" == "1" ]]; then
      strict_projection_evidence_args=(--emit-nearest-visible-evidence)
    fi
    "$SD_PY" scripts/build_polygon_photo_source_from_colmap.py \
      --dataset-dir "$DATASET_DIR" \
      --polygon-source-dir "$SOURCE_PACKAGE_DIR" \
      --colmap-model-dir "$COLMAP_MODEL_DIR" \
      --pose-source da3_hfalign \
      --da3-dir "$DA3_DIR" \
      --out-dir "$STRICT_PROJECTION_DIR" \
      --faces "$faces" \
      --views-per-face 0 \
      --depth-abs-tol 0.045 \
      --depth-rel-tol 0.035 \
      --distance-weight-scale 1.15 \
      --distance-weight-power 1.15 \
      --mask-boundary-safe-px 18 \
      --mask-boundary-power 1.15 \
      --min-mask-boundary-trust 0.55 \
      --object-risk-hard-thresh 0.05 \
      --footprint-min-area 0.32 \
      --footprint-power 0.85 \
      --adaptive-short-face-footprint \
      --short-face-length-median-frac "${SHORT_FACE_LENGTH_MEDIAN_FRAC:-0.50}" \
      --short-face-footprint-median-multiplier "${SHORT_FACE_FOOTPRINT_MEDIAN_MULTIPLIER:-3.50}" \
      --short-face-footprint-min-area "${SHORT_FACE_FOOTPRINT_MIN_AREA:-0.008}" \
      --adaptive-horizontal-footprint \
      --horizontal-footprint-median-multiplier "${HORIZONTAL_FOOTPRINT_MEDIAN_MULTIPLIER:-1.25}" \
      --horizontal-footprint-min-area "${HORIZONTAL_FOOTPRINT_MIN_AREA:-0.008}" \
      --surface-distance-tol 0.055 \
      --surface-distance-clean-tol "$SURFACE_DISTANCE_CLEAN_TOL" \
      --surface-distance-power 1.0 \
      --surface-distance-hard-gate \
      --surface-normal-min-cos "$SURFACE_NORMAL_MIN_COS" \
      --object-mask-dir "$STRICT_VIEW_MASK_DIR" \
      --object-mask-dilate-px 2 \
      --object-risk-dilate-px 4 \
      --object-risk-blur-px 7 \
      --color-std-clean-tol 0.18 \
      --valid-ratio-penalty 0.45 \
      --hole-dilate-px 1 \
      --min-valid-views 2 \
      --min-view-cos 0.08 \
      --min-conf 1.0 \
      --zbuffer-stride 2 \
      --inpaint-radius 5 \
      --min-output-reliability 0.04 \
      --min-output-clean-score 0.58 \
      --max-output-contamination-score 0.42 \
      --max-output-object-risk 0.05 \
      --strict-empty-low-quality \
      "${strict_projection_evidence_args[@]}"
    mark_done strict_projection
  fi
fi
if [[ "$STRICT_V3B" == "1" && "$until_i" -ge "$(stage_index strict_projection)" ]]; then
  require_path "$STRICT_PROJECTION_DIR/nearest_visible_evidence.json"
fi

if should_run completed_observed; then
  if ! already_done completed_observed; then
    faces="$(faces_csv)"
    faces_array_from_csv "$faces"
    "$SD_PY" scripts/build_lama_completed_observed_target.py \
      --source-dir "$STRICT_PROJECTION_DIR" \
      --out-dir "$COMPLETED_OBSERVED_DIR" \
      --faces "${face_args[@]}" \
      --iopaint-bin "$IOPAINT_BIN" \
      --model "$IOPAINT_MODEL" \
      --model-dir "$IOPAINT_MODEL_DIR" \
      --device "$IOPAINT_DEVICE" \
      --completion-mode legacy_raw_iopaint \
      --filled-weight 0.35
    mark_done completed_observed
  fi
fi

if should_run chord_inputs; then
  if ! already_done chord_inputs; then
    read -r -a face_args <<< "$(faces_csv | tr ',' ' ')"
    floor_lowres_args=(--no-floor-lowres-source-adaptation)
    if [[ "$MATERIAL_FLOOR_LOWRES_SOURCE_ADAPTATION" == "1" ]]; then
      floor_lowres_args=(--floor-lowres-source-adaptation)
    fi
    "$SD_PY" scripts/generate_chord_view_contributor_region_priors.py \
      --stage prepare \
      "${STRICT_MATERIAL_ARGS[@]}" \
      --source-dir "$CHORD_SOURCE_DIR" \
      --polygon-source-dir "$SOURCE_PACKAGE_DIR" \
      --dataset-dir "$CHORD_DATASET_DIR" \
      --colmap-model-dir "$COLMAP_MODEL_DIR" \
      --da3-dir "$CHORD_DA3_DIR" \
      --object-mask-dir "$CHORD_OBJECT_MASK_DIR" \
      --out-dir "$CHORD_CANDIDATE_DIR" \
      --faces "${face_args[@]}" \
      --tile-size 512 \
      --tile-stride 64 \
      --floor-min-source-fraction 0.34 \
      --ceiling-min-source-fraction 0.42 \
      --wall-min-source-fraction 0.42 \
      --max-candidates 64 \
      --max-priors-per-face "$MATERIAL_MAX_PER_FACE" \
      --min-priors-per-face 1 \
      --candidate-nms-iou 0.24 \
      --candidate-min-center-frac 0.14 \
      --material-cluster-discovery \
      --material-cluster-components "$MATERIAL_CLUSTER_COMPONENTS" \
      --material-cluster-min-fraction "$MATERIAL_CLUSTER_MIN_FRACTION" \
      --material-cluster-chroma-merge-threshold "$MATERIAL_CLUSTER_CHROMA_MERGE_THRESHOLD" \
      --material-cluster-purity 0.86 \
      --material-cluster-min-region-size "$MATERIAL_CLUSTER_MIN_REGION_SIZE" \
      --material-cluster-exemplars 3 \
      --no-discover-persistent-wall-bands \
      --no-reject-cross-face-edge-singletons \
      --no-remove-tiny-material-islands \
      --no-thin-territory-source-adaptation \
      --wall-band-max-height-frac "$MATERIAL_WALL_BAND_MAX_HEIGHT_FRAC" \
      --wall-band-min-texture-delta "$MATERIAL_WALL_BAND_MIN_TEXTURE_DELTA" \
      --strict-ma-support \
      --strict-support-min-reliability 0.04 \
      --strict-support-clean-thresh 0.58 \
      --strict-support-object-risk-thresh 0.05 \
      --strict-support-boundary-trust-thresh 0.55 \
      --keep-valid-views 3 \
      --candidate-valid-views 2 \
      --min-valid-views 2 \
      --min-view-mask-pixels "$MATERIAL_MIN_VIEW_MASK_PIXELS" \
      --keep-clean-thresh 0.62 \
      --source-clean-thresh 0.58 \
      --keep-object-risk-thresh 0.05 \
      --source-object-risk-thresh 0.05 \
      --mask-boundary-keep-thresh 0.55 \
      --footprint-keep-min 0.12 \
      "$MATERIAL_WALL_LOWRES_ADAPT_FLAG" \
      --wall-lowres-adapt-trigger-final-keep-frac "$MATERIAL_WALL_LOWRES_ADAPT_TRIGGER_FINAL_KEEP_FRAC" \
      --wall-lowres-adapt-min-support-frac "$MATERIAL_WALL_LOWRES_ADAPT_MIN_SUPPORT_FRAC" \
      --wall-lowres-adapt-clean-thresh "$MATERIAL_WALL_LOWRES_ADAPT_CLEAN_THRESH" \
      --wall-lowres-adapt-min-valid-views "$MATERIAL_WALL_LOWRES_ADAPT_MIN_VALID_VIEWS" \
      --wall-lowres-adapt-object-risk-thresh "$MATERIAL_WALL_LOWRES_ADAPT_OBJECT_RISK_THRESH" \
      --wall-lowres-adapt-boundary-trust-thresh "$MATERIAL_WALL_LOWRES_ADAPT_BOUNDARY_TRUST_THRESH" \
      --wall-lowres-adapt-footprint-min "$MATERIAL_WALL_LOWRES_ADAPT_FOOTPRINT_MIN" \
      --wall-lowres-min-rectified-valid-frac "$MATERIAL_WALL_LOWRES_MIN_RECTIFIED_VALID_FRAC" \
      --wall-lowres-rectified-min-size "$MATERIAL_WALL_LOWRES_RECTIFIED_MIN_SIZE" \
      --wall-lowres-min-final-source-unique-pixels "$MATERIAL_WALL_LOWRES_MIN_FINAL_SOURCE_UNIQUE_PIXELS" \
      --wall-lowres-min-final-source-bbox-short-side-px "$MATERIAL_WALL_LOWRES_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX" \
      --wall-lowres-rectified-search-box-scale "$MATERIAL_WALL_LOWRES_RECTIFIED_SEARCH_BOX_SCALE" \
      --wall-lowres-rectified-search-box-min-size "$MATERIAL_WALL_LOWRES_RECTIFIED_SEARCH_BOX_MIN_SIZE" \
      --wall-lowres-planar-color-delta "$MATERIAL_WALL_LOWRES_PLANAR_COLOR_DELTA" \
      --views-per-face 0 \
      --min-view-cos 0.08 \
      --depth-abs-tol 0.045 \
      --depth-rel-tol 0.035 \
      --distance-weight-scale 1.15 \
      --distance-weight-power 1.15 \
      --surface-distance-tol 0.055 \
      --surface-distance-power 1.0 \
      --surface-distance-hard-gate \
      --surface-normal-min-cos "$SURFACE_NORMAL_MIN_COS" \
      --object-risk-hard-thresh 0.05 \
      --min-mask-boundary-trust 0.55 \
      --mask-boundary-safe-px 18 \
      --mask-boundary-power 1.15 \
      --footprint-min-area 0.32 \
      --footprint-power 0.85 \
      --zbuffer-stride 2 \
      --color-std-clean-tol 0.18 \
      --valid-ratio-penalty 0.45 \
      --chord-input-mode atlas_rectified \
      --min-rectified-valid-frac "$MATERIAL_MIN_RECTIFIED_VALID_FRAC" \
      --rectified-inner-crop \
      --rectified-inner-min-size "$MATERIAL_RECTIFIED_INNER_MIN_SIZE" \
      --rectified-inner-max-side-frac 0.82 \
      --rectified-inner-min-valid-frac "$MATERIAL_RECTIFIED_INNER_MIN_VALID_FRAC" \
      --rectified-inner-min-safe-frac "$MATERIAL_RECTIFIED_INNER_MIN_SAFE_FRAC" \
      --rectified-search-box-scale "$MATERIAL_RECTIFIED_SEARCH_BOX_SCALE" \
      --rectified-search-box-min-size "$MATERIAL_RECTIFIED_SEARCH_BOX_MIN_SIZE" \
      --floor-rectified-search-box-scale "$MATERIAL_FLOOR_RECTIFIED_SEARCH_BOX_SCALE" \
      --floor-rectified-search-box-min-size "$MATERIAL_FLOOR_RECTIFIED_SEARCH_BOX_MIN_SIZE" \
      --rectified-inner-safe-border-px 6 \
      --rectified-inner-stride-frac 0.05 \
      --rectified-inner-fallback-min-size "$MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_SIZE" \
      --rectified-inner-fallback-min-valid-frac "$MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_VALID_FRAC" \
      --rectified-inner-fallback-min-safe-frac "$MATERIAL_RECTIFIED_INNER_FALLBACK_MIN_SAFE_FRAC" \
      --chord-input-size "$CANDIDATE_CHORD_INPUT_SIZE" \
      --min-final-source-unique-pixels "$MATERIAL_MIN_FINAL_SOURCE_UNIQUE_PIXELS" \
      --min-final-source-bbox-short-side-px "$MATERIAL_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX" \
      --floor-min-final-source-unique-pixels "$MATERIAL_FLOOR_MIN_FINAL_SOURCE_UNIQUE_PIXELS" \
      --floor-min-final-source-bbox-short-side-px "$MATERIAL_FLOOR_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX" \
      "${floor_lowres_args[@]}" \
      --floor-lowres-retry-source-unique-pixels "$MATERIAL_FLOOR_LOWRES_RETRY_SOURCE_UNIQUE_PIXELS" \
      --floor-lowres-retry-source-bbox-short-side-px "$MATERIAL_FLOOR_LOWRES_RETRY_SOURCE_BBOX_SHORT_SIDE_PX" \
      --floor-lowres-rectified-min-size "$MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_SIZE" \
      --floor-lowres-rectified-max-side-frac "$MATERIAL_FLOOR_LOWRES_RECTIFIED_MAX_SIDE_FRAC" \
      --floor-lowres-rectified-min-valid-frac "$MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_VALID_FRAC" \
      --floor-lowres-rectified-min-safe-frac "$MATERIAL_FLOOR_LOWRES_RECTIFIED_MIN_SAFE_FRAC" \
      --floor-lowres-planar-color-delta "$MATERIAL_FLOOR_LOWRES_PLANAR_COLOR_DELTA" \
      --thin-territory-min-span-frac "$MATERIAL_THIN_TERRITORY_MIN_SPAN_FRAC" \
      --thin-territory-max-thickness-frac "$MATERIAL_THIN_TERRITORY_MAX_THICKNESS_FRAC" \
      --thin-territory-min-source-unique-pixels "$MATERIAL_THIN_TERRITORY_MIN_SOURCE_UNIQUE_PIXELS" \
      --thin-territory-min-source-bbox-short-side-px "$MATERIAL_THIN_TERRITORY_MIN_SOURCE_BBOX_SHORT_SIDE_PX" \
      --thin-territory-min-rectified-valid-frac "$MATERIAL_THIN_TERRITORY_MIN_RECTIFIED_VALID_FRAC" \
      --thin-territory-rectified-search-min-size "$MATERIAL_THIN_TERRITORY_RECTIFIED_SEARCH_MIN_SIZE" \
      --final-source-resolution-score-weight "$MATERIAL_FINAL_SOURCE_RESOLUTION_SCORE_WEIGHT" \
      --final-source-resolution-reference-side-px "$MATERIAL_FINAL_SOURCE_RESOLUTION_REFERENCE_SIDE_PX" \
      --no-include-atlas-fallback \
      --seed 20260626
    mark_done chord_inputs
  fi
  if [[ "$RUN_UNTIL" == "chord_inputs" ]]; then
    echo "[handoff] candidate CHORD inputs: $CHORD_CANDIDATE_DIR/chord_inputs"
    echo "[handoff] run SIZE=$CANDIDATE_CHORD_SIZE CHORD to: $CANDIDATE_CHORD_OUTPUT_DIR"
  fi
fi

if should_run candidate_chord; then
  if ! already_done candidate_chord; then
    if [[ "$RUN_CANDIDATE_CHORD_ON_2080" != "1" ]]; then
      echo "[stop] candidate CHORD not run on 2080 by default." >&2
      echo "       Run CHORD with SIZE=$CANDIDATE_CHORD_SIZE on $CHORD_CANDIDATE_DIR/chord_inputs and write $CANDIDATE_CHORD_OUTPUT_DIR" >&2
      echo "       Or set RUN_CANDIDATE_CHORD_ON_2080=1." >&2
      exit 3
    fi
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    "$CHORD_PY" "$CHORD_LOCAL_SCRIPT" \
      --chord-repo "$CHORD_REPO" \
      --ckpt "$CHORD_CKPT" \
      --config-path "$CHORD_CONFIG" \
      --input-dir "$CHORD_CANDIDATE_DIR/chord_inputs" \
      --output-dir "$CANDIDATE_CHORD_OUTPUT_DIR" \
      --device cuda \
      --size "$CANDIDATE_CHORD_SIZE" \
      --pad-multiple 32
    mark_done candidate_chord
  fi
fi

if should_run chord_compose; then
  if ! already_done chord_compose; then
    require_path "$CANDIDATE_CHORD_OUTPUT_DIR"
    read -r -a face_args <<< "$(faces_csv | tr ',' ' ')"
    "$SD_PY" scripts/generate_chord_view_contributor_region_priors.py \
      --stage compose \
      "${STRICT_MATERIAL_ARGS[@]}" \
      --source-dir "$CHORD_SOURCE_DIR" \
      --polygon-source-dir "$SOURCE_PACKAGE_DIR" \
      --dataset-dir "$CHORD_DATASET_DIR" \
      --colmap-model-dir "$COLMAP_MODEL_DIR" \
      --da3-dir "$CHORD_DA3_DIR" \
      --object-mask-dir "$CHORD_OBJECT_MASK_DIR" \
      --out-dir "$CHORD_CANDIDATE_DIR" \
      --faces "${face_args[@]}" \
      --chord-output-dir "$CANDIDATE_CHORD_OUTPUT_DIR" \
      --basecolor-key basecolor \
      --pbr-keys basecolor,normal,roughness,metallic \
      "$MATERIAL_WALL_LOWRES_ADAPT_FLAG" \
      --wall-lowres-adapt-trigger-final-keep-frac "$MATERIAL_WALL_LOWRES_ADAPT_TRIGGER_FINAL_KEEP_FRAC" \
      --wall-lowres-adapt-min-support-frac "$MATERIAL_WALL_LOWRES_ADAPT_MIN_SUPPORT_FRAC" \
      --wall-lowres-adapt-clean-thresh "$MATERIAL_WALL_LOWRES_ADAPT_CLEAN_THRESH" \
      --wall-lowres-adapt-min-valid-views "$MATERIAL_WALL_LOWRES_ADAPT_MIN_VALID_VIEWS" \
      --wall-lowres-adapt-object-risk-thresh "$MATERIAL_WALL_LOWRES_ADAPT_OBJECT_RISK_THRESH" \
      --wall-lowres-adapt-boundary-trust-thresh "$MATERIAL_WALL_LOWRES_ADAPT_BOUNDARY_TRUST_THRESH" \
      --wall-lowres-adapt-footprint-min "$MATERIAL_WALL_LOWRES_ADAPT_FOOTPRINT_MIN" \
      --wall-lowres-min-rectified-valid-frac "$MATERIAL_WALL_LOWRES_MIN_RECTIFIED_VALID_FRAC" \
      --wall-lowres-rectified-min-size "$MATERIAL_WALL_LOWRES_RECTIFIED_MIN_SIZE" \
      --wall-lowres-min-final-source-unique-pixels "$MATERIAL_WALL_LOWRES_MIN_FINAL_SOURCE_UNIQUE_PIXELS" \
      --wall-lowres-min-final-source-bbox-short-side-px "$MATERIAL_WALL_LOWRES_MIN_FINAL_SOURCE_BBOX_SHORT_SIDE_PX" \
      --seed 20260626
    mark_done chord_compose
  fi
fi

if should_run material_layout; then
  if ! already_done material_layout; then
    freeze_manifest="$(write_freeze_manifest)"
    layout_generalization_args=(
      --use-discovered-material-masks
      --allow-neutral-ceiling-wall-fallback
      --merge-small-neutral-ceiling-shading-clusters
    )
    if [[ "$STRICT_V3B_MATERIAL_PROVENANCE" == "1" ]]; then
      layout_generalization_args=(
        --no-use-discovered-material-masks
        --no-allow-neutral-ceiling-wall-fallback
        --no-merge-small-neutral-ceiling-shading-clusters
      )
    fi
    "$SD_PY" scripts/compose_material_base_atlas_v1.py \
      "${STRICT_MATERIAL_ARGS[@]}" \
      --freeze-manifest "$freeze_manifest" \
      --out-dir "$MATERIAL_LAYOUT_DIR" \
      --completed-observed-dir "$COMPLETED_OBSERVED_DIR" \
      --observed-confidence 0.58 \
      --observed-margin 0.12 \
      --color-calibration-strength 0.0 \
      --axis-boundary-min-accuracy 0.84 \
      --axis-boundary-min-class-accuracy 0.68 \
      --linear-min-accuracy 0.91 \
      --linear-min-class-accuracy 0.84 \
      --linear-min-tangent-span 0.5 \
      --linear-min-strict-fraction 0.055 \
      --axis-curve-max-deviation-frac 0.075 \
      --axis-curve-smooth-frac 0.03 \
      --axis-curve-max-tangent-samples 640 \
      --data-energy-weight 1.0 \
      --seed-mismatch-weight 5.0 \
      --boundary-complexity-weight 0.025 \
      --no-axis-layer-candidates \
      --soft-material-blend \
      --soft-probability-sigma 5.5 \
      --soft-probability-power 0.82 \
      --soft-confidence-margin 0.55 \
      --soft-boundary-radius-frac 0.08 \
      --soft-region-blur-frac 0.02 \
      --soft-weight-source target_reconstruction \
      --target-reconstruction-blur-sigma 5.5 \
      --target-reconstruction-temperature 0.55 \
      --target-reconstruction-label-prior 0.08 \
      --target-reconstruction-smooth-sigma 1.4 \
      --target-reconstruction-min-weight 0.24 \
      --target-reconstruction-weight-power 1.45 \
      --label-soft-mix-base 0.18 \
      --label-soft-mix-lowconf 0.62 \
      --soft-boundary-only \
      --soft-boundary-width-frac 0.038 \
      --pairwise-boundary-blend \
      --pairwise-target-mix 0.55 \
      "${layout_generalization_args[@]}" \
      --no-target-lowfreq-transfer
    mark_done material_layout
  fi
fi

# v3b_v3 continues with its unified MatSeg/trace-back frontend, scale-locked
# whole-territory PBR, and Unity stages in run_from_images.sh.
exit 0
