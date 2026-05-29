#!/usr/bin/env bash
set -euo pipefail

HW3_ROOT="${HW3_ROOT:-/mnt/d/2026_Spring/deeplearning/hw3}"
RUN_CMD="$0 $*"
source "$HW3_ROOT/scripts/wsl/common.sh"

CONDA="${CONDA:-/home/xundaoying/miniconda3/bin/conda}"
COLMAP_ENV="${COLMAP_ENV:-hw3-colmap}"
GS_ENV="${GS_ENV:-hw3-3d}"
ENV_PREFIX="${ENV_PREFIX:-/home/xundaoying/miniconda3/envs/$GS_ENV}"
GS_ROOT="${GS_ROOT:-/home/xundaoying/hw3_repos/gaussian-splatting}"
DATA_ROOT="${DATA_ROOT:-$HW3_ROOT/data/A_real_object}"
IMAGE_DIR="${IMAGE_DIR:-$DATA_ROOT/images}"
SPARSE_ROOT="${SPARSE_ROOT:-$DATA_ROOT/sparse}"
UNDISTORT_ROOT="${UNDISTORT_ROOT:-/home/xundaoying/hw3_work/data/A_real_object_undistorted}"
OUT_ROOT="${OUT_ROOT:-/home/xundaoying/hw3_work/outputs/A_real_object_3dgs_${ITERATIONS:-3000}}"
CPU_SET="${CPU_SET:-0-7}"
THREADS="${THREADS:-8}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-1280}"
ITERATIONS="${ITERATIONS:-3000}"
RESOLUTION="${RESOLUTION:-2}"
REBUILD_COLMAP="${REBUILD_COLMAP:-0}"
REBUILD_UNDISTORT="${REBUILD_UNDISTORT:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
WANDB_UPLOAD="${WANDB_UPLOAD:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-hw3-3d-assets}"
WANDB_GROUP="${WANDB_GROUP:-hw3-task1-A}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-A_3dgs_${ITERATIONS}_iters}"

start_run_log "A_colmap_3dgs" "$OUT_ROOT" "project=$WANDB_PROJECT name=$WANDB_RUN_NAME"

require_dir "$IMAGE_DIR"
require_dir "$GS_ROOT"
require_file "$GS_ROOT/train.py"
require_file "$CONDA"
mkdir -p "$SPARSE_ROOT" "$UNDISTORT_ROOT" "$OUT_ROOT"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export CUDA_HOME="$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

COLMAP_RUN=("$CONDA" run -n "$COLMAP_ENV" colmap)

if [[ "$REBUILD_COLMAP" == "1" || ! -d "$SPARSE_ROOT/0" ]]; then
  echo "[INFO] Running COLMAP for Object A"
  rm -f "$DATA_ROOT/database.db"
  taskset -c "$CPU_SET" "${COLMAP_RUN[@]}" feature_extractor \
    --database_path "$DATA_ROOT/database.db" \
    --image_path "$IMAGE_DIR" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --FeatureExtraction.use_gpu 0 \
    --FeatureExtraction.max_image_size "$MAX_IMAGE_SIZE"
  taskset -c "$CPU_SET" "${COLMAP_RUN[@]}" exhaustive_matcher \
    --database_path "$DATA_ROOT/database.db" \
    --FeatureMatching.use_gpu 0
  taskset -c "$CPU_SET" "${COLMAP_RUN[@]}" mapper \
    --database_path "$DATA_ROOT/database.db" \
    --image_path "$IMAGE_DIR" \
    --output_path "$SPARSE_ROOT"
fi

require_dir "$SPARSE_ROOT/0"
taskset -c "$CPU_SET" "${COLMAP_RUN[@]}" model_analyzer --path "$SPARSE_ROOT/0"

if [[ "$REBUILD_UNDISTORT" == "1" || ! -d "$UNDISTORT_ROOT/sparse/0" ]]; then
  echo "[INFO] Undistorting Object A images"
  taskset -c "$CPU_SET" "${COLMAP_RUN[@]}" image_undistorter \
    --image_path "$IMAGE_DIR" \
    --input_path "$SPARSE_ROOT/0" \
    --output_path "$UNDISTORT_ROOT" \
    --output_type COLMAP \
    --max_image_size "$MAX_IMAGE_SIZE" \
    --num_threads "$THREADS"
  if [[ -f "$UNDISTORT_ROOT/sparse/cameras.bin" ]]; then
    mkdir -p "$UNDISTORT_ROOT/sparse/0"
    mv "$UNDISTORT_ROOT"/sparse/*.bin "$UNDISTORT_ROOT/sparse/0/"
  fi
fi

require_dir "$UNDISTORT_ROOT/sparse/0"
taskset -c "$CPU_SET" "${COLMAP_RUN[@]}" model_analyzer --path "$UNDISTORT_ROOT/sparse/0"

if [[ "$RUN_TRAIN" == "1" ]]; then
  cd "$GS_ROOT"
  taskset -c "$CPU_SET" "$CONDA" run -n "$GS_ENV" python train.py \
    -s "$UNDISTORT_ROOT" \
    -m "$OUT_ROOT" \
    -r "$RESOLUTION" \
    --data_device cpu \
    --iterations "$ITERATIONS" \
    --save_iterations "$ITERATIONS" \
    --checkpoint_iterations "$ITERATIONS" \
    --test_iterations "$ITERATIONS" \
    --disable_viewer

  if [[ "$WANDB_UPLOAD" == "1" ]]; then
    upload_tb_to_wandb "$OUT_ROOT" "$WANDB_RUN_NAME" "$WANDB_PROJECT" "$WANDB_GROUP" "3dgs-A-train"
  fi
fi
