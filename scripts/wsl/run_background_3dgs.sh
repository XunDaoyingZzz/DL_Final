#!/usr/bin/env bash
set -euo pipefail

HW3_ROOT="${HW3_ROOT:-/mnt/d/2026_Spring/deeplearning/hw3}"
RUN_CMD="$0 $*"
source "$HW3_ROOT/scripts/wsl/common.sh"

CONDA="${CONDA:-/home/xundaoying/miniconda3/bin/conda}"
GS_ENV="${GS_ENV:-hw3-3d}"
ENV_PREFIX="${ENV_PREFIX:-/home/xundaoying/miniconda3/envs/$GS_ENV}"
GS_ROOT="${GS_ROOT:-/home/xundaoying/hw3_repos/gaussian-splatting}"
DATA_ROOT="${DATA_ROOT:-/home/xundaoying/hw3_work/data/background_mipnerf360/counter}"
IMAGE_DIR="${IMAGE_DIR:-images_2}"
OUT_ROOT="${OUT_ROOT:-/home/xundaoying/hw3_work/outputs/background_counter_3dgs_${ITERATIONS:-3000}}"
CPU_SET="${CPU_SET:-0-7}"
THREADS="${THREADS:-4}"
ITERATIONS="${ITERATIONS:-3000}"
RESOLUTION="${RESOLUTION:-1}"
WANDB_UPLOAD="${WANDB_UPLOAD:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-hw3-3d-assets}"
WANDB_GROUP="${WANDB_GROUP:-hw3-task1-background}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-BG_counter_3dgs_${ITERATIONS}_iters}"

start_run_log "background_3dgs" "$OUT_ROOT" "project=$WANDB_PROJECT name=$WANDB_RUN_NAME"

require_dir "$DATA_ROOT"
require_dir "$DATA_ROOT/$IMAGE_DIR"
require_dir "$DATA_ROOT/sparse/0"
require_dir "$GS_ROOT"
require_file "$GS_ROOT/train.py"
require_file "$CONDA"
mkdir -p "$OUT_ROOT"

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export CUDA_HOME="$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

cd "$GS_ROOT"
taskset -c "$CPU_SET" "$CONDA" run -n "$GS_ENV" python train.py \
  -s "$DATA_ROOT" \
  -i "$IMAGE_DIR" \
  -m "$OUT_ROOT" \
  -r "$RESOLUTION" \
  --data_device cpu \
  --iterations "$ITERATIONS" \
  --save_iterations "$ITERATIONS" \
  --checkpoint_iterations "$ITERATIONS" \
  --test_iterations "$ITERATIONS" \
  --disable_viewer

if [[ "$WANDB_UPLOAD" == "1" ]]; then
  upload_tb_to_wandb "$OUT_ROOT" "$WANDB_RUN_NAME" "$WANDB_PROJECT" "$WANDB_GROUP" "3dgs-background-train"
fi
