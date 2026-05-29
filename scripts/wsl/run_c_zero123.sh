#!/usr/bin/env bash
set -euo pipefail

HW3_ROOT="${HW3_ROOT:-/mnt/d/2026_Spring/deeplearning/hw3}"
RUN_CMD="$0 $*"
source "$HW3_ROOT/scripts/wsl/common.sh"

THREESTUDIO_ROOT="${THREESTUDIO_ROOT:-/home/xundaoying/hw3_repos/threestudio}"
ENV_PREFIX="${ENV_PREFIX:-/home/xundaoying/miniconda3/envs/hw3-threestudio}"
OUT_ROOT="${OUT_ROOT:-/home/xundaoying/hw3_work/outputs/threestudio}"
CPU_SET="${CPU_SET:-0-7}"
THREADS="${THREADS:-4}"
if [[ -z "${IMAGE_PATH:-}" ]]; then
  if [[ -f "$HW3_ROOT/data/C_single_image/c_zero123_sam3_rgba_512.png" ]]; then
    IMAGE_PATH="$HW3_ROOT/data/C_single_image/c_zero123_sam3_rgba_512.png"
  else
    IMAGE_PATH="$HW3_ROOT/data/C_single_image/c_zero123_rgba_512.png"
  fi
fi
TRIAL="${TRIAL:-hw3_object_c_stable_zero123}"
MAX_STEPS="${MAX_STEPS:-400}"
TAG="${TAG:-steps_${MAX_STEPS}}"
VAL_INTERVAL="${VAL_INTERVAL:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SAMPLES_PER_RAY="${SAMPLES_PER_RAY:-256}"
ALLOW_NET_DOWNLOADS="${ALLOW_NET_DOWNLOADS:-0}"
CONFIG="${CONFIG:-configs/stable-zero123.yaml}"
ZERO123_CKPT="${ZERO123_CKPT:-$THREESTUDIO_ROOT/load/zero123/stable_zero123.ckpt}"
ZERO123_EXPECTED_BYTES="${ZERO123_EXPECTED_BYTES:-8584287851}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-hw3-3d-assets}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-C_zero123_${MAX_STEPS}_steps}"
EXPORT_AFTER="${EXPORT_AFTER:-1}"

RUN_DIR="$OUT_ROOT/$TRIAL/$TAG"
start_run_log "C_zero123" "$RUN_DIR" "project=$WANDB_PROJECT name=$WANDB_RUN_NAME"

enable_offline_model_mode "$ALLOW_NET_DOWNLOADS"
require_dir "$THREESTUDIO_ROOT"
require_file "$THREESTUDIO_ROOT/launch.py"
require_file "$THREESTUDIO_ROOT/$CONFIG"
require_file "$ENV_PREFIX/bin/python"
require_file "$IMAGE_PATH"
require_file "$ZERO123_CKPT"

actual_bytes="$(stat -c%s "$ZERO123_CKPT")"
if [[ "$actual_bytes" -lt "$ZERO123_EXPECTED_BYTES" ]]; then
  die "Zero123 checkpoint is incomplete: $ZERO123_CKPT has $actual_bytes bytes, expected about $ZERO123_EXPECTED_BYTES. Refusing to download."
fi

mkdir -p "$OUT_ROOT"
export PATH="$ENV_PREFIX/bin:$PATH"
export PYTHONPATH="$THREESTUDIO_ROOT:${PYTHONPATH:-}"
export CUDA_HOME="$ENV_PREFIX"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$ENV_PREFIX/targets/x86_64-linux/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

WANDB_ARGS=(system.loggers.wandb.enable=false)
if [[ "$WANDB_ENABLE" == "1" ]]; then
  WANDB_ARGS=(
    system.loggers.wandb.enable=true
    system.loggers.wandb.project="$WANDB_PROJECT"
    system.loggers.wandb.name="$WANDB_RUN_NAME"
  )
fi

cd "$THREESTUDIO_ROOT"
taskset -c "$CPU_SET" "$ENV_PREFIX/bin/python" launch.py \
  --config "$CONFIG" \
  --train --gpu 0 \
  use_timestamp=False \
  exp_root_dir="$OUT_ROOT" \
  name="$TRIAL" \
  tag="$TAG" \
  data.image_path="$IMAGE_PATH" \
  data.random_camera.batch_size="[$BATCH_SIZE,$BATCH_SIZE,$BATCH_SIZE]" \
  trainer.max_steps="$MAX_STEPS" \
  trainer.val_check_interval="$VAL_INTERVAL" \
  checkpoint.every_n_train_steps="$MAX_STEPS" \
  system.renderer.num_samples_per_ray="$SAMPLES_PER_RAY" \
  "${WANDB_ARGS[@]}"

if [[ "$EXPORT_AFTER" == "1" ]]; then
  taskset -c "$CPU_SET" "$ENV_PREFIX/bin/python" launch.py \
    --config "$RUN_DIR/configs/parsed.yaml" \
    --export --gpu 0 \
    resume="$RUN_DIR/ckpts/last.ckpt" \
    system.exporter_type=mesh-exporter \
    system.loggers.wandb.enable=false
fi
