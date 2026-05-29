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
PROMPT_FILE="${PROMPT_FILE:-$HW3_ROOT/prompts/object_b_prompt.txt}"
PROMPT="${PROMPT:-}"
PROMPT_FRONT="${PROMPT_FRONT:-}"
PROMPT_SIDE="${PROMPT_SIDE:-}"
PROMPT_BACK="${PROMPT_BACK:-}"
PROMPT_OVERHEAD="${PROMPT_OVERHEAD:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
TRIAL="${TRIAL:-hw3_object_b_text3d}"
MAX_STEPS="${MAX_STEPS:-3000}"
TAG="${TAG:-steps_${MAX_STEPS}}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"
WIDTH="${WIDTH:-64}"
HEIGHT="${HEIGHT:-64}"
SAMPLES_PER_RAY="${SAMPLES_PER_RAY:-256}"
HASH_SIZE="${HASH_SIZE:-18}"
SD_MODEL="${SD_MODEL:-runwayml/stable-diffusion-v1-5}"
ALLOW_NET_DOWNLOADS="${ALLOW_NET_DOWNLOADS:-0}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-hw3-3d-assets}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-B_text3d_${MAX_STEPS}_steps}"
EXPORT_AFTER="${EXPORT_AFTER:-1}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-100.0}"

RUN_DIR="$OUT_ROOT/$TRIAL/$TAG"
start_run_log "B_text_to_3d" "$RUN_DIR" "project=$WANDB_PROJECT name=$WANDB_RUN_NAME"

enable_offline_model_mode "$ALLOW_NET_DOWNLOADS"
require_dir "$THREESTUDIO_ROOT"
require_file "$THREESTUDIO_ROOT/launch.py"
require_file "$THREESTUDIO_ROOT/configs/dreamfusion-sd.yaml"
require_file "$ENV_PREFIX/bin/python"
require_file "$PROMPT_FILE"
if [[ -z "$PROMPT" ]]; then
  PROMPT="$(grep -v '^[[:space:]]*#' "$PROMPT_FILE" | sed '/^[[:space:]]*$/d' | head -n 1)"
fi
[[ -n "$PROMPT" ]] || die "Empty prompt for Object B"

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

PROMPT_ARGS=()
if [[ -n "$PROMPT_FRONT" ]]; then
  PROMPT_ARGS+=(system.prompt_processor.prompt_front="$PROMPT_FRONT")
fi
if [[ -n "$PROMPT_SIDE" ]]; then
  PROMPT_ARGS+=(system.prompt_processor.prompt_side="$PROMPT_SIDE")
fi
if [[ -n "$PROMPT_BACK" ]]; then
  PROMPT_ARGS+=(system.prompt_processor.prompt_back="$PROMPT_BACK")
fi
if [[ -n "$PROMPT_OVERHEAD" ]]; then
  PROMPT_ARGS+=(system.prompt_processor.prompt_overhead="$PROMPT_OVERHEAD")
fi
if [[ -n "$NEGATIVE_PROMPT" ]]; then
  PROMPT_ARGS+=(system.prompt_processor.negative_prompt="$NEGATIVE_PROMPT")
fi

cd "$THREESTUDIO_ROOT"
taskset -c "$CPU_SET" "$ENV_PREFIX/bin/python" launch.py \
  --config configs/dreamfusion-sd.yaml \
  --train --gpu 0 \
  use_timestamp=False \
  exp_root_dir="$OUT_ROOT" \
  name="$TRIAL" \
  tag="$TAG" \
  data.batch_size=1 \
  data.width="$WIDTH" \
  data.height="$HEIGHT" \
  trainer.max_steps="$MAX_STEPS" \
  trainer.val_check_interval="$VAL_INTERVAL" \
  checkpoint.every_n_train_steps="$MAX_STEPS" \
  system.renderer.num_samples_per_ray="$SAMPLES_PER_RAY" \
  system.geometry.pos_encoding_config.log2_hashmap_size="$HASH_SIZE" \
  system.prompt_processor.pretrained_model_name_or_path="$SD_MODEL" \
  system.guidance.pretrained_model_name_or_path="$SD_MODEL" \
  system.guidance.guidance_scale="$GUIDANCE_SCALE" \
  system.prompt_processor.prompt="$PROMPT" \
  "${PROMPT_ARGS[@]}" \
  "${WANDB_ARGS[@]}"

if [[ "$EXPORT_AFTER" == "1" ]]; then
  taskset -c "$CPU_SET" "$ENV_PREFIX/bin/python" launch.py \
    --config "$RUN_DIR/configs/parsed.yaml" \
    --export --gpu 0 \
    resume="$RUN_DIR/ckpts/last.ckpt" \
    system.exporter_type=mesh-exporter \
    system.loggers.wandb.enable=false
fi
