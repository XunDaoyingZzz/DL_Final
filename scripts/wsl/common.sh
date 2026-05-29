#!/usr/bin/env bash

HW3_ROOT="${HW3_ROOT:-/mnt/d/2026_Spring/deeplearning/hw3}"
RESULTS_LOG="${RESULTS_LOG:-$HW3_ROOT/RESULTS_LOG.md}"
RUN_CMD="${RUN_CMD:-$0 ${*:-}}"
RUN_LABEL="${RUN_LABEL:-unnamed_run}"
RUN_OUTPUT="${RUN_OUTPUT:-}"
RUN_WANDB="${RUN_WANDB:-}"
RUN_START_TS=0
RUN_START_ISO=""

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

enable_offline_model_mode() {
  local allow="${1:-0}"
  if [[ "$allow" != "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export DIFFUSERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    echo "[INFO] Network model/dataset downloads disabled. Set ALLOW_NET_DOWNLOADS=1 only after approval."
  fi
}

start_run_log() {
  RUN_LABEL="$1"
  RUN_OUTPUT="${2:-}"
  RUN_WANDB="${3:-}"
  RUN_START_TS="$(date +%s)"
  RUN_START_ISO="$(date -Iseconds)"
  trap finish_run_log EXIT
}

finish_run_log() {
  local status=$?
  set +e
  local end_ts elapsed
  end_ts="$(date +%s)"
  elapsed=$((end_ts - RUN_START_TS))
  mkdir -p "$(dirname "$RESULTS_LOG")"
  {
    echo ""
    echo "## ${RUN_START_ISO} ${RUN_LABEL}"
    echo ""
    echo "- status: ${status}"
    echo "- elapsed_seconds: ${elapsed}"
    echo "- command: \`${RUN_CMD}\`"
    if [[ -n "$RUN_OUTPUT" ]]; then
      echo "- outputs: ${RUN_OUTPUT}"
    fi
    if [[ -n "$RUN_WANDB" ]]; then
      echo "- wandb: ${RUN_WANDB}"
    fi
  } >> "$RESULTS_LOG"
  exit "$status"
}

upload_tb_to_wandb() {
  local logdir="$1"
  local run_name="$2"
  local project="${3:-hw3-3d-assets}"
  local group="${4:-hw3-task1}"
  local job_type="${5:-train}"
  local env_prefix="${ENV_PREFIX:?Set ENV_PREFIX before upload_tb_to_wandb}"
  local cpu_set="${CPU_SET:-0-7}"

  require_file "$HW3_ROOT/scripts/upload_tb_scalars_to_wandb.py"
  taskset -c "$cpu_set" "$env_prefix/bin/python" \
    "$HW3_ROOT/scripts/upload_tb_scalars_to_wandb.py" \
    "$logdir" \
    --project "$project" \
    --name "$run_name" \
    --group "$group" \
    --job-type "$job_type"
}
