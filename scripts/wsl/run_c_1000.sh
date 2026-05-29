#!/usr/bin/env bash
# Object C: single-image-to-3D (Stable Zero123). Reuses the clean SAM3 RGBA cutout.
# Improves on old2's 400-step coarse mesh by running 1000 steps for finer geometry.
set -euo pipefail

export HW3_ROOT=/mnt/d/2026_Spring/deeplearning/hw3
export IMAGE_PATH=/mnt/d/2026_Spring/deeplearning/hw3/data/C_single_image/c_zero123_sam3_rgba_512.png
export MAX_STEPS=1000
export TAG=steps_1000_sam3
export VAL_INTERVAL=200
export WANDB_ENABLE=1
export WANDB_RUN_NAME=C_zero123_1000_sam3
export EXPORT_AFTER=1

exec bash "$HW3_ROOT/scripts/wsl/run_c_zero123.sh"
