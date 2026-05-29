#!/usr/bin/env bash
# Background scene: Mip-NeRF360 garden, 3DGS to 30k iters at images_4 (the standard
# 3DGS recipe for outdoor scenes). Params baked in (MINGW strips inline $vars).
set -euo pipefail

export HW3_ROOT=/mnt/d/2026_Spring/deeplearning/hw3
export DATA_ROOT=/mnt/d/2026_Spring/deeplearning/hw3/data/background_mipnerf360/garden
export IMAGE_DIR=images_4
export OUT_ROOT=/home/xundaoying/hw3_work/outputs/background_garden_3dgs_30k
export ITERATIONS=30000
export RESOLUTION=1
export WANDB_UPLOAD=1
export WANDB_PROJECT=hw3-3d-assets
export WANDB_GROUP=hw3-task1-background
export WANDB_RUN_NAME=BG_garden_3dgs_30k

exec bash "$HW3_ROOT/scripts/wsl/run_background_3dgs.sh"
