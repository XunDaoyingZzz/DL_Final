#!/usr/bin/env bash
# Object A: retrain 3DGS to 30k iters at full resolution, reusing the existing
# full-frame COLMAP poses (background IS used for pose estimation, as required).
# Params baked in so we never pass $vars through MINGW->wsl (which strips them).
set -euo pipefail

export HW3_ROOT=/mnt/d/2026_Spring/deeplearning/hw3
# Reuse old2's COLMAP sparse (full frames) + the undistorted reconstruction.
export DATA_ROOT=/mnt/d/2026_Spring/deeplearning/hw3/old2/data/A_real_object
export UNDISTORT_ROOT=/home/xundaoying/hw3_work/data/A_real_object_undistorted
export OUT_ROOT=/home/xundaoying/hw3_work/outputs/A_real_object_3dgs_30k_full
export REBUILD_COLMAP=0
export REBUILD_UNDISTORT=0
export RUN_TRAIN=1
export ITERATIONS=30000
export RESOLUTION=1          # full 1280px undistorted images -> sharper doll
export WANDB_UPLOAD=1
export WANDB_PROJECT=hw3-3d-assets
export WANDB_GROUP=hw3-task1-A
export WANDB_RUN_NAME=A_3dgs_30k_full

exec bash "$HW3_ROOT/scripts/wsl/run_a_colmap_3dgs.sh"
