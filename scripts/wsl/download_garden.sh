#!/usr/bin/env bash
# Download ONLY what 3DGS needs for the Mip-NeRF360 garden scene:
#   garden/images_4/*  (quarter-res images, the standard 3DGS recipe for outdoor)
#   garden/sparse/*     (COLMAP poses)
#   garden/poses_bounds.npy
# Source: hf-mirror.com (domestic mirror) with the VPN/Clash proxy bypassed, so
# this costs NO VPN traffic. (User instruction: download via NO_PROXY -> mirror.)
set -euo pipefail

# Bypass the Clash proxy entirely; hf-mirror.com is reachable directly.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1

DEST=/mnt/d/2026_Spring/deeplearning/hw3/data/background_mipnerf360
mkdir -p "$DEST"
PY=/home/xundaoying/miniconda3/envs/hw3-threestudio/bin/python

"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
dest = "/mnt/d/2026_Spring/deeplearning/hw3/data/background_mipnerf360"
path = snapshot_download(
    repo_id="mileleap/mipnerf360",
    repo_type="dataset",
    allow_patterns=["garden/images_4/*", "garden/sparse/*", "garden/poses_bounds.npy"],
    local_dir=dest,
    local_dir_use_symlinks=False,
)
print("snapshot:", path)
PYEOF

echo "=== garden size ==="
du -sh "$DEST/garden" 2>/dev/null
echo "=== tree ==="
find "$DEST/garden" -maxdepth 2 -type d | sort
echo "=== images_4 count ==="
ls "$DEST/garden/images_4" 2>/dev/null | wc -l
echo "=== sparse files ==="
ls -lh "$DEST/garden/sparse/0" 2>/dev/null
