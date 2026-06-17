# 环境配置

硬件：单卡 **NVIDIA RTX 4080 SUPER 32GB**，driver 591.74。
系统：**Windows 11 + WSL Ubuntu 24.04**（所有训练/渲染在 WSL 内）。
三个互不冲突的 conda 环境（两个 GPU 环境共用 **torch 2.1.2+cu118 / CUDA 11.8**）：

## 1. `hw3-colmap` — COLMAP 位姿
- COLMAP 4.0.4（CPU SIFT 即可）。用于物体 A 的 SfM 位姿与去畸变。
- 也提供 `ffmpeg`（H.264 转码备用）。

## 2. `hw3-3d` — 3D Gaussian Splatting
- 克隆 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)。
- torch 2.1.2+cu118，并按其说明安装子模块：`diff-gaussian-rasterization`、`simple-knn`。
- 另需：numpy、opencv-python、pillow、tensorboard、wandb（见 requirements.txt）。

## 3. `hw3-threestudio` — 文本/单图→3D
- 克隆 [threestudio-project/threestudio](https://github.com/threestudio-project/threestudio) 并按其 README 装依赖。
- torch 2.1.2+cu118。

## 预训练权重（本地，均不入库）
| 用途 | 权重 | 大小 | 放置位置 |
|---|---|---|---|
| 物体 B (SDS 先验) | `runwayml/stable-diffusion-v1-5` | ~4.0 GB | HuggingFace 缓存（离线加载） |
| 物体 C (Zero123) | `stable_zero123.ckpt` | ~8.0 GB | `threestudio/load/zero123/` |
| 物体 C 前景抠图 | SAM3 `sam3.pt` | ~3.3 GB | SAM3 仓库 `checkpoints/` |

> 训练产物（`*.pth/*.ckpt/*.ply`）见 README 第 5 节网盘链接。

## 数据
- 物体 A：手机环绕视频（`assets/A/`，约 39s 1080p）。
- 物体 B：文本提示词（`prompts/object_b_v2_prompt.txt`）。
- 物体 C：单张照片（`assets/C/`）→ SAM3 抠图。
- 背景：Mip-NeRF 360 `garden`，用 `scripts/wsl/download_garden.sh` 从国内镜像只拉
  `images_4 + sparse`（约 246 MB，零 VPN）。

## W&B
通过本机 `~/.netrc` 登录；
