# W&B 曲线下载清单（报告用）

**项目**: https://wandb.ai/1037309108-fudan-university/hw3-3d-assets
**实体/项目**: `1037309108-fudan-university / hw3-3d-assets`

作业要求（4.2）：报告需含**训练 Loss 曲线 + 验证集指标曲线**。下面每个 run 列出要导出的面板，
请下载 PNG（也建议同时 "Download as CSV" 备份数据）后放进对应子目录。

## 怎么在 W&B 导出
- 打开 run 链接 → 左侧 **Charts/Workspace**。
- 鼠标悬停某个面板 → 右上 **⋮ → Export panel → Download PNG**（或 Download as CSV）。
- 面板很多时可用顶部搜索框输入下面的 tag 名快速定位。

---

## 1) Object A — 真实多视图 3DGS  →  放入 `A_3dgs/`
**run**: https://wandb.ai/1037309108-fudan-university/hw3-3d-assets/runs/zygn0wah  (`A_3dgs_30k_full`, finished)

| 面板 tag | 建议文件名 | 说明 |
|---|---|---|
| `train_loss_patches/total_loss` | `A_loss_total.png` | 训练总损失（L1+D-SSIM）逐迭代 |
| `train_loss_patches/l1_loss` | `A_loss_l1.png` | L1 损失逐迭代 |
| `total_points` | `A_num_gaussians.png` | 高斯点数随致密化增长 |

> ⚠️ PSNR（`train/loss_viewpoint - psnr`）这次只在最后 30k 迭代评估了一次，**不是曲线、只有一个点**
> （最终 **PSNR 40.22 dB / L1 0.0069**，已记在 RESULTS_LOG，填进超参/指标表即可）。

## 2) 背景 garden — 3DGS  →  放入 `BG_garden_3dgs/`
**run**: https://wandb.ai/1037309108-fudan-university/hw3-3d-assets/runs/p1a62kun  (`BG_garden_3dgs_30k`, finished)

| 面板 tag | 建议文件名 | 说明 |
|---|---|---|
| `train_loss_patches/total_loss` | `BG_loss_total.png` | 训练总损失逐迭代 |
| `train_loss_patches/l1_loss` | `BG_loss_l1.png` | L1 损失逐迭代 |
| `total_points` | `BG_num_gaussians.png` | 高斯点数增长（→约 3.8M） |

> ⚠️ 同样 PSNR 只有最终一个点（**PSNR 30.22 dB / L1 0.0200**）。

## 3) Object B — 文本→3D (threestudio DreamFusion-SD)  →  放入 `B_text3d/`
**run**: https://wandb.ai/1037309108-fudan-university/hw3-3d-assets/runs/w27p1z0y  (`B_text3d_4000_v2`, finished)
> ❗ 不要下另一个同名失败 run `lebywsz2`（那是提示词超 CLIP 77-token 限制崩掉的首次尝试）。

| 面板 tag | 建议文件名 | 说明 |
|---|---|---|
| `train/loss_sds` | `B_loss_sds.png` | SDS 主损失（核心 Loss 曲线） |
| `train/loss_sparsity` | `B_loss_sparsity.png` | 稀疏正则 |
| `train/loss_opaque` | `B_loss_opaque.png` | 不透明度正则 |

## 4) Object C — 单图→3D (threestudio Stable Zero123)  →  放入 `C_zero123/`
**run**: https://wandb.ai/1037309108-fudan-university/hw3-3d-assets/runs/t4qzmr9r  (`C_zero123_1000_sam3`, finished)

| 面板 tag | 建议文件名 | 说明 |
|---|---|---|
| `train/loss` | `C_loss_total.png` | 总损失 |
| `train/loss_zero123_sds` | `C_loss_zero123_sds.png` | Zero123 SDS 损失（核心 Loss 曲线） |
| `train/loss_ref_rgb` | `C_loss_ref_rgb.png` | 参考视角 RGB 重建损失（输入图保真，类"验证"曲线） |
| `train/loss_ref_mask` | `C_loss_ref_mask.png` | 参考视角 mask 损失 |

---

## 备注
- 4 个核心 Loss 曲线（A/BG 的 total_loss、B 的 loss_sds、C 的 loss_zero123_sds）是报告必放的。
- 若 W&B 面板默认是平滑后的，导出前可把 smoothing 调到 0 或保留默认（报告里注明即可）。
- 这些数值的 first/last/min/mean 我已在 `RESULTS_LOG.md` 里逐条记过，可直接引用填表。
