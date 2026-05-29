# HW3 题目一 报告素材（给写 TeX 的同学）

> 多源 3D 资产生成与真实场景融合（3DGS + AIGC）。本文件汇总方法、超参、指标、耗时、对比分析、
> 图表清单，可直接转写成 NeurIPS/CVPR 风格报告。

## 外部链接（报告首/末页需附）

- **代码仓库 (GitHub)**：https://github.com/XunDaoyingZzz/DL_Final
- **模型权重 (Google Drive)**：https://drive.google.com/drive/folders/1LzN8HotaPt-oEGw0u8jNojHdot2qO88T
- **训练曲线 (W&B)**：`wandb.ai/1037309108-fudan-university/hw3-3d-assets`（导出图见 `report_materials/wandb_curves/`）

---

## 1. 任务与流水线总览

构建一个"全链路"3D 场景：用三种不同技术各生成一个 3D 物体，重建一个真实背景场景，再把三者融合到同一 3D 场景中做漫游渲染。

- **物体 A —— 真实多视角重建**：手机环绕视频 → COLMAP 位姿 → **3D Gaussian Splatting (3DGS)** 重建。
- **物体 B —— 文本→3D**：**threestudio + DreamFusion**，Stable Diffusion v1.5 作为 2D 扩散先验，**SDS Loss** 优化，仅凭一段文本生成。
- **物体 C —— 单图→3D**：手机单张照片 → **SAM3** 抠前景 → **Stable Zero123** 生成。
- **背景**：Mip-NeRF 360 `garden` 场景，3DGS 重建。
- **融合**：把 A/B/C 以合理比例/位置放到 garden 桌面，从场景**真实训练相机位姿**渲染多角度漫游视频。

物体形象统一为同一个角色（紫色波波头娃娃头挂件：星星眼、吐舌、侧边兔子发夹），便于三种方法横向对比。

---

## 2. 输入素材

| 物体 | 输入 | 规格 |
|---|---|---|
| A | 环绕视频 `assets/A/VID_20260527_155909.mp4` | 1920×1080, 30fps, ~39s；抽帧 118 张@1280×720 |
| B | 文本 prompt | "紫发、单侧兔子发夹、星星眼、吐舌的娃娃头"（英文，见 `prompts/object_b_v2_prompt.txt`） |
| C | 单张照片 `assets/C/IMG_20260527_161109.jpg` | 4096×3072；SAM3 抠图得 512×512 RGBA（score 0.95） |
| 背景 | Mip-NeRF360 `garden` | `images_4`，185 张 ~1037px（hf-mirror 下载，零 VPN） |

---

## 3. 方法与超参数（核心表 1：实验设置）

### 3.1 物体 A / 背景 garden —— 3D Gaussian Splatting
- **表示/架构**：各向异性 3D 高斯（每个高斯：位置、各向异性协方差=尺度+旋转四元数、不透明度、球谐颜色 SH degree 3）；可微光栅化渲染。
- **位姿**：COLMAP（A：118 帧注册 84 帧，平均重投影误差 0.857 px）。
- **优化器**：Adam（gaussian-splatting 默认）。
- **学习率**：position 1.6e-4 → 1.6e-6（指数衰减，max_steps 30k）；feature 2.5e-3；opacity 2.5e-2；scaling 5e-3；rotation 1e-3。
- **迭代**：30,000。**Batch**：1 张视图/迭代。
- **Loss**：`(1-λ)·L1 + λ·D-SSIM`，λ_dssim = 0.2。
- **致密化**：500→15,000 迭代，间隔 100，梯度阈值 2e-4，opacity reset 间隔 3000。
- **分辨率**：A 用 1280px（-r 1）；garden 用 images_4（-r 1）。

### 3.2 物体 B —— threestudio DreamFusion（文本→3D）
- **表示/架构**：Implicit Volume（hash-grid NeRF，几何 6.6M 参数）+ DiffuseWithPointLight 材质 + Neural Environment Map 背景 + NeRF 体渲染器；训练后用 marching cubes 导出带纹理 mesh。
- **2D 先验**：Stable Diffusion v1.5（本地缓存，离线）。**引导**：SDS，guidance_scale = 75。
- **优化器**：Adam，lr 0.01（geometry/encoding）/ 0.001（部分参数）。
- **迭代**：4,000。**Batch**：1，NeRF 渲染分辨率 64×64，256 samples/ray，hash log2 size 18。
- **Loss**：λ_sds 1.0 + λ_sparsity 1.0 + λ_opaque 0.0（+ 视角相关 anti-Janus 提示词：front/side/back/overhead + negative prompt）。

### 3.3 物体 C —— threestudio Stable Zero123（单图→3D）
- **表示/架构**：Implicit Volume（hash-grid NeRF）+ NeRF 体渲染器；导出带纹理 mesh。
- **先验/条件**：Stable Zero123（`stable_zero123.ckpt`，8.0 GB 本地权重）；条件图 = SAM3 抠出的 512 RGBA。
- **引导**：guidance_scale = 3.0。**优化器**：Adam，lr 0.01。
- **迭代**：1,000。**Batch**：2，256 samples/ray，eval 512×512。
- **Loss**：λ_sds 0.1 + λ_ref_rgb + λ_mask 50.0 + λ_orient 1.0 + λ_sparsity 0.5 + λ_opaque 0.5。

---

## 4. 指标与耗时（核心表 2：结果指标）

### 4.1 3DGS 质量指标（训练视角评估）
| 模型 | PSNR↑ | L1↓ | 高斯数 | 备注 |
|---|---|---|---|---|
| 物体 A (30k) | **40.22 dB** | 0.00694 | 270,900（裁剪后 40,691） | 干净完整娃娃头 |
| 背景 garden (30k) | **30.22 dB** | 0.0200 | ~3.8M | 户外无界场景（原论文测试集 ~27dB） |

> 说明：本次 3DGS 仅在最后 30k 步评估了一次 PSNR（单点，非曲线）；训练 Loss（total_loss / l1_loss）是逐迭代曲线（见 W&B）。

### 4.2 B/C 训练损失（threestudio）
| 物体 | 关键 Loss | first → last | min | 解读 |
|---|---|---|---|---|
| B | train/loss_opaque | 0.255 → 0.040 | 0.019 | 物体逐渐成形/实心化 |
| B | train/loss_sds | 高方差（SDS 固有） | 0.79 | SDS 梯度噪声大，量级不代表收敛 |
| C | train/loss（总） | 34.1 → 5.0 | 2.73 | 总损失稳定下降 |
| C | train/loss_ref_rgb | 0.0486 → **0.00033** | 0.00031 | **输入视角重建极佳** |
| C | train/loss_ref_mask | 0.0952 → 0.00062 | 0.00057 | 前景 mask 拟合好 |
| C | train/loss_zero123_sds | 高方差 | 8.71 | 新视角靠 Zero123 幻觉，噪声大 |

### 4.3 计算耗时（RTX 4080 SUPER 32GB，WSL）
| 阶段 | 耗时 |
|---|---|
| A：COLMAP 位姿（复用） | 44 s |
| A：3DGS 30k 训练 | 787 s（~13.1 min） |
| 背景 garden：3DGS 30k 训练 | 2070 s（~34.5 min） |
| B：DreamFusion 4000 步（含导出） | 718 s（~12.0 min） |
| C：Stable Zero123 1000 步（含导出） | 401 s（~6.7 min） |
| C：SAM3 抠图 | ~数秒 |
| 融合：组合 + 渲染 | ~分钟级 |

---

## 5. 三种生成路线对比（核心分析）

| 维度 | A 真实多视角 3DGS | B 文本→3D (SDS) | C 单图→3D (Zero123) |
|---|---|---|---|
| **输入成本** | 高（需环绕拍摄 + COLMAP） | 最低（仅一段文本） | 低（单张照片 + 抠图） |
| **几何准确度** | **高**，真实物体形状准确 | 低，marching cubes 等值面凹凸、SDS 易 Janus | 中低，正面合理、背面/侧面靠幻觉、偏粗 |
| **纹理细节** | **高**，照片级（脸、星星眼、发夹清晰） | 低，SDS 反照率不可靠；导出纹理为碎片图集（见 §6） | 中低，输入视角保真，新视角编造；导出纹理同样碎片化 |
| **计算耗时** | 中（13min + COLMAP） | 中（12min） | 低（7min） |
| **原生表示** | 显式 3DGS 高斯 | 隐式 NeRF → mesh | 隐式 NeRF → mesh |
| **可控性** | 受限于真实拍摄 | 文本可控、可幻想任意物 | 由单图主导，新视角不可控 |

**结论**：真实多视角 3DGS 在几何与纹理上**全面领先**，代价是采集成本；文本→3D 自由度最高但几何/纹理最弱；单图→3D 介于两者之间，输入视角好、其余靠生成先验补全。这与可视化结果一致：融合场景中 A 是清晰可辨的照片级娃娃头，B/C 是粗糙的纯色头。

---

## 6. 融合：隐式 mesh 与显式 3DGS 的统一渲染（作业要求的关键讨论）

**问题**：背景与 A 是**显式 3DGS 高斯球**；B/C 是 threestudio 产出的**隐式场→三角网格 mesh**。两种表示要在同一场景、同一渲染器下出图。

**做法**（`scripts/compose_garden.py` + `render_scene_from_cameras.py`）：
1. **mesh → 高斯**：对 B/C 的 mesh 做**按面积加权的表面采样**，每个采样点生成一个高斯。我们用**贴合表面的各向异性高斯（surfel）**：按三角面法线把高斯压扁成朝向表面的小圆片（沿法线方向极薄，用四元数定向），比各向同性球更接近实体表面。
2. **颜色**：threestudio 导出的 `texture_kd.jpg` 是**碎片噪声图集**（SDS/Zero123 的反照率烘焙不可靠，好看的外观只存在于 NeRF 渲染、不在导出 mesh 上）。因此对 B/C 采用**代表性纯色**（B 紫、C 暗紫灰）+ 轻微抖动着色。这本身就是"隐式生成资产→显式渲染"会损失纹理的一个实证。
3. **坐标统一**：garden 的"上"轴为 +y（由训练相机分布 + 点云侧视投影确定），桌面 y≈1.2；A 裁剪后同为 y-up。物体按真实尺度缩放、置于桌面，A 的高斯用**四元数正确旋转**（位置与高斯朝向一起转，避免 old2 只转位置导致的协方差错位）。
4. **统一渲染**：所有高斯（背景 3DGS + A 3DGS + B/C 合成高斯）拼成**单个高斯 PLY**，用原生 3DGS 光栅化器渲染。
   - 相机轨迹：从 garden 的训练相机**反求场景几何**——用相机光线的最小二乘交点估计 look-at 中心 `(0.42,1.51,1.10)`、用各相机 up 的均值估计**真实场景上方向 `(0,-0.88,-0.47)`**（garden 地面在 COLMAP 里本就倾斜 ~28°，必须用真 up 否则画面会歪）、用相机到中心的中位距离估计轨道半径（~3.65）与高度。据此生成一条**完全均匀的解析圆周轨道**（300 帧、340°），并把半径放大到 ~1.8×（镜头拉远、视野开阔）。
   - 这样既**平滑连续**（均匀采样，非离散机位），又**水平不歪**（真 up），半径贴合训练分布 → 中央桌面区域**锐利无拖影**（old2 的合成轨道因偏离训练分布而产生白色拖影，本方案通过反求训练几何避免）。
   - 视频转码为 **H.264 (yuv420p)** 以便各播放器/网页流畅播放（`report_materials/fusion/fusion_garden_final.mp4`，300 帧@30fps）。

**融合规模**：背景 ~4.20M + A 40,691 + B 200k + C 200k ≈ **4.64M 高斯**。
**输出**：`report_materials/fusion/fusion_garden_final.mp4`（漫游展示视频）。

---

## 6.5 B/C 在融合场景中渲染较差的成因分析（报告重点）

**现象**：在统一融合场景中，物体 A（真实多视图 3DGS）是照片级、可清晰辨认（脸、星星眼、兔子发夹清晰）；
而 B（文本→3D）、C（单图→3D）只能呈现为**几何凹凸、表面纯色**的粗糙头。下面给出根因分析——
这并非管线缺陷，而是三条生成路线本质差异的体现，正对应作业"对比三种方式 + 讨论表示统一"的要求。

**根因 1：几何监督方式不同（决定几何精度）。**
A 由真实多视角照片 + COLMAP 位姿 + 光度重建（L1+D-SSIM）得到，有**多视图一致的真实观测约束**，
几何准确（训练视角 PSNR 40.2 dB）。B/C 则是用 **SDS / Zero123-SDS** 优化一个隐式 NeRF 密度场：
SDS 梯度高方差、由 2D 扩散先验"想象"而非真实光度约束，密度场本身就是含噪、块状的；再经 marching cubes
抽取等值面，得到的是**疙瘩状、非干净闭合**的网格。增加迭代步数只能边际改善，无法改变这一本质。

**根因 2：导出网格的反照率纹理不可用（决定纹理细节）。**
threestudio 的 mesh-exporter 从模型 albedo 烘焙 UV 纹理。但 SDS 训练优化的是**渲染（着色后）图像**而非物理
意义上的 albedo——NeRF 旋转视频里好看的紫发其实来自"着色 + 扩散先验"，并不存在一张干净的 albedo 贴图。
我们实测导出的 `texture_kd.jpg`：B 是品红/黄/黑噪点图集、C 是灰/橙/黑碎片岛屿（见 `report_materials` 纹理图）。
直接采样会得到"礼花"噪点，因此我们改用**代表性纯色**着色——这就是 B/C 呈现为纯色头的直接原因。

**根因 3：C 的单图先天信息缺失（新视角靠幻觉）。**
Zero123 对**输入视角**重建极佳（`train/loss_ref_rgb` 由 0.049 降到 **0.0003**），但其余视角没有任何真实观测，
完全由 2D 先验**幻觉**补全 → 背面/侧面几何粗糙、可能不一致。这是单图→3D 的固有局限。

**根因 4：表示转换的信息损失（正是"隐式 mesh × 显式 3DGS 统一"的讨论点）。**
A 与背景是**原生优化的各向异性 3D 高斯**（含视角相关 SH、光度准确）。B/C 走的是
`隐式 NeRF → marching cubes mesh → 重采样为合成高斯`：每一步都丢信息——网格离散化密度场、
采样为高斯 + 纯色丢弃了（本就不可用的）纹理。因此即便网格干净，把一个生成式 mesh 资产塞进 3DGS 场景，
也天然劣于一个在该场景坐标系下原生训练的 3DGS 物体。

**证据小结（可引用）**

| 维度 | A（真实多视图 3DGS） | B（文本 SDS） | C（单图 Zero123） |
|---|---|---|---|
| 监督 | 多视图真实光度 | SDS（2D 先验，无真实多视图） | 单视图真实 + 其余幻觉 |
| 几何 | 准确（PSNR 40.2） | 块状等值面 | 粗糙，背面幻觉 |
| 导出纹理 | —（原生高斯，无需导出） | 噪点图集（不可用） | 碎片图集（不可用） |
| 在融合场景表示 | 原生高斯（无损） | mesh→合成高斯+纯色（多重损失） | mesh→合成高斯+纯色（多重损失） |

**结论**：本实验清晰地复现了三条路线的本质权衡——真实多视图 3DGS 以较高采集成本换取**高几何+纹理保真**；
文本/单图→3D 以极低输入成本换取**显著更弱的几何与纹理**，并在统一进显式高斯场时叠加**表示转换损失**。
B/C 的"差"是该对比研究的**预期结论与发现**，而非实现缺陷；它也支撑了报告中关于"隐式生成资产与显式 3DGS
如何统一渲染、以及为何会有质量落差"的讨论。

**若要进一步缩小落差（未来工作，可在报告"改进方向"中提及）**：将 NeRF 的逐顶点颜色烘焙到 mesh（恢复真实颜色）、
对 mesh 做 Laplacian 平滑、或采用更强的几何先验/多视图蒸馏的生成方法；但这些都改不掉 SDS/单图的根本局限。

## 7. 图表清单（插入报告）

- **训练曲线**（W&B 导出，见 `report_materials/wandb_curves/`）：A & garden 的 `total_loss`/`l1_loss`；B 的 `loss_sds`；C 的 `loss_zero123_sds`/`loss_ref_rgb`。W&B 项目：`1037309108-fudan-university/hw3-3d-assets`。
  - 注：A/garden 的 `total_loss` 曲线每 ~3000 步出现的尖峰是 **opacity reset**（每 3000 步重置不透明度）所致，属 3DGS 正常现象，随后迅速回落收敛。
  - 注：B/C 的 SDS 类损失（`loss_sds`/`loss_zero123_sds`）**高方差是 SDS 固有特性**，其绝对量级不代表收敛；B 的收敛更宜看 `loss_opaque`↓、C 看 `loss`/`loss_ref_rgb`↓。
- **物体 A 单体多视角**：`report_materials/fusion/A_object_contactsheet.jpg`（8 视角，证明几何/纹理完整）。
- **背景 garden 单体**：`report_materials/fusion/garden_only_contactsheet.jpg`（锐利、无拖影）。
- **融合场景**：`report_materials/fusion/fusion_garden_final.mp4` + `fusion_arc_contactsheet.jpg`（A 清晰立于桌面）。
- **B/C 单体**（可选，展示原貌）：threestudio 的 `it4000-test.mp4` / `it1000-test.mp4`（NeRF 旋转视频）。

---

## 8. 环境与可复现命令（详见仓库 README）

- 硬件：RTX 4080 SUPER 32GB；WSL Ubuntu 24.04。
- 环境（互不冲突）：`hw3-colmap`（COLMAP）、`hw3-3d`（torch 2.1.2+cu118 + diff-gaussian-rasterization）、`hw3-threestudio`（threestudio）。
- 权重：均为本地已有，**无新增大文件下载**（SD v1.5 缓存 4.0G、stable_zero123.ckpt 8.0G、SAM3 3.3G）。
- 关键脚本：`scripts/wsl/run_a_colmap_3dgs.sh`、`run_bg_garden_30k.sh`、`run_b_v2.sh`、`run_c_1000.sh`；`scripts/crop_a_geometric.py`、`compose_garden.py`、`render_scene_from_cameras.py`。

---

## 9. 限制与改进方向（诚实记录）

- B/C 的几何受 SDS/Zero123 上限制约（凹凸/幻觉），且 threestudio 导出纹理不可用 → 融合中表现为纯色粗糙头。
- 3DGS PSNR 仅末步评估一次，缺验证集 PSNR 曲线（如需，可加密 `--test_iterations`）。
- 桌面小物体在远距固定训练相机下取景敏感，最终视频选取了能清晰展示 A 的方位角弧段。
- 可能的改进：B/C mesh 做 Laplacian 平滑、或将 NeRF 顶点色烘焙到 mesh 以保留颜色；A 训练时引入前景 mask 进一步去背景。
