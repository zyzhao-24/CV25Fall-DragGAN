# 实验方案：混合跟踪策略 (Hybrid Tracking Strategy)

## 1. 背景与动机 (Background & Motivation)
在 Task 1 (DragGAN) 中，点跟踪（Point Tracking）是连接连续优化步骤的关键环节。
*   **问题**: 原始的最近邻搜索（Nearest Neighbor / L2 Search）在纹理较少（texture-less）的区域（如天空、平滑物体表面）容易失效。因为在这些区域，特征图（Feature Map）非常相似，导致跟踪点在错误的“相似”位置之间跳动或产生漂移 (Drift)。
*   **候选方案**: 单纯使用光流网络（如 RAFT）可以捕捉像素级的运动趋势，但在 GAN 生成图像存在伪影或大形变时，光流本身的精度可能不足以精确锁定语义点。
*   **提出的方案**: **RAFT-Guided Patch Refinement (混合跟踪策略)**。

**核心思想**: 利用 RAFT 提供的光流作为“宏观指导” (Global Guidance) 来确定点的粗略位置，然后在一个受限的小范围内使用 Patch-based L2 Search 进行“微观修正” (Local Refinement)。

---

## 2. 提出的 Pipeline (Proposed Pipeline)

在 DragGAN 的每一次迭代循环中（Motion Supervision -> W 优化 -> Image Generation -> Point Tracking），新的跟踪流程如下：

### 步骤 1: 图像生成与预处理
*   利用优化后的 Latent $w$ 生成当前帧图像 $I_{curr}$ 和对应的特征图 $F_{curr}$。
*   保留上一帧图像 $I_{last}$。

### 步骤 2: 光流粗定位 (Coarse Localization / RAFT Guidance)
*   **输入**: $I_{last}, I_{curr}$。
*   **操作**: 运行 RAFT 模型，计算从上一帧到当前帧的光流场: $\Delta \mathcal{F} = RAFT(I_{last}, I_{curr})$。
*   **预测**: 对于每一个当前控制点 $p_{i}^{(t-1)}$，根据其坐标处的光流向量 $(\Delta x, \Delta y)$，预测一个粗略的新位置 $p_{rough}$:
    $$ p_{rough} = p_{i}^{(t-1)} + (\Delta x, \Delta y) $$
*   **作用**: 这一步利用了图像的整体运动信息，可以跨越纹理缺失区域，防止点因为局部相似性而“迷路”。

### 步骤 3: 局部特征精修 (Local Patch Refinement / L2 Constraint)
*   **输入**: 粗略位置 $p_{rough}$，当前特征图 $F_{curr}$，控制点的初始参考特征 $f_{ref} = F_0(p_0)$。
*   **操作**:
    *   在以 $p_{rough}$ 为中心的小半径 $r_{small}$ (例如 3-5 像素) 范围内定义搜索区域 $\Omega$。
    *   计算区域 $\Omega$ 内每个位置特征与参考特征 $f_{ref}$ 的 L2 距离。
    *   找到 L2 距离最小的位置作为最终更新点 $p_{i}^{(t)}$:
        $$ p_{i}^{(t)} = \arg\min_{q \in \Omega(p_{rough})} || F_{curr}(q) - f_{ref} ||_2 $$
*   **作用**: 纠正光流的微小误差，确保控制点始终锚定在语义特征最匹配的位置。

---

## 3. 实验计划 (Experiment Plan)

### 3.1 对比实验设置 (Baselines)
我们将对比三种跟踪策略在相同 Drag 操作下的表现：
1.  **Baseline 1 (Original L2)**: 原始 DragGAN 方法，在全图或较大半径内仅进行特征最近邻搜索。
2.  **Baseline 2 (Pure RAFT)**: 仅使用 RAFT 输出更新点坐标，不进行特征修正。
3.  **Ours (Mixed/Hybrid)**: 上述提出的 "RAFT Guide + Small Patch Refinement"。

### 3.2 评估场景 (Test Cases)
*   **Case A (Rich Texture)**: 这一点通常容易跟踪（如猫的眼睛、汽车轮毂），用于验证混合方法没有退化。
*   **Case B (Texture-less / Smooth)**: 这一点是重点（如蓝天、人脸平滑皮肤、汽车引擎盖）。观察点是否能跟随物体移动而不滑向周围相似区域。

### 3.3 实施路线 (Implementation Roadmap)
1.  **代码准备**: 
    *   确保 `core.py` 中已经集成了 RAFT 模型加载和推断代码（已存在，需验证正确性）。
    *   在 `core.py` 中实现新的逻辑分支 `tracking_method == 'mixed'`。
2.  **逻辑实现**:
    *   调用 `point_tracking_raft` 获取 `dx, dy`。
    *   不直接返回新点，而是将新点作为中心输入给修改后的 `point_tracking_L2_point`（限制搜索范围）。
3.  **运行与观察**:
    *   运行 `demo.py` 或 `visualizer.py`，设置不同的 `tracking_method`。
    *   保存中间帧流可视化 (Flow Visualization) 和点的轨迹图 (Points Log)。

---

## 4. 预期结论 (Hypothesis)
我们预计 **Mixed 方法** 将表现出如下特性：
*   **抗干扰性强**: 在大位移或纹理模糊时，依靠 RAFT 不会跟丢。
*   **精度高**: 在物体停止运动或精细调整时，依靠 L2 特征匹配能精准对齐。
