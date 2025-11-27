# aware-snn-mm: 多模态意识核脉冲网络（CIFAR-100, 纯 STDP）

> 使用 **纯 STDP / 奖励调制 STDP（R-STDP）**，在 CIFAR-100 上实现 **图片–文字多模态对齐 + 意识核 + 惊讶驱动注意力** 的脉冲神经网络。  
> 不使用反向传播和替代梯度（surrogate gradient）。

---

## 1. 项目简介

本仓库实现了一个面向研究的原型系统：

- 视觉通路：CIFAR-100 图片 → 初级 / 中级 / 高级 脉冲视觉通路（多子网）
- 文字/符号通路：CIFAR-100 标签（类名）→ 文本/语义脉冲通路
- 意识核（Core）：接收视觉 + 文本高层脉冲，执行：
  - 多模态对齐（图片–文字共享表征）
  - 惊讶度估计（surprise）
  - 振荡周期中的注意力门控（选择/抑制子网）
  - 奖励调制 STDP（R-STDP）更新
- 训练方式：**完全基于 STDP / R-STDP，无 BP/SGD，无替代梯度**

适合作为以下方向的实验平台：

- 多模态 SNN（图片+标签/文字）对齐
- 意识核 / 意识空间 / 概念空间的脉冲实现
- 惊讶驱动的注意力与结构生长
- 纯本地学习规则（STDP）与少量全局奖励信号的结合

---

## 2. 核心特性

- ✅ **纯 STDP 学习**：各层采用局部 STDP 或奖励调制 STDP；没有任何形式的反向传播或替代梯度。
- ✅ **多模态对齐**：CIFAR-100 图片 + 标签词共同输入，意识核对齐视觉表征与文字表征。
- ✅ **分层视觉通路**：
  - 初级视觉：边缘/方向/颜色等低级特征
  - 中级视觉：部件组合、形状片段
  - 高级视觉：物体/场景级表征
  - 每层默认 8 个子网，模拟不同特征维度/专家。
- ✅ **标签/文字通路**：
  - 支持简单 one-hot 标签脉冲输入（100 维）
  - 预留扩展到单词/字符级文字 SNN
- ✅ **意识核（Awareness Core）**：
  - 接收视觉 + 文字高层脉冲
  - 建立多模态共享表示
  - 计算惊讶度，调节注意力与学习率
  - 支持神经元生长与剪枝
- ✅ **振荡与注意力**：
  - 仿真时间划分为多个振荡周期（例如 gamma 周期）
  - 每个周期末计算惊讶度
  - 下一周期根据惊讶度重新分配各层子网的门控权重
- ✅ **结构生长/剪枝**：
  - 对高惊讶模式执行意识核神经元生长
  - 对长期不活跃的神经元执行剪枝

---

## 3. 目录结构

建议参考的目录结构如下（可根据实际实现微调）：

```text
aware-snn-mm/
├── configs/
│   ├── cifar100_unsup_vision.yaml        # 视觉通路无监督 STDP 配置
│   ├── cifar100_unsup_text.yaml          # 文本/标签通路无监督 STDP 配置
│   ├── cifar100_align_core.yaml          # 多模态对齐 + 意识核训练配置
│   └── cifar100_rstdp_classifier.yaml    # R-STDP 分类头配置
├── data/
│   └── cifar-100-python/                 # CIFAR-100 官方格式数据
├── snnmm/
│   ├── datasets/
│   │   └── cifar100.py                   # 数据加载与预处理
│   ├── encoding/
│   │   ├── vision_encoding.py            # 图片 → 脉冲编码
│   │   └── text_encoding.py              # 标签词 → 脉冲编码
│   ├── layers/
│   │   ├── neurons.py                    # LIF/IF 神经元实现
│   │   ├── stdp.py                       # STDP 与 R-STDP 规则
│   │   ├── oscillation.py                # 振荡周期 & 惊讶度计算
│   │   ├── gating.py                     # 8 子网门控/注意力
│   │   └── growth.py                     # 意识核生长与剪枝逻辑
│   ├── models/
│   │   ├── vision_path.py                # 初级/中级/高级视觉通路（多子网）
│   │   ├── text_path.py                  # 标签/文字通路
│   │   ├── core.py                       # 意识核 SNN + 多模态对齐
│   │   └── full_model.py                 # 端到端组装
│   ├── training/
│   │   ├── train_unsup_vision.py         # 视觉通路无监督 STDP 预训练
│   │   ├── train_unsup_text.py           # 文本通路无监督 STDP 预训练
│   │   ├── train_align_core.py           # 多模态对齐 + 意识核 STDP
│   │   ├── train_rstdp_classifier.py     # R-STDP 分类头训练
│   │   └── evaluate.py                   # 测试与评估
│   └── utils/
│       ├── logging.py                    # 日志与可视化
│       ├── checkpoint.py                 # 权重保存与加载
│       └── seed.py                       # 随机种子/实验复现实验
├── scripts/
│   ├── run_all.sh                        # 一键跑完整训练流程
│   └── visualize_spikes.py               # 脉冲栅格图/对齐可视化
├── README.md
└── LICENSE
```

---

## 4. 环境与安装

建议环境：

- Python 3.9+
- NumPy
- PyTorch / JAX（任选其一，按实现选择；示例以下假定 PyTorch）
- Matplotlib（可视化）
- tqdm 等辅助库

安装步骤（示例）：

```bash
git clone https://github.com/your-name/aware-snn-mm.git
cd aware-snn-mm

# 建议使用虚拟环境
python -m venv venv
source venv/bin/activate  # Windows 下使用 venv\\Scripts\\activate

pip install -r requirements.txt
```

`requirements.txt` 示例：

```text
torch>=2.0
torchvision>=0.15
numpy
matplotlib
tqdm
pyyaml
```

---

## 5. 数据准备：CIFAR-100

下载 CIFAR-100 官方 Python 版本数据集：

```bash
mkdir -p data
cd data
# 这里用官方链接或手动下载
# 下载后解压到 data/cifar-100-python
```

目录结构应类似：

```text
data/
└── cifar-100-python/
    ├── train
    ├── test
    ├── meta
    └── ...
```

`datasets/cifar100.py` 将负责：

- 加载 train/test
- 可选标准化（均值/方差）
- 提供：
  - 原始图片（32×32×3）
  - 类别索引（0–99）
  - 类别名称（英文字符串）

---

## 6. 架构概览

### 6.1 整体数据流

图片脉冲 + 标签/文字脉冲  
→ 多层视觉 SNN + 文本 SNN  
→ 意识核 SNN（核心自我/意识空间）  
→ 分类输出 + 惊讶度 + 注意力反馈

- 视觉通路：3 层（初级 / 中级 / 高级），每层 8 个子网（专家）
- 文本通路：2–3 层（标签 → 低级文字 → 语义）
- 意识核：循环脉冲网络，具有视觉偏向、文本偏向、共享/对齐神经元

### 6.2 多子网（8 专家）

对每个视觉层 k（初级/中级/高级），有 8 个子网 f_{k,i}，门控向量 g_{k,i} 控制各子网参与度：

```
h_k(t) = sum_{i=1}^8 g_{k,i} * f_{k,i}(h_{k-1}(t))
```

门控由意识核状态 + 惊讶度决定；门控本身可通过慢速 R-STDP 或简单启发式更新。

### 6.3 振荡与惊讶驱动

- 仿真时间离散为 T 步（例如 100 步），划分为多个振荡周期（例如每 10 步为一个周期）。
- 每个周期结束时：
  - 意识核汇总该周期内的发放情况
  - 估计分类困惑度 + 多模态对齐误差，得到惊讶度 S
  - 根据 S 更新下一周期的子网门控与学习率

---

## 7. 脉冲编码设计

### 7.1 图片 → 脉冲

- 将 CIFAR-100 图片转换为灰度或 Y 通道（或保留 RGB，分通道编码）。
- 映射像素强度到发放率 r：

```
r_xy = r_max * I_xy / 255.0
```

- 在时间窗 T 内，用泊松过程或 Bernoulli 过程生成脉冲：输入神经元在每个时间步以概率 r_xy * Δt 产生脉冲。

### 7.2 标签/文字 → 脉冲

- 简单模式（默认）：CIFAR-100 有 100 个 coarse/fine label，为每个类别分配一个输入神经元；正确类别对应神经元在整个时间窗内以高频发放，其他类不发放或低频噪声。
- 可扩展为“文字 SNN”：标签字符串 → token → embedding → 将 embedding 各维度映射为脉冲发放率。

---

## 8. 学习算法：STDP 与 R-STDP

### 8.1 基本 STDP（无监督）

- 若 pre 在前、post 在后，则加强连接：Δw_ij^+ ∝ A^+ * exp(-Δt / τ^+)，Δt = t_post - t_pre > 0。
- 若 post 在前、pre 在后，则削弱连接：Δw_ij^- ∝ -A^- * exp(Δt / τ^-)，Δt < 0。
- 实现时通常使用离散时间 + 可塑性迹（trace）：维护 pre-trace 和 post-trace，每个时间步衰减，当神经元发放时 trace += 1，以 trace 乘积近似 STDP 更新。

### 8.2 奖励调制 STDP（R-STDP）

- 在意识核 + 分类读出层中使用 R-STDP，每个突触 w_ij 维护 eligibility trace e_ij。
- 正 STDP 事件使 e_ij 增加，负 STDP 事件使 e_ij 减少。
- 一个样本或一个振荡周期结束时，计算全局奖励信号 R（如分类正确 +1，错误 −1 或 0，或结合惊讶度构造）。
- 最终权重更新：Δw_ij = η * R * e_ij。STDP 提供候选更新，R 决定被强化还是被抑制，无需 BP/SGD。

### 8.3 多模态对齐的 Hebbian 规则

- 对视觉高层神经元 v 与文本高层神经元 t 之间的连接，使用 Hebbian + R-STDP：同一振荡周期内高频共发放则正向累积 eligibility trace。
- 样本结束时，根据多模态对齐程度给奖励：若视觉与文本属于同一类别，奖励 R_align > 0；若属于不同类别（batch 内负样本组合），奖励 R_align < 0。
- 最终更新：Δw_vt = η_align * R_align * e_vt，在意识核附近形成共享多模态概念空间。

---

## 9. 惊讶度 S 与注意力门控

### 9.1 惊讶度定义

- 分类困惑度：C_cls = 1 - p(预测正确) 或基于脉冲率估计的置信度反函数。
- 多模态对齐误差（例如平均发放率向量 z）：C_align = ||z_vis - z_text||^2。
- 综合：S = α * C_cls + β * C_align。

### 9.2 基于 S 的门控与学习率调节

- 基础学习率 η_k,i 与门控 g_k,i 可被调节。
- 实际生效的学习率：η_k,i_eff = η_k,i * (1 + γ * S * g_k,i)。
- 对高 S 的样本：激活更多子网（门控更平/更广）、提高参与子网的 STDP 学习率、可延长仿真时间或振荡周期数。
- 对低 S 的样本：保持稀疏专家组合，低学习率微调，避免破坏已有记忆。

---

## 10. 意识核生长与剪枝

### 10.1 生长触发

- 周期性检查哪些样本/类别长期具有高惊讶度 S，对这些样本在意识核中的发放模式做聚类。
- 若某个簇内的模式“挤在一起”（类内分散度大），认为现有概念维度不足，触发生长。
- 对该簇样本计算平均激活向量 z_bar，新建一组意识核神经元，权重初始化偏向 z_bar，再由 STDP/R-STDP 细化。

### 10.2 剪枝与合并

- 对长期激活率接近 0 的意识核神经元进行剪枝。
- 对激活模式高度相似的神经元进行合并（权重平均或拼接）。

---

## 11. 训练流程示例

### 11.1 阶段 1：无监督 STDP 预训练视觉通路

```bash
python -m snnmm.training.train_unsup_vision --config configs/cifar100_unsup_vision.yaml
```

- 仅使用图片脉冲，在初级/中级/高级视觉层上应用 STDP，目标是学习边缘/纹理/形状的稳定表征。

### 11.2 阶段 2：无监督 STDP 预训练文本/标签通路

```bash
python -m snnmm.training.train_unsup_text --config configs/cifar100_unsup_text.yaml
```

- 仅使用标签脉冲（或标签文字编码），学习标签之间的相对结构（如 coarse/fine label 关系）。

### 11.3 阶段 3：多模态对齐 + 意识核 STDP

```bash
python -m snnmm.training.train_align_core --config configs/cifar100_align_core.yaml
```

- 同时输入图片脉冲 + 标签脉冲，在意识核与高层视觉/文本之间应用 Hebbian + R-STDP，建立共享多模态概念空间；训练惊讶度估计与注意力门控（基于简单规则或 R-STDP）。

### 11.4 阶段 4：R-STDP 分类读出

```bash
python -m snnmm.training.train_rstdp_classifier --config configs/cifar100_rstdp_classifier.yaml
```

- 在固定的意识核表示基础上，使用 R-STDP 训练读出层；分类正确给予正奖励，错误给予负奖励；不使用任何形式的 BP/SGD。

### 11.5 评估

```bash
python -m snnmm.training.evaluate --checkpoint path/to/checkpoint.pt
```

- 输出分类准确率，可选输出：图片–文字对齐质量（例如利用核心表征做最近邻检索/对比）、惊讶度分布、神经元生长/剪枝统计。

---

## 12. 可视化与分析

使用 `scripts/visualize_spikes.py` 可以生成脉冲栅格图（raster plot），并对比：

- 视觉 vs 文本在意识核中的发放模式
- 高惊讶 vs 低惊讶样本的神经元激活
- 意识核生长前后，概念空间的变化（例如用 t-SNE/UMAP 对发放率投影）

用法示例：

```bash
# 使用 checkpoint 前向采样并可视化指定层的脉冲
python -m scripts.visualize_spikes \
  --checkpoint checkpoints/rstdp_epoch1.pt \
  --mode core \            # 可选：vision_stage1/vision_stage3/core/classifier
  --save-path logs/spikes_core.png \
  --timesteps 10 \
  --num-samples 8

# 直接可视化已有 npy 脉冲文件
python -m scripts.visualize_spikes --spike-file path/to/spikes.npy --save-path logs/spikes.png

# 文本无监督 STDP（高发放率 + 低阈值，激活更高）
python -m snnmm.training.train_unsup_text \
  --config configs/cifar100_unsup_text.yaml \
  --device mps \
  --epochs 10 \
  --batch-size 128 \
  --timesteps 60 \
  --high-rate 0.98 \
  --low-rate 0.02 \
  --lr 0.002 \
  --use-third-layer \
  --threshold 0.2

# R-STDP 分类（较长训练）
python -m snnmm.training.train_rstdp_classifier \
  --config configs/cifar100_rstdp_classifier.yaml \
  --device mps \
  --epochs 20 \
  --timesteps 60 \
  --cycle-length 10 \
  --lr-cls 0.003 \
  --surprise-alpha 0.5 \
  --surprise-beta 0.2 \
  --batch-size 64
```

---

## 13. 后续扩展方向

- 文本通路：将 CIFAR-100 标签扩展为自然语言描述；使用字符/子词级 SNN 实现“真正的”文字通路。
- 多模态扩展：增加音频（为每个类别生成语音标签）→ 三模态对齐（图像/文字/语音）。
- 结构优化：引入三胞神经元（dendrite compartments）以模拟更细致的树突计算；引入更复杂的生长/再拓扑机制（区域分化、子脑）。
- 任务扩展：从 CIFAR-100 迁移到更复杂数据集（Tiny ImageNet 等）；增加序列任务（视频），利用振荡周期处理时间信息。
- 在线/持续学习：利用惊讶度驱动增量学习与记忆保护，研究无需重放的持续学习效果。

---

## 14. License

本项目建议使用 MIT License（可根据实际需求修改）。

```text
MIT License
...
```

---

## 15. 引用（如用于论文/报告）

如在科研工作中使用本仓库，请在文中简单说明：

> We used a spike-based multi-modal awareness-core architecture (aware-snn-mm) trained purely with STDP and reward-modulated STDP on CIFAR-100 for image–label alignment and classification, without any surrogate gradient or backpropagation.
