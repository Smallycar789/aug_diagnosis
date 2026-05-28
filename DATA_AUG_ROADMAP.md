# 数据增强算法验证路线

本文档记录当前对 `aug_diagnosis` 项目的理解，以及后续数据增强算法验证计划。目标是让数据增强部分能服务于后续故障诊断验证，包括迁移学习、小工况样本迁移、少样本补充等场景。

## 当前项目认知

项目目前分为两条主线：

- `data_aug/`：数据增强与生成算法，当前已有 `VAE`、`GAN`、`GAN-VAE`、`RVAE` 的工程化入口。
- `diagnosis/`：故障诊断算法，当前已有 `CNN-BiGRU`、`ResNet/STFT`、`TL-Meta` 等诊断验证代码。

`sensitivity` 与 `image_quality` 来自同一个红外退化仿真系统 `Degradation`。当前可以把它们作为两个任务视角分别验证生成算法，但后续数据生成目标应回到统一的 `Degradation` 数据：同时生成灵敏度与成像质量相关参数，并保留统一故障标签。

当前数据增强部分已经做了第一步改造：

- `data_aug/data_load.py` 开始支持通用 `class_files` 形式的多类别 CSV 滑窗加载。
- `data_aug/rvae.py` 已有一版条件 RVAE 改造，支持多维输入和 label embedding。
- 新增了 `configs/data_aug/smoke_rvae_sensitivity.yaml`，用于先打通 `sensitivity` 数据集的 RVAE 训练、生成、评估链路。

该链路还需要在目标 conda 环境 `diag` 中实际运行验证。

## 总目标

实现并验证三类数据增强算法：

- `RVAE`
- `GAN`
- `VAE-GAN`

所有算法都必须支持**多维参数生成**。也就是说，模型输入输出不应只是一维序列，而应支持类似：

```text
samples: (num_windows, seq_len, num_features)
```

其中 `num_features` 来自不同数据集配置中的 CSV 特征列。当前统一 `Degradation` 生成目标建议使用五个核心特征：

```yaml
value_columns:
  - avg_detectivity
  - NETD_mK
  - SiTF
  - signal_gain
  - read_noise_index
```

生成结果也应保留同样的多维结构，便于后续接入诊断模型或重新拼接为增强数据。

## 条件生成策略

数据增强算法**不强制要求都是条件生成模型**。

对于带有明确类别的数据集，例如统一后的 `degradation`：

- 可以训练一个条件生成模型，通过 label 控制生成类型。
- 也可以针对每一种数据类型单独训练一个模型，例如：
  - `normal` 单独训练一个 RVAE/GAN/VAE-GAN。
  - `sensitivity_degradation` 单独训练一个模型。
  - `mtf_degradation` 单独训练一个模型。
  - `nonuniformity_degradation` 单独训练一个模型。
  - `bad_pixel_degradation` 单独训练一个模型。
  - `coupled_severe_fault` 单独训练一个模型。

两种方案都可接受，但必须满足：

- 生成样本能明确对应原始数据类型。
- 每类生成结果可以单独保存和评估。
- 生成样本的维度、特征列顺序、归一化参数可追踪。

初期建议优先打通**每类可独立生成**的最小链路。条件生成可以作为 RVAE 或后续 GAN/VAE-GAN 的增强能力，而不是所有算法的硬性前置条件。

## 验证数据集

验证数据集使用当前已知的任务视角，先从 `sensitivity` 开始，随后整理统一 `degradation`：

1. `sensitivity`
2. `image_quality`
3. `degradation`
4. `cooler`
5. `sifuqi`

`cwru` 可以作为额外基准或兼容性测试，但不是当前主验证数据集的核心。

### 1. sensitivity

灵敏度任务视角，来自 `Degradation` 仿真系统。关联特征：

- `avg_detectivity`
- `NETD_mK`

该数据集包含三种类型：

- `normal`
- `sensitivity_degradation`
- `coupled_severe_fault`

目标：

- 使用多维特征训练生成模型。
- 能按三类分别生成样本。
- 输出每类生成样本和标签。
- 后续可将生成样本用于 `CNN-BiGRU`、`ResNet/STFT`、`TL-Meta` 的增强验证。

### 2. image_quality

成像质量任务视角，来自同一个 `Degradation` 仿真系统。关联特征：

- `SiTF`
- `signal_gain`
- `read_noise_index`

该数据集包含五种类型：

- `normal`
- `mtf_degradation`
- `nonuniformity_degradation`
- `bad_pixel_degradation`
- `coupled_severe_fault`

需要关注：

- 类别数多于 `sensitivity`。
- 特征列与 `sensitivity` 不同。
- 生成样本是否能保持不同故障类型之间的分布差异。
- `coupled_severe_fault` 与 `sensitivity` 中的同名类别本质上是同一类耦合严重故障数据。

### 3. degradation

统一 Degradation 生成目标。后续希望直接生成五个参数和六类故障标签：

```yaml
value_columns:
  - avg_detectivity
  - NETD_mK
  - SiTF
  - signal_gain
  - read_noise_index

label_names:
  - normal
  - sensitivity_degradation
  - mtf_degradation
  - nonuniformity_degradation
  - bad_pixel_degradation
  - coupled_severe_fault
```

整理思路：

- 以同一仿真系统输出为主键来源，优先按 `unit` 与 `cycle` 对齐五个核心参数。
- 如果当前原始数据仍拆成 `sensitivity` 与 `image_quality` 两套 class files，需要先合并同一 `unit/cycle` 下的五个参数。
- `coupled_severe_fault` 只保留一份，不从两个任务视角重复拼接。
- 输出统一 class files，建议目录为 `data/degradation/`，每类一个 CSV。
- CSV 至少保留 `unit`, `cycle`, 五个核心特征列和类别信息；训练时仍可用现有 `class_files` 滑窗加载。
- 生成结果必须保存 `generated_{label}.npy`、`generated_labels.npy`、特征列顺序、归一化参数，便于还原为带标签 CSV。

### 4. cooler

用于验证退化时序生成。

需要关注：

- 可能更偏连续退化过程，而不是明确多类别分类。
- 可先采用无条件或按阶段/工况拆分训练的方式。
- 多维列如 `T_stable_K`、`t_cool_hours`、`sigma_T_K` 应作为联合生成目标。

### 5. sifuqi

用于验证伺服控制退化数据生成。

需要关注：

- 多通道控制误差、电流、温度、振动等特征。
- 可按 `normal / mild / moderate / severe` 分类型生成。
- 生成样本后可用于四等级故障诊断增强。

## 算法改造计划

### 阶段 1：打通 RVAE

目标是先完成一条稳定的端到端链路。

需要完成：

- 通用多类别多维 CSV loader。
- RVAE 支持 `(N, seq_len, num_features)` 输入输出。
- RVAE 支持可选 label conditioning。
- 生成结果按类别保存。
- 反归一化后保存可读样本。
- smoke 配置可在 `sensitivity` 上跑通。

当前状态：

- 代码已完成第一版改造。
- 尚未在 `diag` 环境中跑完整 smoke。

下一步验证命令：

```bash
conda activate diag
python data_aug/train.py --config configs/data_aug/smoke_rvae_sensitivity.yaml --stage all
```

### 阶段 2：改造 GAN

当前 `GAN` 代码更接近单维窗口生成，且 label 逻辑不是实际类别条件。

需要改造：

- 输入改为多维窗口 `(N, seq_len, num_features)`。
- 支持按类别单独训练，或支持真实 label conditioning。
- Generator 输出多维窗口。
- Discriminator 判断多维窗口真实性。
- 生成结果按类别保存。

初期可以优先实现“每类单独训练一个 GAN”的链路，降低条件 GAN 的复杂度。

### 阶段 3：改造 VAE-GAN

当前 `GAN-VAE` 是 1D Conv 结构，主要面向单通道序列。

需要改造：

- 支持 `num_features > 1` 的输入通道或特征维。
- Encoder / Generator / Discriminator 对多维时序做一致建模。
- 生成结果保持 `(num_samples, seq_len, num_features)`。
- 支持按类别单独训练，条件生成作为可选增强。

### 阶段 4：统一评估与输出

三类算法都应输出一致的核心文件：

```text
generated_samples.npy
generated_samples_denorm.npy
generated_labels.npy            # 有类别时
norm_params.json
metrics.json
loss_history.json
loss_curves.png
```

按类别生成时，额外输出：

```text
generated_{class_name}.npy
generated_{class_name}_denorm.npy
```

评估指标至少包括：

- 全局统计：mean、std、min、max。
- 每特征统计：各特征的 mean/std 差异。
- 均值偏差：mean error、mean absolute error、mean error percent、mean absolute error percent。
- 每类别统计：每类分别计算分布差异。
- 重构误差：适用于 VAE/RVAE/VAE-GAN。
- 多样性指标：适用于 GAN/VAE-GAN。
- 可视化：时域曲线、特征分布、必要时频谱图；多维特征按特征分行绘制（原始物理尺度），指标保留原始尺度。

## 与故障诊断验证的衔接

生成数据最终不是单独目的，而是为了验证对诊断任务的帮助。

后续应设计以下诊断验证：

- 原始数据训练 vs 原始 + 生成数据训练。
- 小样本工况下，生成样本是否提升诊断准确率。
- 类别不均衡场景下，生成少数类是否提升 macro-F1。
- 迁移学习场景下，生成源域或目标域样本是否有帮助。
- 小工况样本迁移中，生成数据是否能改善未充分验证的部分。

已有验证中，部分迁移学习或小样本迁移结论还不充分。数据增强链路跑通后，需要回到 `diagnosis/` 中重新组织对照实验。

建议至少记录：

- baseline 诊断结果。
- 使用每种增强算法后的诊断结果。
- 每类 precision / recall / f1。
- 混淆矩阵变化。
- 数据增强比例，例如原始:生成 = 1:1、1:2、1:5。
- 是否出现生成数据导致的过拟合或类别混淆。

## 建议执行顺序

1. 在 `diag` 环境跑通 `smoke_rvae_sensitivity.yaml`。
2. 修复 smoke 中暴露的 shape、归一化、保存路径问题。
3. 固化 `sensitivity` 的正式 RVAE 配置。
4. 设计生成样本接入诊断训练的最小流程。
5. 对 `sensitivity` 做一次 RVAE 增强诊断对照。
6. 改造并验证 `GAN` 的多维生成。
7. 改造并验证 `VAE-GAN` 的多维生成。
8. 将同一套流程推广到 `image_quality`、`cooler`、`sifuqi`。

## 当前待办

- 等 `diag` 环境准备好后运行 RVAE smoke。
- 检查生成输出形状是否符合 `(num_samples, seq_len, num_features)`。
- 检查每类生成文件是否完整。
- 检查 `metrics.json` 是否能区分类别与特征。
- 决定 GAN 和 VAE-GAN 先走“每类单独训练”还是“条件生成”。

