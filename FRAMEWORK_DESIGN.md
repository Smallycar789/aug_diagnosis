# 光电探测算法集成框架设计

> 本文档定义从 `test_pre` Jupyter 实验代码迁移到可集成 Python 项目的**简化框架**。  
> 原则：**不改算法逻辑**，只做配置化、数据加载统一化、输出规范化。

---

## 1. 设计目标

| 目标 | 说明 |
|------|------|
| 结构简单 | 无 registry / runner / adapter 多层抽象，核心入口为 `train.py` |
| 算法可插拔 | 通过 yaml 配置切换模型与数据集 |
| 输出可复现 | 每次运行落盘完整配置快照与指标 |
| 对照保留 | `test_pre/` 原样保留，作为算法对照源 |

---

## 2. 目录结构

```text
光电探测/
├── data/                              # 原始数据（不改动）
│   ├── DegradationData/               # 红外退化（故障诊断 + 表格/窗口序列）
│   ├── cooler/                        # 斯特林制冷机退化时序
│   └── sifuqi/                        # 伺服控制精度退化时序
│
├── configs/                           # 实验配置（yaml）
│   ├── diagnosis/                     # 故障诊断配置
│   │   ├── cnn_bigru_degradation.yaml
│   │   ├── resnet_sifuqi.yaml
│   │   └── tl_meta_degradation.yaml
│   └── data_aug/                      # 数据生成配置
│       ├── vae_degradation.yaml
│       ├── tvae_cooler.yaml
│       └── gan_vae_sifuqi.yaml
│
├── diagnosis/                         # 故障诊断
│   ├── data_preprocess.py             # 三类数据统一加载
│   ├── train.py                       # 训练入口
│   ├── test.py                        # 测试入口
│   ├── cnn_bigru.py                   # 1DCNN-BiGRU（来自 notebook）
│   ├── resnet.py                      # ResNet + STFT（来自 notebook）
│   └── tl_meta.py                     # SSMN 元学习（来自 notebook）
│
├── data_aug/                          # 数据生成
│   ├── train.py                       # 训练 + 生成 + 评估入口
│   ├── vae.py                         # VAE（来自 notebook）
│   ├── gan.py                         # GAN（来自 notebook）
│   ├── gan_vae.py                     # GAN-VAE 1D（来自 notebook）
│   └── tvae.py                        # TVAE（来自 notebook）
│
├── outputs/                           # 统一输出根目录
│   ├── diagnosis/
│   │   └── {dataset}/{model}/{run_id}/
│   └── data_aug/
│       └── {dataset}/{model}/{run_id}/
│
├── test_pre/                          # 初期 Jupyter 实验（保留对照）
│   └── jupyter_test/
│       ├── augmetation/               # VAE / GAN / GAN-VAE / TVAE
│       └── diagnosis/                 # 1DCNN-BiGRU / ResNet / TL-Meta
│
└── FRAMEWORK_DESIGN.md                # 本文档
```

---

## 3. 三类数据集

| 名称 | 路径 | 数据形态 | 典型用途 |
|------|------|----------|----------|
| `degradation` | `data/DegradationData/fault_diagnosis/` | 表格 + 滑窗序列（22 维特征，6 类故障） | 诊断为主；表格/窗口也可做生成 |
| `cooler` | `data/cooler/cooler_simulation_results/` | 周期退化时序（`T_stable_K`, `t_cool_hours`, `sigma_T_K`） | 序列生成、时序诊断 |
| `sifuqi` | `data/sifuqi/` | 多通道控制时序（13 列，4 档退化等级） | STFT 诊断、序列生成 |

`data_preprocess.py`（诊断）与各 `data_aug/*.py` 内的数据加载函数，按 `dataset.name` 分支读取，**归一化/滑窗公式与 notebook 保持一致**。

---

## 4. 算法清单与来源

### 4.1 数据生成（`data_aug/`）

| 模块文件 | 来源 Notebook | 说明 |
|----------|---------------|------|
| `vae.py` | `test_pre/jupyter_test/augmetation/VAE.ipynb` | 全连接 VAE，[-1,1] 归一化 |
| `gan.py` | `test_pre/jupyter_test/augmetation/GAN.ipynb` | 条件 GAN（Encoder + Generator + LSTM） |
| `gan_vae.py` | `test_pre/jupyter_test/augmetation/GAN-VAE.ipynb` | 1D Conv VAE-GAN |
| `tvae.py` | `test_pre/jupyter_test/augmetation/TVAE.ipynb` | 双向 LSTM TVAE |

### 4.2 故障诊断（`diagnosis/`）

| 模块文件 | 来源 Notebook | 说明 |
|----------|---------------|------|
| `cnn_bigru.py` | `test_pre/jupyter_test/diagnosis/1DCNN-BiGRU.ipynb` | SoftPool + 1DCNN-BiGRU + MMD 域适应 |
| `resnet.py` | `test_pre/jupyter_test/diagnosis/ResNet.ipynb` | STFT 时频图 + ResNet |
| `tl_meta.py` | `test_pre/jupyter_test/diagnosis/TL-Meta.ipynb` | SSMN 半监督元学习 |

---

## 5. 入口脚本设计

### 5.1 故障诊断 `diagnosis/train.py`

```python
# 模型名 -> 模块映射（唯一 switch 点）
MODELS = {
    "cnn_bigru": "diagnosis.cnn_bigru",
    "resnet":    "diagnosis.resnet",
    "tl_meta":   "diagnosis.tl_meta",
}

def main(config_path):
    cfg = load_yaml(config_path)
    out_dir = make_run_dir(cfg)                    # outputs/diagnosis/.../run_id
    save_config_resolved(cfg, out_dir)             # config_resolved.yaml

    train_data, val_data = load_data(cfg["dataset"])
    mod = importlib.import_module(MODELS[cfg["model"]["name"]])

    model = mod.build_model(cfg["model"])
    history = mod.train(model, train_data, val_data, cfg, out_dir)
    # train 内部：按 val 指标保存 checkpoint_best.pth、绘制 loss_curves.png
```

**命令：**

```bash
python diagnosis/train.py --config configs/diagnosis/cnn_bigru_degradation.yaml
```

### 5.2 故障诊断 `diagnosis/test.py`

```python
def main(config_path):
    cfg = load_yaml(config_path)
    out_dir = Path(cfg["output_dir"])              # 指向某次 train 的 run 目录

    test_data = load_data(cfg["dataset"], split="test")
    mod = importlib.import_module(MODELS[cfg["model"]["name"]])

    model = mod.build_model(cfg["model"])
    mod.load_checkpoint(model, out_dir / "checkpoint_best.pth")
    metrics = mod.evaluate(model, test_data, cfg, out_dir)
    # evaluate 内部：混淆矩阵、test_metrics.json
```

**命令：**

```bash
python diagnosis/test.py --config configs/diagnosis/cnn_bigru_degradation.yaml
```

### 5.3 数据生成 `data_aug/train.py`

```python
MODELS = {
    "gan":     "data_aug.gan",
    "gan_vae": "data_aug.gan_vae",
    "tvae":    "data_aug.tvae",
}

def main(config_path, stage="all"):
    # stage: train | generate | evaluate | all
    cfg = load_yaml(config_path)
    out_dir = make_run_dir(cfg)
    save_config_resolved(cfg, out_dir)

    data = load_aug_data(cfg["dataset"])
    mod = importlib.import_module(MODELS[cfg["model"]["name"]])

    if stage in ("train", "all"):
        model, history = mod.train(data, cfg, out_dir)
        save_loss_history(history, out_dir)        # loss_history.json
        # 训练过程中按 val recon loss 保存 checkpoint_best.pth

    if stage in ("generate", "all"):
        model = mod.load_checkpoint(out_dir / "checkpoint_best.pth", cfg)
        samples = mod.generate(model, data, cfg, out_dir)
        np.save(out_dir / "generated_samples.npy", samples)

    if stage in ("evaluate", "all"):
        metrics = mod.evaluate(data, out_dir, cfg)
        save_json(metrics, out_dir / "metrics.json")
        # 同时生成 signal_comparison.png、frequency_comparison.png 等
```

**命令：**

```bash
python data_aug/train.py --config configs/data_aug/tvae_cooler.yaml --stage all
```

---

## 6. 每个算法模块的标准接口

每个 `*.py` 从 notebook **原样复制**模型类、损失函数、训练循环，仅额外暴露以下函数：

### 6.1 诊断模块

```python
def build_model(cfg: dict) -> nn.Module: ...
def train(model, train_data, val_data, cfg, out_dir) -> dict: ...
def load_checkpoint(model, path) -> None: ...
def evaluate(model, test_data, cfg, out_dir) -> dict: ...
```

### 6.2 数据生成模块

```python
def build_model(cfg: dict) -> nn.Module: ...
def train(data, cfg, out_dir) -> tuple[nn.Module, dict]: ...
def load_checkpoint(path, cfg) -> nn.Module: ...
def generate(model, data, cfg, out_dir) -> np.ndarray: ...
def evaluate(data, out_dir, cfg) -> dict: ...
```

**禁止修改：** forward 逻辑、loss 公式、优化器更新步骤、MMD/元学习 Episode 采样规则等。

**允许修改：** 硬编码路径 → 从 `cfg` 读取；`plt.savefig(...)` → 保存到 `out_dir`。

---

## 7. 配置文件规范

### 7.1 公共字段

```yaml
experiment:
  name: cnn_bigru_degradation_v1   # 实验名，用于 run_id
  seed: 42
  device: auto                     # auto | cuda | cpu

output:
  root: outputs/diagnosis          # 或 outputs/data_aug
  # 实际 run 目录由脚本自动生成：
  # {root}/{dataset.name}/{model.name}/{experiment.name}_{timestamp}/
```

### 7.2 诊断配置示例

```yaml
# configs/diagnosis/cnn_bigru_degradation.yaml

experiment:
  name: cnn_bigru_degradation_v1
  seed: 42
  device: auto

output:
  root: outputs/diagnosis

dataset:
  name: degradation
  root: data/DegradationData/fault_diagnosis
  data_file: point_level_all.csv
  split_file: unit_split.csv
  feature_columns: feature_columns.json
  label_map: label_map.json
  window_size: 32
  window_stride: 16
  label_field: fault_mode

model:
  name: cnn_bigru
  hidden_dims: 64
  gru_hidden: 128
  num_layers: 2
  lr: 0.001
  epochs: 50
  batch_size: 32
  use_mmd: true
  mmd_weight: 0.1
```

### 7.3 数据生成配置示例

```yaml
# configs/data_aug/tvae_cooler.yaml

experiment:
  name: tvae_cooler_v1
  seed: 42
  device: auto

output:
  root: outputs/data_aug

dataset:
  name: cooler
  root: data/cooler
  csv: cooler_simulation_results/all_simulation.csv
  value_columns: [T_stable_K, t_cool_hours, sigma_T_K]
  seq_len: 128
  stride: 32
  normalize: minus1_1          # 与 notebook 一致
  sample_rate: 1.0             # FFT 用；CWRU 类数据可设 12000

model:
  name: tvae
  latent_dim: 64
  hidden_dim: 128
  num_layers: 1
  epochs: 100
  batch_size: 32
  lr: 0.001
  beta_start: 0.0
  beta_end: 0.04
  num_generate: 500            # 生成样本数
```

---

## 8. 输出目录规范

### 8.1 目录命名

```text
outputs/
├── diagnosis/
│   └── {dataset}/{model}/{experiment.name}_{YYYYMMDD_HHMMSS}/
└── data_aug/
    └── {dataset}/{model}/{experiment.name}_{YYYYMMDD_HHMMSS}/
```

示例：

```text
outputs/diagnosis/degradation/cnn_bigru/cnn_bigru_degradation_v1_20260525_143022/
outputs/data_aug/cooler/tvae/tvae_cooler_v1_20260525_150000/
```

### 8.2 故障诊断输出（单次 train + test）

| 文件 | 产生阶段 | 说明 |
|------|----------|------|
| `config_resolved.yaml` | train 开始 | 合并默认值后的完整配置快照（含实际 `output_dir` 绝对路径） |
| `checkpoint_best.pth` | train | **最佳模型**权重（按验证集主指标，如 val_acc / val_loss） |
| `loss_history.json` | train | 每 epoch 的 loss / acc 序列，供绘图与复现 |
| `loss_curves.png` | train | 训练/验证 loss（及 acc）曲线图 |
| `confusion_matrix.png` | test | **该轮次**测试集混淆矩阵热力图 |
| `confusion_matrix.csv` | test | 混淆矩阵数值（可选，便于程序读取） |
| `test_metrics.json` | test | 测试指标汇总（见 8.4） |

可选扩展（部分算法 notebook 已有，迁移时按需保留）：

| 文件 | 算法 | 说明 |
|------|------|------|
| `classification_report.txt` | 全部诊断 | sklearn classification_report |
| `target_confusion_matrix.png` | cnn_bigru | 目标域混淆矩阵（域适应场景） |
| `training_curves.png` | tl_meta | loss + val accuracy 双轴图 |
| `tsne_visualization.png` | tl_meta | 特征 t-SNE（可选） |

### 8.3 数据生成输出（train + generate + evaluate）

| 文件 | 产生阶段 | 说明 |
|------|----------|------|
| `config_resolved.yaml` | train 开始 | 完整配置快照 |
| `checkpoint_best.pth` | train | **最佳模型**（通常按 val recon loss 最低） |
| `loss_history.json` | train | 训练 loss 序列 |
| `loss_curves.png` | train | total / recon / kl（及 GAN 的 d_loss、g_loss 等）曲线 |
| `generated_samples.npy` | generate | 生成样本数组，形状与 notebook 一致（如 `(N, seq_len, 1)` 或 `(N, input_dim)`） |
| `norm_params.json` | generate | 归一化参数 `min/max` 或 `mean/std`，用于反归一化 |
| `metrics.json` | evaluate | 生成质量指标（见 8.5） |
| `signal_comparison.png` | evaluate | 原始 vs 生成时域波形对比（多子图） |
| `frequency_comparison.png` | evaluate | 原始 vs 生成频谱对比（FFT 幅度谱） |

可选扩展：

| 文件 | 说明 |
|------|------|
| `vae_comprehensive_results.png` | VAE 综合结果图（来自 VAE.ipynb） |
| `generated_samples_denorm.npy` | 反归一化后的生成样本 |

### 8.4 `test_metrics.json` 结构（诊断）

```json
{
  "experiment": "cnn_bigru_degradation_v1",
  "dataset": "degradation",
  "model": "cnn_bigru",
  "checkpoint": "checkpoint_best.pth",
  "best_epoch": 42,
  "best_val_metric": 0.9823,
  "test": {
    "accuracy": 0.9750,
    "precision_macro": 0.9712,
    "recall_macro": 0.9688,
    "f1_macro": 0.9699,
    "cohen_kappa": 0.9650
  },
  "per_class": {
    "normal": {"precision": 0.98, "recall": 0.97, "f1": 0.975, "support": 120},
    "sensitivity_degradation": {"precision": 0.96, "recall": 0.98, "f1": 0.97, "support": 115}
  },
  "confusion_matrix": [[...], [...]],
  "label_names": ["normal", "sensitivity_degradation", "..."]
}
```

字段与 notebook 中 `sklearn.metrics` 输出对齐；`tl_meta` 可增加 `n_way`、`n_shot` 等元学习参数。

### 8.5 `metrics.json` 结构（数据生成）

参考 `test_pre` 中 VAE / TVAE / GAN-VAE notebook 的评估逻辑：

```json
{
  "experiment": "tvae_cooler_v1",
  "dataset": "cooler",
  "model": "tvae",
  "num_generated": 500,
  "statistics": {
    "original": {"mean": -0.001510, "std": 0.236105},
    "generated": {"mean": -0.000526, "std": 0.064864},
    "mean_diff": 0.000984,
    "std_diff": 0.171241
  },
  "distribution": {
    "js_divergence": 0.042315,
    "correlation": 0.1234
  },
  "reconstruction_diagnostic": {
    "mse_teacher_forced": 0.002550,
    "mse_open_loop": 0.059359
  },
  "spectrum": {
    "sample_rate": 12000,
    "dominant_freq_original_hz": 123.4,
    "dominant_freq_generated_hz": 118.7
  },
  "training_best": {
    "best_epoch": 85,
    "best_val_recon_loss": 0.00233
  }
}
```

**各指标来源（notebook 对应）：**

| 指标 | 来源 Notebook | 计算方式 |
|------|---------------|----------|
| mean / std / mean_diff / std_diff | VAE, TVAE, GAN-VAE | 展平后 numpy 统计 |
| js_divergence | VAE.ipynb `evaluate_samples` | 直方图近似 JS 散度 |
| correlation | VAE / TVAE | `np.corrcoef`（TVAE 中 concatenated correlation 可选记录） |
| mse_teacher_forced / mse_open_loop | TVAE.ipynb `validate_and_visualize` | 同窗口 teacher forcing vs open-loop 诊断 |
| FFT 频谱图 | TVAE, GAN-VAE | `np.fft.fft` + `np.fft.fftfreq`，保存为 `frequency_comparison.png` |
| 时域对比图 | 全部 aug notebook | 多子图 `signal_comparison.png` |

算法模块的 `evaluate()` 应**直接复用 notebook 中的评估函数**，仅将 print 结果写入 `metrics.json`、将 plt 保存到 `out_dir`。

### 8.6 `loss_history.json` 结构

**诊断：**

```json
{
  "train_loss": [0.85, 0.62, "..."],
  "val_loss": [0.78, 0.55, "..."],
  "val_accuracy": [0.72, 0.81, "..."],
  "epochs": 50
}
```

**数据生成：**

```json
{
  "total_loss": ["..."],
  "recon_loss": ["..."],
  "kl_loss": ["..."],
  "d_loss": ["..."],
  "g_loss": ["..."],
  "beta": ["..."],
  "epochs": 100
}
```

字段按各算法 notebook 实际记录的 history 键名保留，不做统一删减。

---

## 9. 最佳模型保存策略

| 任务 | 主指标 | 保存时机 |
|------|--------|----------|
| 诊断 cnn_bigru / resnet | val_accuracy ↑ | 验证 acc 创新高时覆盖 `checkpoint_best.pth` |
| 诊断 tl_meta | val_accuracy ↑ | 验证 episode acc 创新高时保存 |
| 数据生成 VAE / TVAE / GAN-VAE | val_recon_loss ↓ | 验证重建 loss 创新低时保存 |
| 数据生成 GAN | g_loss + 可选 recon | 按 notebook 原有逻辑（如固定 epoch 或 recon 稳定后） |

`checkpoint_best.pth` 建议同时保存：

```python
{
    "epoch": int,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,   # 可选
    "best_metric": float,
    "metric_name": "val_accuracy" | "val_recon_loss",
}
```

---

## 10. `data_preprocess.py` 职责

```python
def load_data(dataset_cfg, split="train"):
    """
    split: train | val | test
    返回格式由各诊断算法 train/evaluate 约定（与 notebook 一致）：
      - cnn_bigru: (X, y) numpy 或 DataLoader
      - resnet: list of (PIL.Image, label)
      - tl_meta: Dataset 对象（支持 Episode 采样）
    """
    name = dataset_cfg["name"]
    if name == "degradation":
        return _load_degradation(dataset_cfg, split)
    elif name == "cooler":
        return _load_cooler(dataset_cfg, split)
    elif name == "sifuqi":
        return _load_sifuqi(dataset_cfg, split)
    else:
        raise ValueError(f"Unknown dataset: {name}")
```

**degradation 要点：**

- 读取 `point_level_all.csv`、`unit_split.csv`、`feature_columns.json`、`label_map.json`
- 按 unit 滑窗：`window_size=32`, `window_stride=16`（与 `dataset_summary.json` 一致）
- 标签字段默认 `fault_mode`

**cooler 要点：**

- 读取 `all_simulation.csv` 或指定 group
- 列：`T_stable_K`, `t_cool_hours`, `sigma_T_K`
- 可按 `group_id` 划分 train/val/test

**sifuqi 要点：**

- 读取 `servo_normal.csv` ~ `servo_severe.csv`
- 标签来自 `label` 列或文件名映射
- 特征列见 `data/sifuqi/README.md`

---

## 11. 迁移步骤（逐算法）

对每个算法重复以下步骤，**一次只迁一个**：

1. 从对应 notebook 复制模型类、损失、训练/评估函数到目标 `*.py`
2. 将硬编码路径替换为 `cfg["dataset"]` / `cfg["model"]` / `out_dir`
3. 实现 `build_model` / `train` / `evaluate`（及生成的 `generate`）
4. 编写对应 `configs/**/*.yaml`
5. 运行并与 notebook 同配置对比 loss / acc / 生成指标
6. 确认 `outputs/` 下文件齐全

**迁移对照表：**

| 步骤 | 诊断 | 数据生成 |
|------|------|----------|
| 1 | 复制 TL-Meta / 1DCNN / ResNet 代码 | 复制 VAE / GAN / TVAE 代码 |
| 2 | 改 `diagnosis/train.py` 的 MODELS | 改 `data_aug/train.py` 的 MODELS |
| 3 | 跑 `train.py` + `test.py` | 跑 `train.py --stage all` |
| 4 | 检查 8.2 输出 | 检查 8.3 输出 |

---

## 12. 建议实施顺序

1. 创建 `configs/` 目录结构与示例 yaml
2. 实现 `diagnosis/train.py`、`diagnosis/test.py` 骨架（含 `make_run_dir`、`save_config_resolved`）
3. 实现 `data_aug/train.py` 骨架
4. 实现 `diagnosis/data_preprocess.py`（三类 loader）
5. **模板算法 1**：`diagnosis/cnn_bigru.py` + `degradation` 配置
6. **模板算法 2**：`data_aug/tvae.py` + `cooler` 配置
7. 其余算法按模板逐个迁入

---

## 13. 依赖与环境

参考 notebook 常用库：

```
torch
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
Pillow
tqdm
pyyaml
```

可按项目需要增加 `requirements.txt`，版本与 `data/sifuqi/environment.yml` 对齐。

---

## 14. 与外部系统集成（预留）

后续 ATE 或其他系统调用时，只需：

1. 传入 `--config` 路径（或环境变量 `EXPERIMENT_CONFIG`）
2. 读取 `outputs/.../config_resolved.yaml` 复现实验
3. 加载 `checkpoint_best.pth` 做推理
4. 读取 `test_metrics.json` 或 `metrics.json` 获取结果

无需改动算法模块内部逻辑。

---

## 15. 附录：单次运行产出清单速查

### 诊断（train + test 完整流程）

```
outputs/diagnosis/{dataset}/{model}/{run_id}/
├── config_resolved.yaml
├── checkpoint_best.pth          ← 最佳模型
├── loss_history.json
├── loss_curves.png                ← loss 曲线
├── confusion_matrix.png           ← 该轮次混淆矩阵
├── confusion_matrix.csv           ← 可选
└── test_metrics.json
```

### 数据生成（--stage all）

```
outputs/data_aug/{dataset}/{model}/{run_id}/
├── config_resolved.yaml
├── checkpoint_best.pth            ← 最佳模型
├── loss_history.json
├── loss_curves.png
├── generated_samples.npy
├── norm_params.json               ← 可选
├── metrics.json                   ← 统计 + JS + FFT 等
├── signal_comparison.png          ← 时域对比
└── frequency_comparison.png       ← 傅里叶频谱对比
```

---

*文档版本：v1.0 | 更新日期：2026-05-25*
