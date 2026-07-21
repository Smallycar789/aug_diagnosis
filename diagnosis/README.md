# 故障诊断模块

## Python 文件说明

| 文件 | 作用 |
|------|------|
| `__init__.py` | 故障诊断子包入口，标识本目录为从 Jupyter Notebook 迁移而来的诊断算法模块。 |
| `train.py` | 统一训练入口，根据 YAML 配置调度 `cnn_bigru` / `resnet` / `tl_meta` 并完成训练与 checkpoint 保存。 |
| `test.py` | 统一测试入口，加载训练输出目录中的模型，在测试集上评估并生成混淆矩阵与指标 JSON。 |
| `io_utils.py` | 负责读取 YAML 配置、解析路径、选择设备、创建带时间戳的输出目录及保存训练元数据。 |
| `common.py` | 提供滑窗切分、归一化（per_window / global）、训练/验证/测试划分等通用数据处理工具函数。 |
| `data_preprocess.py` | 按数据集类型（CWRU、degradation、cooler、sifuqi 等）统一加载数据并封装为 `DiagnosisDataBundle`。 |
| `simulation_load.py` | 专用于 cooler / sifuqi 仿真 CSV 的读取与时间阈值滑窗（正常段 / 故障段分段采样）。 |
| `cnn_bigru.py` | 实现 1D-CNN + BiGRU 分类器及 MMD 域自适应训练与测试流程（对应 `model.name: cnn_bigru`）。 |
| `resnet.py` | 实现基于 STFT 时频图与 ResNet18 的迁移学习故障分类（对应 `model.name: resnet`）。 |
| `tl_meta.py` | 实现 SSMN 半监督元学习小样本故障诊断（对应 `model.name: tl_meta`）。 |

## 算法

| 模块 | model.name | 来源 Notebook |
|------|------------|---------------|
| `cnn_bigru.py` | `cnn_bigru` | 1DCNN-BiGRU.ipynb |
| `resnet.py` | `resnet` | ResNet.ipynb |
| `tl_meta.py` | `tl_meta` | TL-Meta.ipynb |

## 运行（conda envv）

```bash
conda activate envv
cd /path/to/aug_diagnosis

# 训练（简写：自动解析 configs/diagnosis/{algorithm}_{dataset}.yaml）
python diagnosis/train.py --dataset sensitivity --algorithm cnn_bigru

# 或显式指定配置
python diagnosis/train.py --config configs/diagnosis/cnn_bigru_sensitivity.yaml

# 测试（指定训练输出目录）
python diagnosis/test.py --config configs/diagnosis/cnn_bigru_sensitivity.yaml --output-dir outputs/diagnosis/sensitivity/cnn_bigru/xxx

# 其他数据集示例
python diagnosis/train.py --dataset image_quality --algorithm resnet
python diagnosis/train.py --dataset cooler --algorithm tl_meta
python diagnosis/train.py --config configs/diagnosis/cnn_bigru_sifuqi.yaml
```

## 输出

```
outputs/diagnosis/{dataset}/{model}/{run_id}/
├── config_resolved.yaml
├── checkpoint_best.pth
├── loss_history.json
├── loss_curves.png
├── confusion_matrix.png      # test.py 生成
└── test_metrics.json         # test.py 生成
```

## 数据集

当前 `configs/diagnosis/` 提供四个数据集 × 三种算法的正式配置：

- `image_quality` — 图像质量退化（5 类）
- `sensitivity` — 灵敏度退化（3 类）
- `cooler` — 斯特林制冷机时序（2 类）
- `sifuqi` — 伺服跟踪精度退化（2 类）

参数均在 `configs/diagnosis/{algorithm}_{dataset}.yaml` 中配置。
