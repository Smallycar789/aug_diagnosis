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
cd /path/to/光电探测

# 训练
python diagnosis/train.py --config configs/diagnosis/cnn_bigru_cwru.yaml

# 测试（指定训练输出目录）
python diagnosis/test.py --config configs/diagnosis/cnn_bigru_cwru.yaml --output-dir outputs/diagnosis/cwru_csv/cnn_bigru/xxx

# 伺服器（sifuqi）四等级故障诊断 — 一键训练
chmod +x scripts/run_cnn_bigru_sifuqi.sh   # 首次
./scripts/run_cnn_bigru_sifuqi.sh
# 或
bash scripts/run_cnn_bigru_sifuqi.sh
# 或
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

- `cwru_csv` — 使用项目根目录 CWRU CSV（与 notebook 一致）
- `degradation` — 红外退化 fault_diagnosis 滑窗数据
- `cooler` / `sifuqi` — 时序滑窗（用于后续验证）

参数均在 `configs/diagnosis/*.yaml` 中配置。
