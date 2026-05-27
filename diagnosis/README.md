# 故障诊断模块

从 `test_pre/jupyter_test/diagnosis/` 迁移的三个算法，核心逻辑与 notebook 保持一致。

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
