# 数据增强模块

## 算法

| 模块 | 配置 model.name | 来源 Notebook |
|------|-----------------|---------------|
| `vae.py` | `vae` | VAE.ipynb |
| `gan.py` | `gan` | GAN.ipynb |
| `gan_vae.py` | `gan_vae` | GAN-VAE.ipynb |
| `rvae.py` | `rvae` | RVAE.ipynb |

## 运行方式

使用 conda 环境 `envv`：

```powershell
conda activate envv
cd C:\Users\86188\Desktop\ATE\光电探测

# 完整流程：训练 + 生成 + 评估
python data_aug/train.py --config configs/data_aug/gan_cooler.yaml --stage all

# 分阶段（需指定同一 output 目录）
python data_aug/train.py --config configs/data_aug/rvae_cooler.yaml --stage train
python data_aug/train.py --config configs/data_aug/rvae_cooler.yaml --stage generate --output-dir outputs/data_aug/cooler/rvae/rvae_cooler_v1_YYYYMMDD_HHMMSS
```

一键冒烟测试：

```powershell
powershell -File scripts/run_data_aug_smoke.ps1
```

快速冒烟测试（2 epoch）：

```powershell
python data_aug/train.py --config configs/data_aug/smoke_vae_cooler.yaml --stage all
python data_aug/train.py --config configs/data_aug/smoke_gan_cooler.yaml --stage all
python data_aug/train.py --config configs/data_aug/smoke_gan_vae_sifuqi.yaml --stage all
python data_aug/train.py --config configs/data_aug/smoke_rvae_cooler.yaml --stage all
```

## 输出目录

```
outputs/data_aug/{dataset}/{model}/{experiment}_{timestamp}/
├── config_resolved.yaml
├── checkpoint_best.pth
├── loss_history.json
├── loss_curves.png
├── generated_samples.npy
├── norm_params.json
├── metrics.json
├── signal_comparison.png          # gan / gan_vae / rvae
├── frequency_comparison.png       # gan / gan_vae / rvae
└── vae_comprehensive_results.png  # vae
```

## 数据集配置

`dataset.name` 支持三类数据：

- `degradation` — `data/DegradationData/fault_diagnosis/`
- `cooler` — `data/cooler/cooler_simulation_results/`
- `sifuqi` — `data/sifuqi/servo_*.csv`

详见 `FRAMEWORK_DESIGN.md`。
