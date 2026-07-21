# 数据增强模块

## 算法

| 模块 | 配置 model.name | 来源 Notebook |
|------|-----------------|---------------|
| `vae.py` | `vae` | VAE.ipynb |
| `gan.py` | `gan` | GAN.ipynb |
| `gan_vae.py` | `gan_vae` | GAN-VAE.ipynb |
| `tvae.py` | `tvae` | TVAE.ipynb |

## 运行方式

使用 conda 环境 `envv`：

```bash
conda activate envv
cd /path/to/aug_diagnosis

# 完整流程：训练 + 生成 + 评估
python data_aug/train.py --config configs/data_aug/gan_cooler.yaml --stage all

# 分阶段（需指定同一 output 目录）
python data_aug/train.py --config configs/data_aug/tvae_cooler.yaml --stage train
python data_aug/train.py --config configs/data_aug/tvae_cooler.yaml --stage generate --output-dir outputs/data_aug/cooler/tvae/xxx
```

当前 `configs/data_aug/` 保留四个数据集 × 三种算法的正式配置（共 12 个）：

| 配置文件 | 数据集 | 算法 |
|----------|--------|------|
| `gan_image_quality.yaml` | image_quality | gan |
| `tvae_image_quality.yaml` | image_quality | tvae |
| `gan_vae_image_quality.yaml` | image_quality | gan_vae |
| `gan_sensitivity.yaml` | sensitivity | gan |
| `tvae_sensitivity.yaml` | sensitivity | tvae |
| `gan_vae_sensitivity.yaml` | sensitivity | gan_vae |
| `gan_cooler.yaml` | cooler | gan |
| `tvae_cooler.yaml` | cooler | tvae |
| `gan_vae_cooler.yaml` | cooler | gan_vae |
| `gan_sifuqi.yaml` | sifuqi | gan |
| `tvae_sifuqi.yaml` | sifuqi | tvae |
| `gan_vae_sifuqi.yaml` | sifuqi | gan_vae |

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
├── signal_comparison.png          # gan / gan_vae / tvae
├── frequency_comparison.png       # gan / gan_vae / tvae
└── vae_comprehensive_results.png  # vae
```

## 数据集配置

`dataset.name` 支持以下数据：

- `cooler` — `data/cooler/all_simulation.csv`（10 组，`time_hours` + `T_stable_K` / `t_cool_s` / `sigma_T_K`；正常 &lt;2000h、故障 &gt;6000h 分段滑窗，按类生成）
- `sifuqi` — `data/sifuqi/servo_accuracy.csv`（12 组宽表 `group_01`~`group_12`；正常 &lt;1000h、跟踪故障 &gt;6000h 分段滑窗，按类生成）
- `cwru` — CWRU CSV
- `sensitivity` / 其他 `class_files` 数据 — 多类别 CSV 滑窗，支持 TVAE 条件生成
- `degradation` / `Degradation` — 红外退化仿真系统的统一生成视角，可合并灵敏度与成像质量相关指标和故障类型

### 性能退化指标与特征参数

| 性能退化指标 | 简称 | 对应特征参数 |
|--------------|------|--------------|
| 温控性能退化 | `cooler` | 冷端温度、制冷时间、温度波动水平 |
| 灵敏度下降 | `sensitivity` | 平均探测率、噪声等效温差 |
| 图像质量下降 | `image_quality` | MTF 指标、图像均一性、坏点率 |
| 伺服跟踪精度下降 | `sifuqi` | 伺服运动精度（deg） |

### Degradation 仿真系统

`sensitivity` 与 `image_quality` 来自同一个红外退化仿真系统 `Degradation`。两者整体仿真参数一致，只是关注的特征参数与故障类型不同。

| 任务视角 | 关联特征列 | 故障类型 |
|----------|------------|----------|
| `sensitivity` 灵敏度 | `avg_detectivity`, `NETD_mK` | `normal`, `sensitivity_degradation`, `coupled_severe_fault` |
| `image_quality` 成像质量 | `SiTF`, `signal_gain`, `read_noise_index` | `normal`, `mtf_degradation`, `nonuniformity_degradation`, `bad_pixel_degradation`, `coupled_severe_fault` |
| `degradation` 统一生成 | `avg_detectivity`, `NETD_mK`, `SiTF`, `signal_gain`, `read_noise_index` | `normal`, `sensitivity_degradation`, `mtf_degradation`, `nonuniformity_degradation`, `bad_pixel_degradation`, `coupled_severe_fault` |

`coupled_severe_fault` 在两个任务视角中本质上对应同一类耦合严重故障数据，整理统一 `degradation` 数据时应作为一个类别处理，避免重复计入。

#### Degradation 数据整理思路

1. 以同一仿真系统输出为基准，整理一套统一 CSV/class-files 数据目录，例如 `data/degradation/`。
2. 固定五个核心特征列顺序：`avg_detectivity`, `NETD_mK`, `SiTF`, `signal_gain`, `read_noise_index`。
3. 将类别统一为六类：`normal`, `sensitivity_degradation`, `mtf_degradation`, `nonuniformity_degradation`, `bad_pixel_degradation`, `coupled_severe_fault`。
4. 如果原始文件仍按 `sensitivity` 和 `image_quality` 拆分，需要按仿真单元/周期对齐后合并五个特征；`coupled_severe_fault` 只保留一份。
5. 统一保留 `unit` 与 `cycle` 字段，后续仍按 `unit` 分组、按 `cycle` 排序后滑窗。
6. 生成模型输出应保存五维窗口、类别标签、特征列顺序和归一化参数，便于还原为带标签的 Degradation 增强样本。

详见 `FRAMEWORK_DESIGN.md`。
