#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/data_aug/rvae_sensitivity.yaml"
RUN_ID="rvae_sensitivity_matvae_v2_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="outputs/data_aug/sensitivity/rvae/${RUN_ID}"
mkdir -p "$RUN_DIR"
export RUN_DIR

LOG_FILE="${RUN_DIR}/run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Project: $PROJECT_ROOT"
echo "Config:  $CONFIG"
echo "Output:  $RUN_DIR"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # Common non-interactive shell path.
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found. Please install conda or update this script." >&2
  exit 1
fi

conda activate diag
echo "Python: $(python --version)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"

python data_aug/train.py --config "$CONFIG" --stage all --output-dir "$RUN_DIR"

python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

run_dir = Path(os.environ["RUN_DIR"])
summary: dict = {"run_dir": str(run_dir)}

metrics_path = run_dir / "metrics.json"
if metrics_path.exists():
    with open(metrics_path, encoding="utf-8") as f:
        summary["metrics"] = json.load(f)

for name in [
    "generated_samples.npy",
    "generated_samples_denorm.npy",
    "generated_labels.npy",
]:
    path = run_dir / name
    if path.exists():
        arr = np.load(path)
        summary[name] = {
            "shape": list(arr.shape),
            "mean": float(np.mean(arr)) if arr.size else None,
            "std": float(np.std(arr)) if arr.size else None,
            "min": float(np.min(arr)) if arr.size else None,
            "max": float(np.max(arr)) if arr.size else None,
        }

label_names = summary.get("metrics", {}).get("label_names", [])
for label_name in label_names:
    path = run_dir / f"generated_{label_name}_denorm.npy"
    if path.exists():
        arr = np.load(path)
        summary[f"generated_{label_name}_denorm.npy"] = {
            "shape": list(arr.shape),
            "feature_mean": np.mean(arr, axis=(0, 1)).tolist() if arr.ndim == 3 else None,
            "feature_std": np.std(arr, axis=(0, 1)).tolist() if arr.ndim == 3 else None,
        }

with open(run_dir / "generation_quality_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

lines = [
    f"Run dir: {run_dir}",
    "Generated quality summary:",
]
for key, value in summary.items():
    if key == "metrics":
        continue
    lines.append(f"- {key}: {value}")

(run_dir / "generation_quality_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Quality summary saved: {run_dir / 'generation_quality_summary.json'}")
PY

echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
