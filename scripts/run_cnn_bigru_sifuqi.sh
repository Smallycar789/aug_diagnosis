#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

run_train() {
  if [[ -n "${PYTHON:-}" ]]; then
    "${PYTHON}" diagnosis/train.py --config configs/diagnosis/cnn_bigru_sifuqi.yaml
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    "${CONDA_PREFIX}/bin/python" diagnosis/train.py --config configs/diagnosis/cnn_bigru_sifuqi.yaml
  elif [[ -x "${HOME}/miniconda3/envs/envv/bin/python" ]]; then
    "${HOME}/miniconda3/envs/envv/bin/python" diagnosis/train.py --config configs/diagnosis/cnn_bigru_sifuqi.yaml
  elif [[ -x "${HOME}/anaconda3/envs/envv/bin/python" ]]; then
    "${HOME}/anaconda3/envs/envv/bin/python" diagnosis/train.py --config configs/diagnosis/cnn_bigru_sifuqi.yaml
  elif command -v conda >/dev/null 2>&1; then
    conda run -n envv python diagnosis/train.py --config configs/diagnosis/cnn_bigru_sifuqi.yaml
  elif command -v python3 >/dev/null 2>&1; then
    python3 diagnosis/train.py --config configs/diagnosis/cnn_bigru_sifuqi.yaml
  elif command -v python >/dev/null 2>&1; then
    python diagnosis/train.py --config configs/diagnosis/cnn_bigru_sifuqi.yaml
  else
    echo "Python not found. Run: conda activate envv  (or export PYTHON=/path/to/python)" >&2
    exit 1
  fi
}

echo "=== CNN-BiGRU fault diagnosis on sifuqi ==="
echo "Project: ${PROJECT_ROOT}"
run_train
echo "Training finished."
