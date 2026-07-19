"""Config loading and standardized output helpers for diagnosis."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    with open(resolve_path(config_path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_dir(cfg: dict[str, Any]) -> Path:
    exp = cfg.get("experiment", {})
    dataset_name = cfg["dataset"]["name"]
    model_name = cfg["model"]["name"]
    exp_name = exp.get("name", "run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = resolve_path(cfg.get("output", {}).get("root", "outputs/diagnosis"))
    run_dir = root / dataset_name / model_name / f"{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_resolved(cfg: dict[str, Any], out_dir: Path) -> Path:
    resolved = copy.deepcopy(cfg)
    resolved["output_dir"] = str(out_dir.resolve())
    path = out_dir / "config_resolved.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, allow_unicode=True, sort_keys=False)
    return path


def save_json(data: dict[str, Any], path: Path) -> None:
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_default)


def save_loss_history(history: dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "loss_history.json"
    save_json(history, path)
    return path


def save_checkpoint_best(
    out_dir: Path,
    state: dict[str, Any],
    epoch: int,
    best_metric: float,
    metric_name: str = "val_accuracy",
) -> Path:
    import torch

    path = out_dir / "checkpoint_best.pth"
    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "metric_name": metric_name,
        **state,
    }
    torch.save(payload, path)
    return path


def get_device(cfg: dict[str, Any]):
    import torch

    device_cfg = cfg.get("experiment", {}).get("device", "auto")
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
