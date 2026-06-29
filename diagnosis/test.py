"""Unified testing entry for fault diagnosis algorithms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnosis import cnn_bigru, resnet, tl_meta
from diagnosis.data_preprocess import load_data
from diagnosis.io_utils import get_device, load_config, save_json, set_seed

MODELS = {
    "cnn_bigru": cnn_bigru,
    "resnet": resnet,
    "tl_meta": tl_meta,
}

# 故障类别英文 slug → 中文（平台展示）
LABEL_NAME_ZH = {
    "normal": "正常",
    "temperature_control_fault": "温控故障",
    "tracking_fault": "跟踪故障",
    "sensitivity_degradation": "灵敏度退化",
    "coupled_severe_fault": "耦合严重故障",
    "mtf_degradation": "MTF退化",
    "nonuniformity_degradation": "非均匀性退化",
    "bad_pixel_degradation": "坏元退化",
    "mild": "轻度故障",
    "moderate": "中度故障",
    "severe": "严重故障",
}

# 测试指标英文字段 → 中文（仅下列几项）
METRIC_KEY_ZH = {
    "accuracy": "诊断准确率",
    "accuracy_percent": "诊断准确率(%)",
    "precision_macro": "宏平均精确率",
    "recall_macro": "宏平均召回率",
    "f1_macro": "平均F1分数",
}


def localize_test_metrics(metrics: dict) -> dict:
    """后加工：将 test_metrics 中部分字段名与故障类别名改为中文。"""
    out = dict(metrics)
    for en_key, zh_key in METRIC_KEY_ZH.items():
        if en_key in out:
            out[zh_key] = out.pop(en_key)
    if isinstance(out.get("label_names"), list):
        out["label_names"] = [LABEL_NAME_ZH.get(name, name) for name in out["label_names"]]
    return out


def _load_meta(out_dir: Path) -> dict:
    meta_path = out_dir / "train_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _resolve_test_config(config_path: str, out_dir: Path) -> dict:
    resolved_cfg_path = out_dir / "config_resolved.yaml"
    if resolved_cfg_path.exists():
        cfg = load_config(resolved_cfg_path)
        print(f"Using resolved config: {resolved_cfg_path}")
        return cfg

    cfg = load_config(config_path)
    print(f"Using config: {config_path}")
    return cfg


def main(config_path: str, output_dir: str | None = None, split: str = "test") -> None:
    if output_dir:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = None

    if out_dir is not None:
        out_dir = out_dir.resolve()
        cfg = _resolve_test_config(config_path, out_dir)
    else:
        cfg = load_config(config_path)

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    if out_dir is None:
        if cfg.get("output_dir"):
            out_dir = Path(cfg["output_dir"])
            if not out_dir.is_absolute():
                out_dir = PROJECT_ROOT / out_dir
            out_dir = out_dir.resolve()
            cfg = _resolve_test_config(config_path, out_dir)
        else:
            raise ValueError("Provide --output-dir or set output_dir in config to the trained run directory.")
    ckpt = out_dir / "checkpoint_best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model_name = cfg["model"]["name"]
    mod = MODELS[model_name]
    device = get_device(cfg)
    print(f"Model: {model_name} | Device: {device} | Output: {out_dir}")

    model = mod.load_checkpoint(ckpt, cfg)

    if model_name == "cnn_bigru":
        bundle = load_data(cfg["dataset"], model_name, seed=seed)
        metrics = mod.evaluate(model, bundle, cfg, out_dir, split=split)
    elif model_name == "resnet":
        metrics = mod.evaluate(model, None, cfg, out_dir, split=split)
    else:
        metrics = mod.evaluate(model, None, cfg, out_dir, split=split)

    print(f"Test metrics saved: {out_dir / 'test_metrics.json'}")
    print(f"Accuracy: {metrics.get('accuracy', metrics.get('accuracy_percent'))}")

    metrics_zh = localize_test_metrics(metrics)
    save_json(metrics_zh, out_dir / "test_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fault diagnosis testing")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--output-dir", required=True, help="Trained run output directory")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args()
    main(args.config, args.output_dir, args.split)
