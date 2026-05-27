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
from diagnosis.io_utils import get_device, load_config, set_seed

MODELS = {
    "cnn_bigru": cnn_bigru,
    "resnet": resnet,
    "tl_meta": tl_meta,
}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fault diagnosis testing")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--output-dir", required=True, help="Trained run output directory")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args()
    main(args.config, args.output_dir, args.split)
