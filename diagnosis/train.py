"""Unified training entry for fault diagnosis algorithms."""

from __future__ import annotations

from typing import Optional, Any

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnosis import cnn_bigru, resnet, tl_meta
from diagnosis.data_preprocess import load_data
from diagnosis.io_utils import get_device, load_config, make_run_dir, save_config_resolved, set_seed

MODELS = {
    "cnn_bigru": cnn_bigru,
    "resnet": resnet,
    "tl_meta": tl_meta,
}


def _resolve_out_dir(cfg: dict, output_dir: Optional[str] = None) -> Path:
    if output_dir:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config_resolved({**cfg, "output_dir": str(out_dir.resolve())}, out_dir)
        return out_dir.resolve()
    if cfg.get("output_dir"):
        out_dir = Path(cfg["output_dir"])
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir.resolve()
    out_dir = make_run_dir(cfg)
    save_config_resolved({**cfg, "output_dir": str(out_dir.resolve())}, out_dir)
    return out_dir


def main(config_path: str, output_dir: Optional[str] = None) -> Path:
    cfg = load_config(config_path)
    exp = cfg.get("experiment", {})
    seed = int(exp.get("seed", 42))
    set_seed(seed)

    out_dir = _resolve_out_dir(cfg, output_dir)
    cfg["output_dir"] = str(out_dir)

    model_name = cfg["model"]["name"]
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    mod = MODELS[model_name]
    device = get_device(cfg)
    print(f"Model: {model_name} | Dataset: {cfg['dataset']['name']} | Device: {device}")
    print(f"Output: {out_dir}")

    bundle = load_data(cfg["dataset"], model_name, seed=seed)

    model, meta = mod.train(bundle, cfg, out_dir)
    print(f"Training done. Best checkpoint: {out_dir / 'checkpoint_best.pth'}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fault diagnosis training")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument("--output-dir", default=None, help="Optional output directory")
    args = parser.parse_args()
    main(args.config, args.output_dir)
