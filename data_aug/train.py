"""Unified entry for data augmentation algorithms."""

from __future__ import annotations

from typing import Optional, Any

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_aug import gan, gan_vae, tvae, vae
from data_aug.data_load import load_data
from data_aug.io_utils import (
    get_device,
    load_config,
    make_run_dir,
    save_config_resolved,
    set_seed,
)

MODELS = {
    "vae": vae,
    "gan": gan,
    "gan_vae": gan_vae,
    "tvae": tvae,
}


def _resolve_out_dir(cfg: dict, config_path: str, output_dir: Optional[str] = None) -> Path:
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


def main(config_path: str, stage: str = "all", output_dir: Optional[str] = None) -> Path:
    cfg = load_config(config_path)
    exp = cfg.get("experiment", {})
    set_seed(exp.get("seed", 42))

    out_dir = _resolve_out_dir(cfg, config_path, output_dir)
    cfg["output_dir"] = str(out_dir)

    model_name = cfg["model"]["name"]
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    mod = MODELS[model_name]
    bundle = load_data(cfg["dataset"])
    device = get_device(cfg)
    print(f"Model: {model_name} | Dataset: {cfg['dataset']['name']} | Device: {device}")
    print(f"Output: {out_dir}")

    model = None
    meta = {}

    if stage in ("train", "all"):
        print("\n=== Training ===")
        result = mod.train(bundle, cfg, out_dir)
        if model_name == "gan":
            enc, gen, disc, meta = result
            model = enc if isinstance(enc, dict) else (enc, gen, disc)
        else:
            model, meta = result
        print(f"Training done. Best checkpoint: {out_dir / 'checkpoint_best.pth'}")

    if stage in ("generate", "all"):
        print("\n=== Generation ===")
        ckpt = out_dir / "checkpoint_best.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run train first.")

        if model_name == "vae":
            model = mod.load_checkpoint(ckpt, cfg)
            mod.generate(model, bundle, cfg, out_dir)
        elif model_name == "gan":
            if model is None:
                enc, gen, disc = mod.load_checkpoint(ckpt, cfg, out_dir)
            elif isinstance(model, dict):
                enc, gen, disc = model, None, None
            else:
                enc, gen, disc = model
            mod.generate(enc, gen, disc, bundle, cfg, out_dir)
        elif model_name == "gan_vae":
            if model is None:
                model = mod.load_checkpoint(ckpt, cfg)
            mod.generate(model, bundle, cfg, out_dir)
        elif model_name == "tvae":
            if model is None:
                model = mod.load_checkpoint(ckpt, cfg)
            if not meta:
                _, _, data_min, data_max, sequences, labels, _, per_class_norm = tvae._prepare_data(
                    bundle, cfg
                )
                meta = {
                    "sequences": sequences,
                    "labels": labels,
                    "label_names": bundle.label_names,
                    "feature_columns": bundle.feature_columns,
                    "data_min": data_min.tolist() if hasattr(data_min, "tolist") else float(data_min),
                    "data_max": data_max.tolist() if hasattr(data_max, "tolist") else float(data_max),
                    "per_class_norm": per_class_norm,
                }
            mod.generate(model, bundle, cfg, out_dir, meta)

        print(f"Generated samples: {out_dir / 'generated_samples.npy'}")

    if stage in ("evaluate", "all"):
        print("\n=== Evaluation ===")
        generated = None
        gen_path = out_dir / "generated_samples.npy"
        if gen_path.exists():
            import numpy as np

            generated = np.load(gen_path)

        if model_name == "vae":
            mod.evaluate(bundle, out_dir, cfg, generated)
        elif model_name == "gan":
            segments = meta.get("segments")
            if segments is None:
                _, _, _, segments, _ = gan._prepare_segments(bundle, cfg)
            mod.evaluate(bundle, out_dir, cfg, generated, segments=segments)
        elif model_name == "gan_vae":
            m = model
            if m is None:
                m = mod.load_checkpoint(out_dir / "checkpoint_best.pth", cfg)
            mod.evaluate(bundle, out_dir, cfg, generated, model=m, meta=meta)
        elif model_name == "tvae":
            m = model
            if m is None:
                m = mod.load_checkpoint(out_dir / "checkpoint_best.pth", cfg)
            if not meta:
                _, _, data_min, data_max, sequences, labels, _, per_class_norm = tvae._prepare_data(
                    bundle, cfg
                )
                meta = {
                    "sequences": sequences,
                    "labels": labels,
                    "label_names": bundle.label_names,
                    "feature_columns": bundle.feature_columns,
                    "data_min": data_min.tolist() if hasattr(data_min, "tolist") else float(data_min),
                    "data_max": data_max.tolist() if hasattr(data_max, "tolist") else float(data_max),
                    "per_class_norm": per_class_norm,
                }
            mod.evaluate(bundle, out_dir, cfg, generated, model=m, meta=meta)

        print(f"Metrics: {out_dir / 'metrics.json'}")

    print("\nDone.")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data augmentation training pipeline")
    parser.add_argument("--config", required=True, help="Path to yaml config")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["train", "generate", "evaluate", "all"],
        help="Pipeline stage",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Reuse an existing run directory (required for generate/evaluate after train)",
    )
    args = parser.parse_args()
    main(args.config, args.stage, args.output_dir)
