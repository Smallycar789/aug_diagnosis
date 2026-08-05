"""Export ImageNet-equivalent ResNet init checkpoints for offline acceptance."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnosis.io_utils import load_config, resolve_path, set_seed
from diagnosis.resnet import build_tl_resnet18

RESNET_CONFIGS = (
    "configs/diagnosis/resnet_image_quality.yaml",
    "configs/diagnosis/resnet_sensitivity.yaml",
    "configs/diagnosis/resnet_cooler.yaml",
    "configs/diagnosis/resnet_sifuqi.yaml",
)


def export_one(config_path: str, out_dir: Path) -> Path:
    cfg = load_config(config_path)
    dataset_name = cfg["dataset"]["name"]
    model_cfg = cfg["model"]
    num_classes = int(model_cfg["num_classes"])
    freeze_layers = bool(model_cfg.get("freeze_layers", True))
    seed = int(cfg.get("experiment", {}).get("seed", 42))

    set_seed(seed)
    model = build_tl_resnet18(num_classes, freeze_layers=freeze_layers)

    out_path = out_dir / "resnet_{}.pth".format(dataset_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "source": "imagenet_init",
        },
        str(out_path),
    )
    print("Exported {} (seed={}, num_classes={})".format(out_path, seed, num_classes))
    return out_path


def main() -> None:
    out_dir = resolve_path("references/diagnosis/pretrained")
    out_dir.mkdir(parents=True, exist_ok=True)
    for config_path in RESNET_CONFIGS:
        export_one(config_path, out_dir)


if __name__ == "__main__":
    main()
