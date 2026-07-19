# -*- coding: utf-8 -*-
"""Built-in dataset profiles and YAML builders for platform services."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from platform_common import PLATFORM_ROOT, PROJECT_ROOT, RUNTIME_ROOT

DEMO_DATA_ROOT = PLATFORM_ROOT / "demo_data"
DEFAULT_SEED = 42

DIAGNOSIS_ALGORITHMS = ("cnn_bigru", "resnet", "tl_meta")
AUGMENTATION_ALGORITHMS = ("gan", "tvae", "gan_vae")

PROFILE_SPECS: Dict[str, Dict[str, Any]] = {
    "sensitivity": {
        "dataset_name": "sensitivity",
        "demo_dir": DEMO_DATA_ROOT / "sensitivity",
        "project_data_dir": PROJECT_ROOT / "data" / "sensitivity",
        "class_files": {
            "normal": "normal.csv",
            "sensitivity_degradation": "sensitivity_degradation.csv",
            "coupled_severe_fault": "coupled_severe_fault.csv",
        },
        "value_columns": ["avg_detectivity", "NETD_mK"],
        "sample_length": 96,
        "stride": 32,
        "split_mode": "random",
        "swap_val_test": True,
        "normalize": "global",
        "val_ratio": 0.15,
        "diagnosis_models": {
            "cnn_bigru": {
                "input_channels": 2,
                "num_classes": 3,
                "hidden_dims": 64,
                "gru_hidden": 128,
                "batch_size": 8,
                "epochs_smoke": 2,
                "epochs_full": 80,
            },
            "resnet": {
                "num_classes": 3,
                "batch_size": 8,
                "epochs_smoke": 2,
                "epochs_full": 30,
            },
            "tl_meta": {
                "num_classes": 3,
                "batch_size": 8,
                "epochs_smoke": 2,
                "epochs_full": 20,
            },
        },
        "aug_models": {
            "tvae": {
                "conditional": True,
                "input_dim": 2,
                "seq_len": 96,
                "latent_dim": 16,
                "hidden_dim": 32,
                "epochs_smoke": 2,
                "num_generate_per_class": 5,
            },
            "gan": {
                "window_size": 96,
                "epochs_smoke": 2,
                "num_generate_per_class": 5,
            },
            "gan_vae": {
                "seq_len": 96,
                "input_channels": 2,
                "latent_dim": 32,
                "epochs_smoke": 2,
                "num_generate_per_class": 5,
            },
        },
    },
    "cooler": {
        "dataset_name": "cooler",
        "demo_dir": DEMO_DATA_ROOT / "cooler",
        "project_data_dir": PROJECT_ROOT / "data" / "cooler",
        "simulation_csv": "all_simulation.csv",
        "time_column": "time_hours",
        "group_column": "group_id",
        "value_columns": ["T_stable_K", "t_cool_s", "sigma_T_K"],
        "normal_time_max": 2000,
        "fault_time_min": 6000,
        "label_names": ["normal", "temperature_control_fault"],
        "sample_length": 30,
        "stride": 15,
        "normalize": "global",
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "diagnosis_models": {
            "cnn_bigru": {
                "input_channels": 3,
                "num_classes": 2,
                "hidden_dims": 64,
                "gru_hidden": 128,
                "batch_size": 8,
                "epochs_smoke": 2,
                "epochs_full": 50,
            },
            "resnet": {
                "num_classes": 2,
                "batch_size": 8,
                "epochs_smoke": 2,
                "epochs_full": 20,
            },
            "tl_meta": {
                "num_classes": 2,
                "batch_size": 8,
                "epochs_smoke": 2,
                "epochs_full": 15,
            },
        },
        "aug_models": {
            "gan": {
                "window_size": 30,
                "epochs_smoke": 2,
                "num_generate_per_class": 5,
            },
            "tvae": {
                "conditional": True,
                "input_dim": 3,
                "seq_len": 30,
                "latent_dim": 16,
                "hidden_dim": 32,
                "epochs_smoke": 2,
                "num_generate_per_class": 5,
            },
            "gan_vae": {
                "seq_len": 30,
                "input_channels": 3,
                "latent_dim": 32,
                "epochs_smoke": 2,
                "num_generate_per_class": 5,
            },
        },
    },
    "sifuqi": {
        "dataset_name": "sifuqi",
        "demo_dir": DEMO_DATA_ROOT / "sifuqi",
        "project_data_dir": PROJECT_ROOT / "data" / "sifuqi",
        "csv": "servo_accuracy.csv",
        "time_column": "hours",
        "value_columns": ["servo_accuracy"],
        "normal_time_max": 1000,
        "fault_time_min": 6000,
        "label_names": ["normal", "tracking_fault"],
        "sample_length": 64,
        "stride": 16,
        "normalize": "global",
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "diagnosis_models": {
            "cnn_bigru": {
                "input_channels": 1,
                "num_classes": 2,
                "hidden_dims": 64,
                "gru_hidden": 128,
                "batch_size": 8,
                "epochs_smoke": 2,
                "epochs_full": 40,
            },
            "resnet": {
                "num_classes": 2,
                "batch_size": 8,
                "epochs_smoke": 2,
                "epochs_full": 20,
            },
            "tl_meta": {
                "num_classes": 2,
                "batch_size": 8,
                "epochs_smoke": 2,
                "epochs_full": 15,
            },
        },
        "aug_models": {
            "tvae": {
                "conditional": True,
                "input_dim": 1,
                "seq_len": 64,
                "latent_dim": 16,
                "hidden_dim": 32,
                "epochs_smoke": 2,
                "num_generate_per_class": 5,
            },
            "gan": {
                "window_size": 64,
                "epochs_smoke": 2,
                "num_generate_per_class": 5,
            },
            "gan_vae": {
                "seq_len": 64,
                "input_channels": 1,
                "latent_dim": 32,
                "epochs_smoke": 2,
                "num_generate_per_class": 5,
            },
        },
    },
}


def resolve_data_dir(data_dir: Optional[str], profile: str) -> Path:
    if data_dir:
        path = Path(data_dir)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path
    spec = PROFILE_SPECS[profile]
    demo = spec["demo_dir"]
    if demo.exists():
        return demo.resolve()
    return spec["project_data_dir"].resolve()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_yaml(cfg: Dict[str, Any], prefix: str) -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    path = RUNTIME_ROOT / f"{prefix}_{_timestamp()}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return path


def build_diagnosis_cfg(
    profile: str,
    algorithm: str,
    data_dir: Path,
    device: str = "auto",
    smoke: bool = True,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    if profile not in PROFILE_SPECS:
        raise ValueError(f"Unknown dataset_profile: {profile}")
    if algorithm not in DIAGNOSIS_ALGORITHMS:
        raise ValueError(f"Unknown diagnosis algorithm: {algorithm}")

    spec = PROFILE_SPECS[profile]
    model_spec = spec["diagnosis_models"][algorithm]
    epochs = model_spec["epochs_smoke"] if smoke else model_spec["epochs_full"]

    dataset: Dict[str, Any] = {
        "name": spec["dataset_name"],
        "root": str(data_dir),
        "normalize": spec.get("normalize", "global"),
        "domain_adaptation": False,
        "sample_length": spec["sample_length"],
        "stride": spec["stride"],
    }

    if profile == "sensitivity":
        dataset.update(
            {
                "class_files": copy.deepcopy(spec["class_files"]),
                "value_columns": list(spec["value_columns"]),
                "split_mode": spec["split_mode"],
                "swap_val_test": spec["swap_val_test"],
            }
        )
    elif profile in ("cooler", "sifuqi"):
        dataset.update(
            {
                "time_column": spec["time_column"],
                "value_columns": list(spec["value_columns"]),
                "normal_time_max": spec["normal_time_max"],
                "fault_time_min": spec["fault_time_min"],
                "label_names": list(spec["label_names"]),
                "val_ratio": spec.get("val_ratio", 0.15),
                "test_ratio": spec.get("test_ratio", 0.15),
            }
        )
        if profile == "cooler":
            dataset["simulation_csv"] = spec["simulation_csv"]
            dataset["group_column"] = spec["group_column"]
        else:
            dataset["csv"] = spec["csv"]

    model: Dict[str, Any] = {
        "name": algorithm,
        "num_classes": model_spec["num_classes"],
        "batch_size": model_spec["batch_size"],
        "epochs": epochs,
        "lr": 0.001,
        "val_ratio": spec.get("val_ratio", 0.15),
    }
    if algorithm == "cnn_bigru":
        model.update(
            {
                "input_channels": model_spec["input_channels"],
                "sample_length": spec["sample_length"],
                "hidden_dims": model_spec["hidden_dims"],
                "gru_hidden": model_spec["gru_hidden"],
                "num_layers": 2,
                "alpha": 0,
            }
        )
    elif algorithm == "resnet":
        model.update({"freeze_layers": True, "train_ratio": 0.6, "val_ratio": 0.2})
    elif algorithm == "tl_meta":
        model.update({"n_way": model_spec["num_classes"], "n_shot": 5, "n_query": 5})

    cfg = {
        "experiment": {
            "name": f"platform_{profile}_{algorithm}",
            "seed": DEFAULT_SEED,
            "device": device,
        },
        "output": {"root": str(output_root or (PROJECT_ROOT / "outputs" / "diagnosis"))},
        "dataset": dataset,
        "model": model,
    }
    return cfg


def build_augmentation_cfg(
    profile: str,
    algorithm: str,
    data_dir: Path,
    device: str = "auto",
    smoke: bool = True,
    output_root: Optional[Path] = None,
) -> Dict[str, Any]:
    if profile not in PROFILE_SPECS:
        raise ValueError(f"Unknown dataset_profile: {profile}")
    if algorithm not in AUGMENTATION_ALGORITHMS:
        raise ValueError(f"Unknown augmentation algorithm: {algorithm}")

    spec = PROFILE_SPECS[profile]
    model_spec = spec["aug_models"][algorithm]
    epochs = model_spec["epochs_smoke"] if smoke else model_spec.get("epochs_full", 50)

    dataset: Dict[str, Any] = {
        "name": spec["dataset_name"],
        "root": str(data_dir),
        "value_columns": list(spec["value_columns"]),
        "sample_length": spec["sample_length"],
        "stride": spec["stride"],
        "sample_rate": 1.0,
    }

    if profile == "sensitivity":
        dataset["class_files"] = copy.deepcopy(spec["class_files"])
        dataset["unit_column"] = "unit"
        dataset["cycle_column"] = "cycle"
    elif profile == "cooler":
        dataset.update(
            {
                "csv": spec["simulation_csv"],
                "time_column": spec["time_column"],
                "group_column": spec["group_column"],
                "normal_time_max": spec["normal_time_max"],
                "fault_time_min": spec["fault_time_min"],
                "label_names": list(spec["label_names"]),
            }
        )
    else:
        dataset.update(
            {
                "csv": spec["csv"],
                "time_column": spec["time_column"],
                "normal_time_max": spec["normal_time_max"],
                "fault_time_min": spec["fault_time_min"],
                "label_names": list(spec["label_names"]),
            }
        )

    model: Dict[str, Any] = {
        "name": algorithm,
        "epochs": epochs,
        "batch_size": 16,
        "lr": 0.001,
        "num_generate_per_class": model_spec.get("num_generate_per_class", 10),
        "per_class": True,
        "per_class_norm": True,
    }

    if algorithm == "tvae":
        model.update(
            {
                "conditional": model_spec.get("conditional", True),
                "seq_len": model_spec.get("seq_len", spec["sample_length"]),
                "input_dim": model_spec["input_dim"],
                "latent_dim": model_spec.get("latent_dim", 16),
                "hidden_dim": model_spec.get("hidden_dim", 32),
                "num_layers": 1,
                "label_embed_dim": 8,
                "phase1_tf_epochs": 1,
                "ramp_tfr_epochs": 1,
                "beta_end": 0.01,
            }
        )
    elif algorithm == "gan":
        model.update(
            {
                "z_dim": 64,
                "window_size": model_spec.get("window_size", spec["sample_length"]),
                "overlap_ratio": 0.2,
                "g_lr": 0.002,
                "d_lr": 0.0001,
                "enc_lr": 0.001,
            }
        )
    elif algorithm == "gan_vae":
        model.update(
            {
                "seq_len": model_spec.get("seq_len", spec["sample_length"]),
                "input_channels": model_spec.get("input_channels", len(spec["value_columns"])),
                "latent_dim": model_spec.get("latent_dim", 32),
                "alpha": 0.01,
            }
        )

    cfg = {
        "experiment": {
            "name": f"platform_{profile}_{algorithm}",
            "seed": DEFAULT_SEED,
            "device": device,
        },
        "output": {"root": str(output_root or (PROJECT_ROOT / "outputs" / "data_aug"))},
        "dataset": dataset,
        "model": model,
    }
    return cfg


def write_diagnosis_config(
    profile: str,
    algorithm: str,
    data_dir: Path,
    device: str = "auto",
    smoke: bool = True,
) -> Path:
    cfg = build_diagnosis_cfg(profile, algorithm, data_dir, device=device, smoke=smoke)
    return _write_yaml(cfg, f"diagnosis_{profile}_{algorithm}")


def write_augmentation_config(
    profile: str,
    algorithm: str,
    data_dir: Path,
    device: str = "auto",
    smoke: bool = True,
) -> Path:
    cfg = build_augmentation_cfg(profile, algorithm, data_dir, device=device, smoke=smoke)
    return _write_yaml(cfg, f"aug_{profile}_{algorithm}")
