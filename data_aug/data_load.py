"""Load 1D series from degradation / cooler / sifuqi / cwru datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from data_aug.io_utils import resolve_path


@dataclass
class AugDataBundle:
    raw_data: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_data(dataset_cfg: dict[str, Any]) -> AugDataBundle:
    name = dataset_cfg["name"]
    if name == "degradation":
        return _load_degradation(dataset_cfg)
    if name == "cooler":
        return _load_cooler(dataset_cfg)
    if name == "sifuqi":
        return _load_sifuqi(dataset_cfg)
    if name == "cwru":
        return _load_cwru(dataset_cfg)
    raise ValueError(f"Unknown dataset name: {name}")


def _load_degradation(cfg: dict[str, Any]) -> AugDataBundle:
    root = resolve_path(cfg.get("root", "data/DegradationData/fault_diagnosis"))
    data_file = root / cfg.get("data_file", "point_level_all.csv")
    df = _read_csv(data_file)

    column = cfg.get("value_column")
    if column is None:
        feature_file = root / cfg.get("feature_columns", "feature_columns.json")
        with open(feature_file, encoding="utf-8") as f:
            features = json.load(f)
        column = features[0]

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in {data_file}")

    raw = df[column].values.astype(np.float32)
    return AugDataBundle(
        raw_data=raw,
        meta={
            "dataset": "degradation",
            "source_file": str(data_file),
            "value_column": column,
            "sample_rate": float(cfg.get("sample_rate", 1.0)),
        },
    )


def _load_cooler(cfg: dict[str, Any]) -> AugDataBundle:
    root = resolve_path(cfg.get("root", "data/cooler"))
    csv_path = root / cfg.get("csv", "cooler_simulation_results/all_simulation.csv")
    df = _read_csv(csv_path)

    columns = cfg.get("value_columns", ["T_stable_K", "t_cool_hours", "sigma_T_K"])
    use_multivariate = cfg.get("multivariate", False)

    if use_multivariate:
        raw = df[columns].values.astype(np.float32).flatten()
        meta_column = "+".join(columns)
    else:
        column = cfg.get("value_column", columns[0])
        raw = df[column].values.astype(np.float32)
        meta_column = column

    return AugDataBundle(
        raw_data=raw,
        meta={
            "dataset": "cooler",
            "source_file": str(csv_path),
            "value_column": meta_column,
            "sample_rate": float(cfg.get("sample_rate", 1.0)),
        },
    )


def _load_sifuqi(cfg: dict[str, Any]) -> AugDataBundle:
    root = resolve_path(cfg.get("root", "data/sifuqi"))
    level = cfg.get("level", "normal")
    file_map = {
        "normal": "servo_normal.csv",
        "mild": "servo_mild.csv",
        "moderate": "servo_moderate.csv",
        "severe": "servo_severe.csv",
    }
    csv_name = cfg.get("csv") or file_map.get(level, "servo_normal.csv")
    csv_path = root / csv_name
    df = _read_csv(csv_path)

    column = cfg.get("value_column", "azimuth_error")
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in {csv_path}")

    raw = df[column].values.astype(np.float32)
    return AugDataBundle(
        raw_data=raw,
        meta={
            "dataset": "sifuqi",
            "source_file": str(csv_path),
            "value_column": column,
            "level": level,
            "sample_rate": float(cfg.get("sample_rate", 100.0)),
        },
    )


def _load_cwru(cfg: dict[str, Any]) -> AugDataBundle:
    csv_path = resolve_path(cfg.get("csv", "data/CWRU/CWRU_12k_1797_10c.csv"))
    df = _read_csv(csv_path)

    column = cfg.get("value_column")
    if column is None:
        column_index = int(cfg.get("column_index", 0))
        column = df.columns[column_index]
    elif column not in df.columns:
        raise KeyError(f"Column '{column}' not found in {csv_path}")

    raw = df[column].values.astype(np.float32)
    return AugDataBundle(
        raw_data=raw,
        meta={
            "dataset": "cwru",
            "source_file": str(csv_path),
            "value_column": column,
            "sample_rate": float(cfg.get("sample_rate", 12000.0)),
        },
    )
