"""Load 1D series and class-window datasets for data augmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_aug.io_utils import resolve_path


@dataclass
class AugDataBundle:
    raw_data: np.ndarray
    labels: np.ndarray | None = None
    label_names: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_data(dataset_cfg: dict[str, Any]) -> AugDataBundle:
    name = dataset_cfg["name"]
    if dataset_cfg.get("class_files"):
        return _load_class_windows(dataset_cfg)
    if name == "cooler":
        return _load_cooler(dataset_cfg)
    if name == "sifuqi":
        return _load_sifuqi(dataset_cfg)
    if name == "cwru":
        return _load_cwru(dataset_cfg)
    raise ValueError(f"Unknown dataset name: {name}")


def _resolve_value_columns(cfg: dict[str, Any]) -> list[str]:
    columns = cfg.get("value_columns")
    if columns is not None:
        return [columns] if isinstance(columns, str) else list(columns)
    return [cfg.get("value_column", "value")]


def _window_starts(length: int, sample_length: int, stride: int) -> range:
    if length < sample_length:
        return range(0)
    return range(0, length - sample_length + 1, stride)


def _load_class_windows(cfg: dict[str, Any]) -> AugDataBundle:
    """Load per-class CSV files as multivariate sliding windows."""
    root = resolve_path(cfg.get("root", f"data/{cfg['name']}"))
    value_columns = _resolve_value_columns(cfg)
    class_files = dict(cfg["class_files"])
    label_names = list(class_files.keys())
    sample_length = int(cfg.get("sample_length", cfg.get("seq_len", 96)))
    stride = int(cfg.get("stride", sample_length))
    unit_column = cfg.get("unit_column", "unit")
    cycle_column = cfg.get("cycle_column", "cycle")
    whole_file_as_series = bool(cfg.get("whole_file_as_series", False))

    windows: list[np.ndarray] = []
    labels: list[int] = []
    source_files: dict[str, str] = {}

    for label_id, (label_name, csv_name) in enumerate(class_files.items()):
        csv_path = root / csv_name
        df = _read_csv(csv_path)
        missing = [column for column in value_columns if column not in df.columns]
        if missing:
            raise KeyError(f"Column(s) {missing} not found in {csv_path}")

        source_files[label_name] = str(csv_path)
        if whole_file_as_series or unit_column not in df.columns:
            groups = [("__series__", df)]
        else:
            groups = list(df.groupby(unit_column))

        for _, group in groups:
            if cycle_column in group.columns:
                group = group.sort_values(cycle_column)
            values = group[value_columns].to_numpy(dtype=np.float32)
            for start in _window_starts(len(values), sample_length, stride):
                windows.append(values[start : start + sample_length])
                labels.append(label_id)

    if not windows:
        raise ValueError(
            f"No windows created for {cfg['name']} "
            f"(sample_length={sample_length}, stride={stride})."
        )

    raw = np.asarray(windows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return AugDataBundle(
        raw_data=raw,
        labels=y,
        label_names=label_names,
        feature_columns=value_columns,
        meta={
            "dataset": cfg["name"],
            "source_files": source_files,
            "value_columns": value_columns,
            "sample_length": sample_length,
            "stride": stride,
            "num_samples": int(len(raw)),
            "num_features": int(raw.shape[-1]),
            "class_counts": {
                name: int((y == idx).sum()) for idx, name in enumerate(label_names)
            },
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
        feature_columns=[meta_column],
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
        label_names=[level],
        feature_columns=[column],
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
        feature_columns=[column],
        meta={
            "dataset": "cwru",
            "source_file": str(csv_path),
            "value_column": column,
            "sample_rate": float(cfg.get("sample_rate", 12000.0)),
        },
    )
