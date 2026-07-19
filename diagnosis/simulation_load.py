"""Shared loaders for cooler/sifuqi simulation CSVs with time-threshold windowing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnosis.io_utils import resolve_path


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]
    return df


def _window_starts(length: int, sample_length: int, stride: int) -> range:
    if length < sample_length:
        return range(0)
    return range(0, length - sample_length + 1, stride)


def _resolve_sample_length(cfg: dict[str, Any]) -> int:
    return int(cfg.get("sample_length", cfg.get("seq_len", cfg.get("window_size", 32))))


def _resolve_stride(cfg: dict[str, Any], sample_length: int) -> int:
    return int(cfg.get("stride", cfg.get("window_stride", max(1, sample_length // 2))))


def _resolve_label_names(cfg: dict[str, Any], default: list[str]) -> list[str]:
    names = cfg.get("label_names")
    if names is None:
        return default
    return list(names)


def collect_time_threshold_windows(
    groups: list[tuple[str, pd.DataFrame]],
    *,
    time_col: str,
    value_columns: list[str],
    sample_length: int,
    stride: int,
    normal_time_max: float,
    fault_time_min: float,
    label_names: list[str],
    layout: str = "channels_first",
) -> tuple[list[np.ndarray], list[int], list[str], list[str]]:
    """Slide windows inside normal (t < max) and fault (t > min) segments per unit."""
    if len(label_names) != 2:
        raise ValueError(f"time-threshold loader expects 2 label_names, got {label_names}")

    windows: list[np.ndarray] = []
    labels: list[int] = []
    unit_ids: list[str] = []

    for unit_id, group in groups:
        group = group.sort_values(time_col)
        for label_id, label_name, mask in (
            (0, label_names[0], group[time_col] < normal_time_max),
            (1, label_names[1], group[time_col] > fault_time_min),
        ):
            segment = group.loc[mask, value_columns].to_numpy(dtype=np.float32)
            if len(segment) < sample_length:
                continue
            for start in _window_starts(len(segment), sample_length, stride):
                window = segment[start : start + sample_length]
                if layout == "channels_first":
                    windows.append(window.T)
                else:
                    windows.append(window)
                labels.append(label_id)
                unit_ids.append(f"{unit_id}:{label_name}")

    return windows, labels, unit_ids, label_names


def read_cooler_groups(cfg: dict[str, Any], root: Optional[Path] = None) -> tuple[list[tuple[str, pd.DataFrame]], Path]:
    root = root or resolve_path(cfg.get("root", "data/cooler"))
    csv_name = cfg.get("simulation_csv", cfg.get("csv", "all_simulation.csv"))
    csv_path = root / csv_name
    time_col = cfg.get("time_column", "time_hours")
    group_col = cfg.get("group_column", "group_id")
    value_columns = list(
        cfg.get("value_columns", ["T_stable_K", "t_cool_s", "sigma_T_K"])
    )

    df = normalize_column_names(pd.read_csv(csv_path))
    for column in [time_col, group_col, *value_columns]:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in {csv_path}")

    groups = [(str(unit), group) for unit, group in df.groupby(group_col)]
    return groups, csv_path


def read_sifuqi_groups(cfg: dict[str, Any], root: Optional[Path] = None) -> tuple[list[tuple[str, pd.DataFrame]], Path]:
    root = root or resolve_path(cfg.get("root", "data/sifuqi"))
    csv_name = cfg.get("csv", "servo_accuracy.csv")
    csv_path = root / csv_name
    time_col = cfg.get("time_column", "hours")
    value_name = cfg.get("value_name", "servo_accuracy")
    group_prefix = cfg.get("group_column_prefix", "group_")

    df = normalize_column_names(pd.read_csv(csv_path))
    if time_col not in df.columns:
        raise KeyError(f"Column '{time_col}' not found in {csv_path}")

    group_columns = [col for col in df.columns if col != time_col and col.startswith(group_prefix)]
    if not group_columns:
        pattern = re.compile(r"^group_\d+$")
        group_columns = [col for col in df.columns if col != time_col and pattern.match(col)]
    if not group_columns:
        raise ValueError(f"No group columns found in {csv_path}")

    long_df = df.melt(
        id_vars=[time_col],
        value_vars=group_columns,
        var_name="unit",
        value_name=value_name,
    )
    long_df = long_df.rename(columns={time_col: "time_hours"})
    groups = [(str(unit), group) for unit, group in long_df.groupby("unit")]
    return groups, csv_path


def load_cooler_time_windows(
    cfg: dict[str, Any],
    *,
    layout: str = "channels_first",
) -> tuple[list[np.ndarray], list[int], list[str], list[str], dict[str, Any]]:
    sample_length = _resolve_sample_length(cfg)
    stride = _resolve_stride(cfg, sample_length)
    label_names = _resolve_label_names(cfg, ["normal", "temperature_control_fault"])
    value_columns = list(cfg.get("value_columns", ["T_stable_K", "t_cool_s", "sigma_T_K"]))
    time_col = cfg.get("time_column", "time_hours")
    normal_time_max = float(cfg.get("normal_time_max", 2000))
    fault_time_min = float(cfg.get("fault_time_min", 6000))

    groups, csv_path = read_cooler_groups(cfg)
    windows, labels, unit_ids, label_names = collect_time_threshold_windows(
        groups,
        time_col=time_col,
        value_columns=value_columns,
        sample_length=sample_length,
        stride=stride,
        normal_time_max=normal_time_max,
        fault_time_min=fault_time_min,
        label_names=label_names,
        layout=layout,
    )
    if not windows:
        raise ValueError(
            f"No cooler windows created (sample_length={sample_length}, stride={stride}, "
            f"normal<{normal_time_max}, fault>{fault_time_min})."
        )

    meta = {
        "source_file": str(csv_path),
        "value_columns": value_columns,
        "time_column": time_col,
        "sample_length": sample_length,
        "stride": stride,
        "normal_time_max": normal_time_max,
        "fault_time_min": fault_time_min,
        "num_groups": len(groups),
    }
    return windows, labels, unit_ids, label_names, meta


def load_sifuqi_time_windows(
    cfg: dict[str, Any],
    *,
    layout: str = "channels_first",
) -> tuple[list[np.ndarray], list[int], list[str], list[str], dict[str, Any]]:
    sample_length = _resolve_sample_length(cfg)
    stride = _resolve_stride(cfg, sample_length)
    label_names = _resolve_label_names(cfg, ["normal", "tracking_fault"])
    value_name = cfg.get("value_name", "servo_accuracy")
    value_columns = list(cfg.get("value_columns", [value_name]))
    time_col = "time_hours"
    normal_time_max = float(cfg.get("normal_time_max", 1000))
    fault_time_min = float(cfg.get("fault_time_min", 6000))

    groups, csv_path = read_sifuqi_groups(cfg)
    windows, labels, unit_ids, label_names = collect_time_threshold_windows(
        groups,
        time_col=time_col,
        value_columns=value_columns,
        sample_length=sample_length,
        stride=stride,
        normal_time_max=normal_time_max,
        fault_time_min=fault_time_min,
        label_names=label_names,
        layout=layout,
    )
    if not windows:
        raise ValueError(
            f"No sifuqi windows created (sample_length={sample_length}, stride={stride}, "
            f"normal<{normal_time_max}, fault>{fault_time_min})."
        )

    meta = {
        "source_file": str(csv_path),
        "value_columns": value_columns,
        "time_column": cfg.get("time_column", "hours"),
        "sample_length": sample_length,
        "stride": stride,
        "normal_time_max": normal_time_max,
        "fault_time_min": fault_time_min,
        "num_groups": len(groups),
    }
    return windows, labels, unit_ids, label_names, meta
