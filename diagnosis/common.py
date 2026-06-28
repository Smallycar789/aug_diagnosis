"""Shared data helpers for fault diagnosis (normalization, windows, splits)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NORMALIZE_MODES = ("none", "per_window", "global")


def resolve_value_columns(cfg: dict[str, Any]) -> list[str]:
    columns = cfg.get("value_columns")
    if columns is not None:
        if isinstance(columns, str):
            return [columns]
        return list(columns)
    column = cfg.get("value_column")
    if column is None:
        raise ValueError("dataset config must provide value_columns or value_column")
    return [column]


def resolve_sample_length(cfg: dict[str, Any], default: int = 32) -> int:
    return int(cfg.get("sample_length", cfg.get("window_size", default)))


def resolve_stride(cfg: dict[str, Any], default: int = 16) -> int:
    return int(cfg.get("stride", cfg.get("window_stride", default)))


def resolve_normalize_mode(cfg: dict[str, Any], default: str = "global") -> str:
    mode = cfg.get("normalize", default) or default
    if mode not in NORMALIZE_MODES:
        raise ValueError(f"Unknown normalize mode: {mode}. Use {', '.join(NORMALIZE_MODES)}.")
    return mode


def zscore_channel(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    mean = signal.mean()
    std = signal.std()
    if std > 1e-8:
        return ((signal - mean) / std).astype(np.float32)
    return (signal - mean).astype(np.float32)


def zscore_window_channels(window: np.ndarray) -> np.ndarray:
    """Z-score each channel of a window shaped (C, L) or (L,)."""
    window = np.asarray(window, dtype=np.float32)
    if window.ndim == 1:
        return zscore_channel(window)
    return np.stack([zscore_channel(window[c]) for c in range(window.shape[0])], axis=0)


def normalize_windows_per_sample(X: np.ndarray) -> np.ndarray:
    """Z-score each sample; supports (N, C, L) or (N, 1, L)."""
    out = X.copy()
    if out.ndim == 2:
        out = out[:, np.newaxis, :]
    for i in range(out.shape[0]):
        for c in range(out.shape[1]):
            out[i, c] = zscore_channel(out[i, c])
    return out.astype(np.float32)


def fit_global_norm_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    meta = {
        "norm_mean": mean.reshape(-1).tolist(),
        "norm_std": std.reshape(-1).tolist(),
    }
    return mean, std, meta


def apply_global_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return X.astype(np.float32)
    return ((X - mean) / std).astype(np.float32)


def apply_dataset_normalization(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    mode: str = "global",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mode = resolve_normalize_mode({"normalize": mode})
    meta: dict[str, Any] = {"normalize": mode}

    if mode == "none":
        return X_train, X_val, X_test, meta
    if mode == "per_window":
        return (
            normalize_windows_per_sample(X_train),
            normalize_windows_per_sample(X_val),
            normalize_windows_per_sample(X_test),
            meta,
        )
    if mode == "global":
        mean, std, norm_meta = fit_global_norm_stats(X_train)
        meta.update(norm_meta)
        return (
            apply_global_norm(X_train, mean, std),
            apply_global_norm(X_val, mean, std),
            apply_global_norm(X_test, mean, std),
            meta,
        )

    raise ValueError(f"Unknown normalize mode: {mode}")


def flatten_multivariate_window(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    if window.ndim == 2:
        return window.reshape(-1)
    return window


def stack_multivariate_window(signals: list[np.ndarray], start: int, sample_length: int) -> np.ndarray:
    return np.stack([signal[start : start + sample_length] for signal in signals], axis=0).astype(np.float32)


def prepare_signal_segment(
    window: np.ndarray,
    normalize: str = "per_window",
    *,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
) -> np.ndarray:
    """Prepare a (C, L) or (L,) window for downstream models such as STFT."""
    window = np.asarray(window, dtype=np.float32)
    if window.ndim == 1:
        window = window[np.newaxis, :]

    mode = resolve_normalize_mode({"normalize": normalize})
    if mode == "none":
        out = window
    elif mode == "per_window":
        out = zscore_window_channels(window)
    elif mode == "global":
        if norm_mean is None or norm_std is None:
            raise ValueError("global normalization requires norm_mean and norm_std")
        out = apply_global_norm(window[np.newaxis, ...], norm_mean, norm_std)[0]
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")

    return flatten_multivariate_window(out)


def resolve_window_starts(
    min_len: int,
    sample_length: int,
    stride: int,
    region: str,
    region_fraction: float,
) -> range:
    """Return valid window start indices for head/tail/full regions."""
    if min_len < sample_length:
        return range(0)

    max_start = min_len - sample_length
    fraction = float(region_fraction)
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"region_fraction must be in (0, 1], got {region_fraction}")

    if region == "head":
        region_end = int(min_len * fraction)
        end_start = min(max(region_end - sample_length, 0), max_start)
        return range(0, end_start + 1, stride)
    if region == "tail":
        region_start = int(min_len * (1.0 - fraction))
        region_start = min(max(region_start, 0), max_start)
        return range(region_start, max_start + 1, stride)
    if region in ("full", "all"):
        return range(0, max_start + 1, stride)

    raise ValueError(f"Unknown window region: {region}. Use head, tail, or full.")


def resolve_class_window_region(label_name: str, cfg: dict[str, Any]) -> str:
    class_regions = cfg.get("class_window_regions", {})
    if label_name in class_regions:
        return str(class_regions[label_name])

    default_region = cfg.get("window_region", "full")
    fault_labels = set(cfg.get("fault_labels", ["mild", "moderate", "severe"]))
    if label_name == "normal":
        return str(class_regions.get("normal", cfg.get("normal_window_region", default_region)))
    if label_name in fault_labels:
        return str(class_regions.get("fault", cfg.get("fault_window_region", default_region)))
    return str(default_region)


def iter_window_starts(min_len: int, sample_length: int, stride: int, cfg: dict[str, Any], label_name: str) -> Iterator[int]:
    region = resolve_class_window_region(label_name, cfg)
    region_fraction = float(cfg.get("region_fraction", 1.0))
    return iter(resolve_window_starts(min_len, sample_length, stride, region, region_fraction))


def load_class_files(cfg: dict[str, Any], root: Path) -> dict[str, str]:
    class_files = cfg.get("class_files")
    if class_files is not None:
        return dict(class_files)
    with open(root / cfg.get("label_map", "label_map.json"), encoding="utf-8") as f:
        label_map = json.load(f)["fault_mode"]
    return {name: f"{name}.csv" for name in sorted(label_map.keys(), key=lambda k: label_map[k])}


def collect_multivariate_windows(
    root: Path,
    class_files: dict[str, str],
    value_columns: list[str],
    sample_length: int,
    stride: int,
    *,
    unit_col: str = "unit",
    cycle_col: str = "cycle",
    region_resolver: Callable[[str], str] | None = None,
    region_fraction: float = 1.0,
    whole_file_as_series: bool = False,
) -> tuple[list[np.ndarray], list[int], list[str], list[str]]:
    """Collect multivariate sliding windows from per-class CSV files."""
    label_names = list(class_files.keys())
    label_map = {name: idx for idx, name in enumerate(label_names)}

    windows: list[np.ndarray] = []
    labels: list[int] = []
    unit_ids: list[str] = []

    for label_name, csv_name in class_files.items():
        csv_path = root / csv_name
        df = pd.read_csv(csv_path)
        for column in value_columns:
            if column not in df.columns:
                raise KeyError(f"Column '{column}' not found in {csv_path}")

        label_id = label_map[label_name]
        region = region_resolver(label_name) if region_resolver else "full"

        if whole_file_as_series:
            group = df.copy()
            if cycle_col in group.columns:
                group = group.sort_values(cycle_col)
            groups = [("__series__", group)]
        else:
            groups = list(df.groupby(unit_col))

        for unit, group in groups:
            if not whole_file_as_series:
                group = group.sort_values(cycle_col)
            signals = [group[column].values.astype(np.float32) for column in value_columns]
            min_len = min(len(signal) for signal in signals)
            if min_len < sample_length:
                continue
            starts = resolve_window_starts(min_len, sample_length, stride, region, region_fraction)
            for start in starts:
                windows.append(stack_multivariate_window(signals, start, sample_length))
                labels.append(label_id)
                unit_ids.append(str(unit))

    return windows, labels, unit_ids, label_names


def maybe_swap_val_test(
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bool(cfg.get("swap_val_test", False)):
        return X_test, y_test, X_val, y_val
    return X_val, y_val, X_test, y_test


def split_train_holdout(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = int(cfg.get("seed", 42))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    test_ratio = float(cfg.get("test_ratio", 0.15))
    holdout_ratio = val_ratio + test_ratio
    return train_test_split(
        X,
        y,
        test_size=holdout_ratio,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )


def split_holdout_val_test(
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = int(cfg.get("seed", 42))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    test_ratio = float(cfg.get("test_ratio", 0.15))
    holdout_ratio = val_ratio + test_ratio
    relative_test = test_ratio / holdout_ratio if holdout_ratio > 0 else 0.5
    stratify = y_holdout if len(np.unique(y_holdout)) > 1 else None

    if cfg.get("_holdout_split") == "reversed":
        X_test, X_val, y_test, y_val = train_test_split(
            X_holdout,
            y_holdout,
            test_size=1.0 - relative_test,
            random_state=seed,
            stratify=stratify,
        )
        return X_val, y_val, X_test, y_test

    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=relative_test,
        random_state=seed,
        stratify=stratify,
    )
    return X_val, y_val, X_test, y_test
