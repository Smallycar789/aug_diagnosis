"""Unified data loading for fault diagnosis datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diagnosis.common import (
    apply_dataset_normalization,
    collect_multivariate_windows,
    load_class_files,
    maybe_swap_val_test,
    normalize_windows_per_sample,
    resolve_class_window_region,
    resolve_sample_length,
    resolve_stride,
    resolve_value_columns,
    resolve_window_starts,
    split_holdout_val_test,
    split_train_holdout,
)
from diagnosis.io_utils import resolve_path


@dataclass
class DiagnosisDataBundle:
    meta: dict[str, Any] = field(default_factory=dict)
    label_names: list[str] = field(default_factory=list)
    # cnn_bigru domain adaptation
    X_source: np.ndarray | None = None
    y_source: np.ndarray | None = None
    X_target: np.ndarray | None = None
    y_target: np.ndarray | None = None
    # resnet / generic splits (images or tensors)
    train_items: list | None = None
    val_items: list | None = None
    test_items: list | None = None
    # tl_meta raw arrays for CWRUDataset-style construction
    full_signals: np.ndarray | None = None
    full_labels: np.ndarray | None = None
    train_indices: np.ndarray | None = None
    val_indices: np.ndarray | None = None
    test_indices: np.ndarray | None = None


def load_cwru_sliding_windows(
    csv_path: str | Path,
    sample_length: int = 1024,
    stride: int = 512,
    zscore_per_sample: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """From 1DCNN-BiGRU.ipynb load_cwru_data — logic unchanged."""
    df = pd.read_csv(resolve_path(csv_path))
    label_names = df.columns.tolist()

    X_list = []
    y_list = []

    for label_idx, col_name in enumerate(label_names):
        signal = df[col_name].dropna().values
        n_samples = (len(signal) - sample_length) // stride + 1

        for i in range(n_samples):
            start = i * stride
            end = start + sample_length
            sample = signal[start:end]
            X_list.append(sample)
            y_list.append(label_idx)

    X = np.array(X_list).astype(np.float32)
    y = np.array(y_list).astype(np.int64)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    if zscore_per_sample:
        X = normalize_windows_per_sample(X)

    return X, y, label_names


def _finalize_classification_bundle(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
    *,
    dataset_name: str,
    label_names: list[str],
    total_samples: int,
    extra_meta: dict[str, Any] | None = None,
) -> DiagnosisDataBundle:
    domain_adaptation = bool(cfg.get("domain_adaptation", False))
    normalize_mode = cfg.get("normalize", "global")

    X_val, y_val, X_test, y_test = maybe_swap_val_test(X_val, y_val, X_test, y_test, cfg)
    X_train, X_val, X_test, norm_meta = apply_dataset_normalization(
        X_train, X_val, X_test, mode=normalize_mode
    )

    sample_shape = X_train if len(X_train) else (X_val if len(X_val) else X_test)
    num_channels = sample_shape.shape[1] if sample_shape.ndim == 3 else 1
    meta = {
        "dataset": dataset_name,
        "scheme": "domain_adaptation" if domain_adaptation else "classification",
        "domain_adaptation": domain_adaptation,
        "split_mode": cfg.get("split_mode", "random"),
        "swap_val_test": bool(cfg.get("swap_val_test", False)),
        "num_channels": num_channels,
        "num_samples": int(total_samples),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "class_counts": {name: int((y_train == idx).sum()) for idx, name in enumerate(label_names)},
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        **norm_meta,
    }
    if extra_meta:
        meta.update(extra_meta)

    return DiagnosisDataBundle(
        meta=meta,
        label_names=label_names,
        X_source=X_train,
        y_source=y_train,
        X_target=X_val if domain_adaptation else None,
        y_target=y_val if domain_adaptation else None,
    )


def _split_classification_bundle(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict[str, Any],
    *,
    dataset_name: str,
    label_names: list[str],
    extra_meta: dict[str, Any] | None = None,
) -> DiagnosisDataBundle:
    X_train, X_holdout, y_train, y_holdout = split_train_holdout(X, y, cfg)
    X_val, y_val, X_test, y_test = split_holdout_val_test(X_holdout, y_holdout, cfg)

    return _finalize_classification_bundle(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        cfg,
        dataset_name=dataset_name,
        label_names=label_names,
        total_samples=len(X),
        extra_meta=extra_meta,
    )


def _split_classification_bundle_by_unit(
    X: np.ndarray,
    y: np.ndarray,
    unit_ids: np.ndarray,
    cfg: dict[str, Any],
    *,
    dataset_name: str,
    label_names: list[str],
    extra_meta: dict[str, Any] | None = None,
) -> DiagnosisDataBundle:
    root = resolve_path(cfg.get("root"))
    split_df = pd.read_csv(root / cfg.get("split_file", "unit_split.csv"))
    unit_col = cfg.get("unit_column", "unit")
    split_map = dict(zip(split_df[unit_col], split_df["split"]))

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    unknown_units: set[str] = set()

    for idx, unit in enumerate(unit_ids):
        split_name = split_map.get(unit)
        if split_name == "train":
            train_idx.append(idx)
        elif split_name == "val":
            val_idx.append(idx)
        elif split_name == "test":
            test_idx.append(idx)
        else:
            unknown_units.add(str(unit))
            train_idx.append(idx)

    if unknown_units:
        raise ValueError(
            f"{len(unknown_units)} unit(s) missing from split file for {dataset_name}: "
            f"{sorted(unknown_units)[:5]}"
        )

    return _finalize_classification_bundle(
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
        cfg,
        dataset_name=dataset_name,
        label_names=label_names,
        total_samples=len(X),
        extra_meta={
            **(extra_meta or {}),
            "split_file": str(root / cfg.get("split_file", "unit_split.csv")),
            "unit_split_counts": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
        },
    )


def _load_head_tail_group_windows(cfg: dict[str, Any], dataset_name: str) -> DiagnosisDataBundle:
    """Load binary data from head/tail CSVs with one window per group."""
    root = resolve_path(cfg.get("root"))
    head_csv = root / cfg.get("head_csv", "cooler_simulation_results/all_head_30.csv")
    tail_csv = root / cfg.get("tail_csv", "cooler_simulation_results/all_tail_30.csv")
    value_columns = resolve_value_columns(cfg)
    group_col = cfg.get("group_column", "group_id")
    cycle_col = cfg.get("cycle_column", "work_cycle")
    sample_length = resolve_sample_length(cfg, default=30)
    label_names = list(cfg.get("label_names", ["normal", "degraded"]))

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    source_files: dict[str, str] = {}

    for label_id, (label_name, csv_path) in enumerate([(label_names[0], head_csv), (label_names[1], tail_csv)]):
        df = pd.read_csv(csv_path)
        for column in value_columns:
            if column not in df.columns:
                raise KeyError(f"Column '{column}' not found in {csv_path}")
        source_files[label_name] = str(csv_path)
        for _, group in df.groupby(group_col):
            group = group.sort_values(cycle_col)
            if len(group) != sample_length:
                continue
            window = np.stack([group[column].values.astype(np.float32) for column in value_columns], axis=0)
            X_list.append(window)
            y_list.append(label_id)

    if not X_list:
        raise ValueError(
            f"No windows created for {dataset_name} head/tail loader "
            f"(sample_length={sample_length}, groups={group_col})."
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    return _split_classification_bundle(
        X,
        y,
        cfg,
        dataset_name=dataset_name,
        label_names=label_names,
        extra_meta={
            "value_columns": value_columns,
            "sample_length": sample_length,
            "head_csv": str(head_csv),
            "tail_csv": str(tail_csv),
            "group_column": group_col,
            "source_files": source_files,
        },
    )


def _load_multivariate_class_files(cfg: dict[str, Any], dataset_name: str) -> DiagnosisDataBundle:
    """Load multi-class data from per-class CSV files with multivariate sliding windows."""
    root = resolve_path(cfg.get("root"))
    value_columns = resolve_value_columns(cfg)
    sample_length = resolve_sample_length(cfg)
    stride = resolve_stride(cfg)
    class_files = load_class_files(cfg, root)

    windows, labels, unit_ids, label_names = collect_multivariate_windows(
        root,
        class_files,
        value_columns,
        sample_length,
        stride,
        unit_col=cfg.get("unit_column", "unit"),
        cycle_col=cfg.get("cycle_column", "cycle"),
    )

    if not windows:
        raise ValueError(
            f"No windows created for {dataset_name} "
            f"(sample_length={sample_length}, stride={stride})."
        )

    X = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    unit_ids_arr = np.array(unit_ids)
    extra_meta = {
        "value_columns": value_columns,
        "sample_length": sample_length,
        "stride": stride,
        "source_files": {name: str(root / fname) for name, fname in class_files.items()},
    }

    if cfg.get("split_mode", "random") == "unit":
        return _split_classification_bundle_by_unit(
            X,
            y,
            unit_ids_arr,
            cfg,
            dataset_name=dataset_name,
            label_names=label_names,
            extra_meta=extra_meta,
        )

    return _split_classification_bundle(
        X,
        y,
        cfg,
        dataset_name=dataset_name,
        label_names=label_names,
        extra_meta=extra_meta,
    )


def _load_cooler(cfg: dict[str, Any]) -> DiagnosisDataBundle:
    """Binary cooler diagnosis: all_head_30=normal, all_tail_30=degraded, one window per group."""
    cfg = {
        **cfg,
        "head_csv": cfg.get("head_csv", "cooler_simulation_results/all_head_30.csv"),
        "tail_csv": cfg.get("tail_csv", "cooler_simulation_results/all_tail_30.csv"),
        "value_columns": cfg.get(
            "value_columns",
            ["T_stable_K", "t_cool_hours", "sigma_T_K"],
        ),
        "sample_length": resolve_sample_length(cfg, default=30),
        "group_column": cfg.get("group_column", "group_id"),
        "cycle_column": cfg.get("cycle_column", "work_cycle"),
        "label_names": cfg.get("label_names", ["normal", "degraded"]),
        "domain_adaptation": bool(cfg.get("domain_adaptation", False)),
        "normalize": cfg.get("normalize", "global"),
    }
    return _load_head_tail_group_windows(cfg, "cooler")


def _load_sensitivity(cfg: dict[str, Any]) -> DiagnosisDataBundle:
    """Three-class sensitivity diagnosis using avg_detectivity + NETD_mK."""
    cfg = {
        **cfg,
        "value_columns": cfg.get("value_columns", ["avg_detectivity", "NETD_mK"]),
        "class_files": cfg.get(
            "class_files",
            {
                "normal": "normal.csv",
                "sensitivity_degradation": "sensitivity_degradation.csv",
                "coupled_severe_fault": "coupled_severe_fault.csv",
            },
        ),
        "sample_length": resolve_sample_length(cfg),
        "stride": resolve_stride(cfg),
        "split_mode": cfg.get("split_mode", "unit"),
        "split_file": cfg.get("split_file", "unit_split.csv"),
        "swap_val_test": bool(cfg.get("swap_val_test", False)),
        "domain_adaptation": bool(cfg.get("domain_adaptation", False)),
        "normalize": cfg.get("normalize", "global"),
    }
    return _load_multivariate_class_files(cfg, "sensitivity")


def _load_image_quality(cfg: dict[str, Any]) -> DiagnosisDataBundle:
    """Five-class image quality diagnosis using MTF50 + response_nonuniformity + bad_pixel_rate."""
    cfg = {
        **cfg,
        "root": cfg.get("root", "data/image_quality"),
        "value_columns": cfg.get(
            "value_columns",
            ["MTF50", "response_nonuniformity", "bad_pixel_rate"],
        ),
        "class_files": cfg.get(
            "class_files",
            {
                "normal": "normal.csv",
                "mtf_degradation": "mtf_degradation.csv",
                "nonuniformity_degradation": "nonuniformity_degradation.csv",
                "bad_pixel_degradation": "bad_pixel_degradation.csv",
                "coupled_severe_fault": "coupled_severe_fault.csv",
            },
        ),
        "sample_length": resolve_sample_length(cfg),
        "stride": resolve_stride(cfg),
        "domain_adaptation": bool(cfg.get("domain_adaptation", False)),
        "normalize": cfg.get("normalize", "global"),
    }
    return _load_multivariate_class_files(cfg, "image_quality")


def _load_sifuqi(cfg: dict[str, Any]) -> DiagnosisDataBundle:
    """Load four degradation levels (normal/mild/moderate/severe) for multi-class diagnosis."""
    root = resolve_path(cfg.get("root", "data/sifuqi"))
    sample_length = resolve_sample_length(cfg, default=256)
    stride = resolve_stride(cfg, default=64)
    region_fraction = float(cfg.get("region_fraction", 0.25))
    value_columns = resolve_value_columns(cfg)
    default_levels = {
        "normal": "servo_normal.csv",
        "mild": "servo_mild.csv",
        "moderate": "servo_moderate.csv",
        "severe": "servo_severe.csv",
    }
    level_files = cfg.get("levels", default_levels)
    label_names = list(level_files.keys())

    windows, labels, _, _ = collect_multivariate_windows(
        root,
        level_files,
        value_columns,
        sample_length,
        stride,
        cycle_col=cfg.get("cycle_column", "cycle"),
        region_resolver=lambda name: resolve_class_window_region(name, cfg),
        region_fraction=region_fraction,
        whole_file_as_series=True,
    )

    if not windows:
        raise ValueError(
            f"No windows created for sifuqi (sample_length={sample_length}, stride={stride}). "
            "Check CSV length and window settings."
        )

    region_counts = {name: 0 for name in label_names}
    label_map = {name: idx for idx, name in enumerate(label_names)}
    for label_id in labels:
        region_counts[label_names[label_id]] += 1

    X = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    split_cfg = {**cfg, "_holdout_split": "reversed"}

    return _split_classification_bundle(
        X,
        y,
        split_cfg,
        dataset_name="sifuqi",
        label_names=label_names,
        extra_meta={
            "value_columns": value_columns,
            "sample_length": sample_length,
            "stride": stride,
            "region_fraction": region_fraction,
            "class_window_regions": {
                name: resolve_class_window_region(name, cfg) for name in label_names
            },
            "region_window_counts": region_counts,
            "source_files": {name: str(root / fname) for name, fname in level_files.items()},
        },
    )


def load_data(dataset_cfg: dict[str, Any], model_name: str, seed: int = 42) -> DiagnosisDataBundle:
    name = dataset_cfg["name"]
    dataset_cfg = {**dataset_cfg, "seed": seed}

    if name == "cwru_csv":
        sample_length = resolve_sample_length(dataset_cfg, default=1024)
        stride = resolve_stride(dataset_cfg, default=512)
        source_csv = dataset_cfg.get("source_csv", dataset_cfg.get("csv", "CWRU_12k_1797_10c.csv"))
        target_csv = dataset_cfg.get("target_csv", "CWRU_12k_1750_10c.csv")

        X_source, y_source, label_names = load_cwru_sliding_windows(source_csv, sample_length, stride)
        X_target, y_target, _ = load_cwru_sliding_windows(target_csv, sample_length, stride)

        val_ratio = float(dataset_cfg.get("val_ratio", 0.2))
        X_train, X_val, y_train, y_val = train_test_split(
            X_source, y_source, test_size=val_ratio, random_state=seed, stratify=y_source
        )

        bundle = DiagnosisDataBundle(
            meta={
                "dataset": "cwru_csv",
                "source_csv": str(resolve_path(source_csv)),
                "target_csv": str(resolve_path(target_csv)),
            },
            label_names=label_names,
            X_source=X_train,
            y_source=y_train,
            X_target=X_target,
            y_target=y_target,
        )
        bundle.meta["X_val"] = X_val
        bundle.meta["y_val"] = y_val
        bundle.meta["X_test"] = X_target
        bundle.meta["y_test"] = y_target
        return bundle

    if name in ("degradation", "image_quality"):
        return _load_image_quality(dataset_cfg)
    if name == "cooler":
        return _load_cooler(dataset_cfg)
    if name == "sensitivity":
        return _load_sensitivity(dataset_cfg)
    if name == "sifuqi":
        return _load_sifuqi(dataset_cfg)

    raise ValueError(f"Unknown dataset name: {name}")
