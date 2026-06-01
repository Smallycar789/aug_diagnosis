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
    apply_global_norm,
    collect_multivariate_windows,
    load_class_files,
    maybe_swap_val_test,
    normalize_windows_per_sample,
    resolve_class_window_region,
    resolve_normalize_mode,
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
    """Load binary data from simulation CSV: head N cycles = normal, tail N cycles = degraded."""
    root = resolve_path(cfg.get("root"))
    simulation_csv = root / cfg.get("simulation_csv", "cooler_simulation_results/all_simulation.csv")
    value_columns = resolve_value_columns(cfg)
    group_col = cfg.get("group_column", "group_id")
    cycle_col = cfg.get("cycle_column", "work_cycle")
    sample_length = resolve_sample_length(cfg, default=30)
    label_names = list(cfg.get("label_names", ["normal", "degraded"]))

    df = pd.read_csv(simulation_csv)
    for column in value_columns:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in {simulation_csv}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for _, group in df.groupby(group_col):
        group = group.sort_values(cycle_col)
        if len(group) < sample_length:
            continue
        # head: first sample_length cycles → normal
        head = group.head(sample_length)
        X_list.append(np.stack([head[col].values.astype(np.float32) for col in value_columns], axis=0))
        y_list.append(0)
        # tail: last sample_length cycles → degraded
        tail = group.tail(sample_length)
        X_list.append(np.stack([tail[col].values.astype(np.float32) for col in value_columns], axis=0))
        y_list.append(1)

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
            "simulation_csv": str(simulation_csv),
            "group_column": group_col,
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
        "simulation_csv": cfg.get("simulation_csv", "cooler_simulation_results/all_simulation.csv"),
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


def _collect_multivariate_windows_from_cfg(
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    root = resolve_path(cfg["root"])
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
            f"No windows created under {root} "
            f"(sample_length={sample_length}, stride={stride})."
        )

    extra_meta = {
        "value_columns": value_columns,
        "sample_length": sample_length,
        "stride": stride,
        "source_files": {name: str(root / fname) for name, fname in class_files.items()},
    }
    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(unit_ids),
        label_names,
        extra_meta,
    )


def _partition_windows_by_split_file(
    X: np.ndarray,
    y: np.ndarray,
    unit_ids: np.ndarray,
    split_file: Path,
    buckets: dict[str, set[str]],
    *,
    unit_column: str = "unit",
    context: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    split_df = pd.read_csv(split_file)
    split_map = {
        str(unit): str(split_name)
        for unit, split_name in zip(split_df[unit_column], split_df["split"])
    }

    indices: dict[str, list[int]] = {name: [] for name in buckets}
    unknown: set[str] = set()

    for idx, unit in enumerate(unit_ids):
        split_name = split_map.get(str(unit))
        if split_name is None:
            unknown.add(str(unit))
            continue
        matched = False
        for bucket_name, split_values in buckets.items():
            if split_name in split_values:
                indices[bucket_name].append(idx)
                matched = True
                break
        if not matched:
            unknown.add(f"{unit}:{split_name}")

    if unknown:
        raise ValueError(
            f"{len(unknown)} window(s) could not be mapped for {context}: "
            f"{sorted(unknown)[:5]}"
        )

    return {
        bucket: (X[idxs], y[idxs]) if idxs else (np.empty((0, *X.shape[1:]), dtype=X.dtype), np.empty((0,), dtype=y.dtype))
        for bucket, idxs in indices.items()
    }


def _subsample_windows_by_unit(
    X: np.ndarray,
    y: np.ndarray,
    unit_ids: np.ndarray,
    *,
    ratio: float | None = None,
    max_units_per_class: int | None = None,
    label_names: list[str],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample windows by unit id (stratified by class)."""
    if len(X) == 0:
        return X, y, unit_ids
    if ratio is None and max_units_per_class is None:
        return X, y, unit_ids

    rng = np.random.default_rng(seed)
    unit_ids = np.asarray(unit_ids)
    selected_units: set[str] = set()

    for class_idx, _ in enumerate(label_names):
        class_mask = y == class_idx
        class_units = sorted({str(u) for u in unit_ids[class_mask]})
        if not class_units:
            continue

        keep_n = len(class_units)
        if ratio is not None:
            keep_n = max(1, int(round(len(class_units) * float(ratio))))
        if max_units_per_class is not None:
            keep_n = min(keep_n, int(max_units_per_class))
        keep_n = min(keep_n, len(class_units))

        picked = rng.choice(class_units, size=keep_n, replace=False)
        selected_units.update(str(u) for u in picked)

    keep_mask = np.array([str(u) in selected_units for u in unit_ids], dtype=bool)
    return X[keep_mask], y[keep_mask], unit_ids[keep_mask]


def _ensure_gen_windows_exceed_real(
    X_gen: np.ndarray,
    y_gen: np.ndarray,
    unit_gen: np.ndarray,
    *,
    X_real: np.ndarray,
    label_names: list[str],
    min_over_real: float = 1.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure generated window count is strictly greater than real source windows."""
    if len(X_real) == 0:
        return X_gen, y_gen, unit_gen

    min_required = len(X_real) + 1
    desired = max(min_required, int(len(X_real) * float(min_over_real)) + 1)

    if len(X_gen) < min_required:
        raise ValueError(
            f"Not enough generated train windows ({len(X_gen)}) to exceed "
            f"real source ({len(X_real)}). Lower real_source_fraction or gen subsample ratio."
        )
    if len(X_gen) >= desired:
        return X_gen, y_gen, unit_gen

    rng = np.random.default_rng(seed)
    unit_gen = np.asarray(unit_gen)
    selected = np.array([True] * len(X_gen), dtype=bool)

    for class_idx, _ in enumerate(label_names):
        class_mask = y_gen == class_idx
        pool_units = sorted({str(u) for u in unit_gen[class_mask]})
        picked_units = {str(u) for u in unit_gen[selected & class_mask]}
        remaining = [u for u in pool_units if u not in picked_units]
        rng.shuffle(remaining)
        for unit in remaining:
            if len(X_gen[selected]) >= desired:
                break
            unit_mask = np.array([str(u) == unit for u in unit_gen], dtype=bool)
            selected |= unit_mask

    out = (X_gen[selected], y_gen[selected], unit_gen[selected])
    if len(out[0]) <= len(X_real):
        raise ValueError(
            f"Generated source ({len(out[0])}) must exceed real source ({len(X_real)})."
        )
    return out


def _load_sensitivity_gen_baseline(cfg: dict[str, Any]) -> DiagnosisDataBundle:
    """Generated-heavy source + small real source, real MMD target, baseline random val/test."""
    sensitivity_classes = {
        "normal": "normal.csv",
        "sensitivity_degradation": "sensitivity_degradation.csv",
        "coupled_severe_fault": "coupled_severe_fault.csv",
    }
    value_columns = cfg.get("value_columns", ["avg_detectivity", "NETD_mK"])
    class_files = cfg.get("class_files", sensitivity_classes)
    sample_length = resolve_sample_length(cfg)
    stride = resolve_stride(cfg, default=32)
    generated_stride = int(cfg.get("generated_stride", stride))
    normalize_mode = resolve_normalize_mode(cfg, default="global")
    seed = int(cfg.get("seed", 42))
    real_source_fraction = float(cfg.get("real_source_fraction", 0.15))
    real_target_fraction = float(cfg.get("real_target_fraction", 0.15))
    gen_subsample_ratio = cfg.get("gen_train_subsample_ratio")
    gen_max_units_per_class = cfg.get("gen_max_units_per_class")
    gen_min_over_real = float(cfg.get("gen_min_over_real", 2.0))

    generated_root = resolve_path(cfg.get("generated_root", "data/sensitivity_generated_gan"))
    real_root = resolve_path(cfg.get("real_root", "data/sensitivity"))
    generated_split_file = generated_root / cfg.get("generated_split_file", "unit_split.csv")

    window_cfg = {
        "value_columns": value_columns,
        "class_files": class_files,
        "sample_length": sample_length,
        "stride": stride,
    }
    real_cfg = {"root": str(real_root), **window_cfg}
    gen_cfg = {"root": str(generated_root), **window_cfg, "stride": generated_stride}

    X_all, y_all, _, label_names, real_meta = _collect_multivariate_windows_from_cfg(real_cfg)
    X_gen, y_gen, unit_gen, _, gen_meta = _collect_multivariate_windows_from_cfg(gen_cfg)

    split_cfg = {
        **cfg,
        "seed": seed,
        "val_ratio": float(cfg.get("val_ratio", 0.15)),
        "test_ratio": float(cfg.get("test_ratio", cfg.get("val_ratio", 0.15))),
    }
    X_train_full, X_holdout, y_train_full, y_holdout = split_train_holdout(X_all, y_all, split_cfg)
    X_val, y_val, X_test, y_test = split_holdout_val_test(X_holdout, y_holdout, split_cfg)
    X_val, y_val, X_test, y_test = maybe_swap_val_test(X_val, y_val, X_test, y_test, cfg)

    if real_target_fraction + real_source_fraction >= 1.0:
        raise ValueError("real_source_fraction + real_target_fraction must be < 1.0")

    stratify = y_train_full if len(np.unique(y_train_full)) > 1 else None
    X_rem, X_target, y_rem, y_target = train_test_split(
        X_train_full,
        y_train_full,
        test_size=real_target_fraction,
        random_state=seed,
        stratify=stratify,
    )
    source_frac = real_source_fraction / (1.0 - real_target_fraction)
    stratify_rem = y_rem if len(np.unique(y_rem)) > 1 else None
    _, X_real_source, _, y_real_source = train_test_split(
        X_rem,
        y_rem,
        test_size=source_frac,
        random_state=seed + 1,
        stratify=stratify_rem,
    )

    split_df = pd.read_csv(generated_split_file)
    unit_col = cfg.get("unit_column", "unit")
    gen_train_unit_set = set(split_df.loc[split_df["split"] == "train", unit_col].astype(str))
    gen_train_mask = np.array([str(u) in gen_train_unit_set for u in unit_gen], dtype=bool)
    X_gen_train = X_gen[gen_train_mask]
    y_gen_train = y_gen[gen_train_mask]
    unit_gen_train = unit_gen[gen_train_mask]

    ratio = float(gen_subsample_ratio) if gen_subsample_ratio is not None else None
    max_per_class = int(gen_max_units_per_class) if gen_max_units_per_class is not None else None
    X_gen_train, y_gen_train, unit_gen_train = _subsample_windows_by_unit(
        X_gen_train,
        y_gen_train,
        unit_gen_train,
        ratio=ratio,
        max_units_per_class=max_per_class,
        label_names=label_names,
        seed=seed,
    )
    X_gen_train, y_gen_train, unit_gen_train = _ensure_gen_windows_exceed_real(
        X_gen_train,
        y_gen_train,
        unit_gen_train,
        X_real=X_real_source,
        label_names=label_names,
        min_over_real=gen_min_over_real,
        seed=seed + 2,
    )

    if len(X_gen_train) <= len(X_real_source):
        raise ValueError(
            f"Generated source ({len(X_gen_train)}) must exceed real source ({len(X_real_source)})."
        )

    X_source = np.concatenate([X_gen_train, X_real_source], axis=0)
    y_source = np.concatenate([y_gen_train, y_real_source], axis=0)

    norm_meta: dict[str, Any] = {"normalize": normalize_mode}
    if normalize_mode == "per_window":
        X_source = normalize_windows_per_sample(X_source)
        X_target = normalize_windows_per_sample(X_target)
        X_val = normalize_windows_per_sample(X_val)
        X_test = normalize_windows_per_sample(X_test)
    elif normalize_mode == "global":
        X_source, X_val, X_test, norm_meta = apply_dataset_normalization(
            X_source, X_val, X_test, mode=normalize_mode
        )
        mean = np.array(norm_meta["norm_mean"], dtype=np.float32).reshape(1, -1, 1)
        std = np.array(norm_meta["norm_std"], dtype=np.float32).reshape(1, -1, 1)
        X_target = apply_global_norm(X_target, mean, std)
    elif normalize_mode != "none":
        raise ValueError(f"Unknown normalize mode: {normalize_mode}")

    sample_shape = X_source if len(X_source) else X_val
    num_channels = sample_shape.shape[1] if sample_shape.ndim == 3 else 1
    meta = {
        "dataset": "sensitivity_gen_baseline",
        "scheme": "generated_heavy_baseline_eval",
        "domain_adaptation": True,
        "protocol": "baseline_random_val_test",
        "num_channels": num_channels,
        "num_samples": int(len(X_all) + len(X_gen)),
        "source_samples": int(len(X_source)),
        "generated_source_samples": int(len(X_gen_train)),
        "real_source_samples": int(len(X_real_source)),
        "target_samples": int(len(X_target)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "real_train_pool_samples": int(len(X_train_full)),
        "class_counts": {name: int((y_source == idx).sum()) for idx, name in enumerate(label_names)},
        "generated_root": str(generated_root),
        "real_root": str(real_root),
        "generated_split_file": str(generated_split_file),
        "stride": stride,
        "generated_stride": generated_stride,
        "real_source_fraction": real_source_fraction,
        "real_target_fraction": real_target_fraction,
        "gen_train_subsample_ratio": gen_subsample_ratio,
        "gen_min_over_real": gen_min_over_real,
        "split_mode": "random",
        "swap_val_test": bool(cfg.get("swap_val_test", True)),
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        **gen_meta,
        **{f"real_{k}": v for k, v in real_meta.items() if k not in gen_meta},
        **norm_meta,
    }

    return DiagnosisDataBundle(
        meta=meta,
        label_names=label_names,
        X_source=X_source,
        y_source=y_source,
        X_target=X_target,
        y_target=y_target,
    )


def _load_sensitivity_gen_mixed(cfg: dict[str, Any]) -> DiagnosisDataBundle:
    """Generated + small real train, real transfer (MMD), real val/test; per-window norm."""
    sensitivity_classes = {
        "normal": "normal.csv",
        "sensitivity_degradation": "sensitivity_degradation.csv",
        "coupled_severe_fault": "coupled_severe_fault.csv",
    }
    value_columns = cfg.get("value_columns", ["avg_detectivity", "NETD_mK"])
    class_files = cfg.get("class_files", sensitivity_classes)
    sample_length = resolve_sample_length(cfg)
    stride = resolve_stride(cfg, default=32)
    generated_stride = int(cfg.get("generated_stride", stride))
    real_stride = int(cfg.get("real_stride", stride))
    normalize_mode = resolve_normalize_mode(cfg, default="per_window")
    seed = int(cfg.get("seed", 42))
    gen_subsample_ratio = cfg.get("gen_train_subsample_ratio")
    gen_max_units_per_class = cfg.get("gen_max_units_per_class")

    generated_root = resolve_path(cfg.get("generated_root", "data/sensitivity_generated"))
    real_root = resolve_path(cfg.get("real_root", "data/sensitivity"))
    generated_split_file = generated_root / cfg.get("generated_split_file", "unit_split.csv")
    real_split_file = real_root / cfg.get("real_split_file", "unit_split_gen_mixed_v1.csv")

    window_cfg = {
        "value_columns": value_columns,
        "class_files": class_files,
        "sample_length": sample_length,
    }
    gen_cfg = {"root": str(generated_root), **window_cfg, "stride": generated_stride}
    real_cfg = {"root": str(real_root), **window_cfg, "stride": real_stride}

    X_gen, y_gen, unit_gen, label_names, gen_meta = _collect_multivariate_windows_from_cfg(gen_cfg)
    X_real, y_real, unit_real, _, real_meta = _collect_multivariate_windows_from_cfg(real_cfg)

    split_df = pd.read_csv(generated_split_file)
    unit_col = cfg.get("unit_column", "unit")
    gen_train_unit_set = set(split_df.loc[split_df["split"] == "train", unit_col].astype(str))
    gen_train_mask = np.array([str(u) in gen_train_unit_set for u in unit_gen], dtype=bool)
    X_gen_train = X_gen[gen_train_mask]
    y_gen_train = y_gen[gen_train_mask]
    unit_gen_train = unit_gen[gen_train_mask]

    ratio = float(gen_subsample_ratio) if gen_subsample_ratio is not None else None
    max_per_class = int(gen_max_units_per_class) if gen_max_units_per_class is not None else None
    X_gen_train, y_gen_train, unit_gen_train = _subsample_windows_by_unit(
        X_gen_train,
        y_gen_train,
        unit_gen_train,
        ratio=ratio,
        max_units_per_class=max_per_class,
        label_names=label_names,
        seed=seed,
    )

    real_parts = _partition_windows_by_split_file(
        X_real,
        y_real,
        unit_real,
        real_split_file,
        {
            "train_real": {"train_real"},
            "transfer": {"transfer"},
            "val": {"val"},
            "test": {"test"},
        },
        context="real sensitivity gen_mixed",
    )

    X_train_real, y_train_real = real_parts["train_real"]
    X_target, y_target = real_parts["transfer"]
    X_val, y_val = real_parts["val"]
    X_test, y_test = real_parts["test"]

    if len(X_train_real) and len(X_gen_train) < len(X_train_real):
        raise ValueError(
            f"Generated train windows ({len(X_gen_train)}) < real train windows ({len(X_train_real)}). "
            "Increase gen_train_subsample_ratio or reduce real train split."
        )

    X_source = np.concatenate([X_gen_train, X_train_real], axis=0) if len(X_train_real) else X_gen_train
    y_source = np.concatenate([y_gen_train, y_train_real], axis=0) if len(y_train_real) else y_gen_train

    if len(X_source) == 0:
        raise ValueError("Mixed source train is empty; check generated CSVs and split files.")
    if len(X_target) == 0:
        raise ValueError("MMD transfer split is empty; run prepare_sensitivity_gen_mixed_split.py first.")
    if len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Real val/test splits are empty; check unit_split_gen_mixed_v1.csv.")

    norm_meta: dict[str, Any] = {"normalize": normalize_mode}
    if normalize_mode == "per_window":
        X_source = normalize_windows_per_sample(X_source)
        X_target = normalize_windows_per_sample(X_target)
        X_val = normalize_windows_per_sample(X_val)
        X_test = normalize_windows_per_sample(X_test)
    elif normalize_mode == "global":
        X_source, X_val, X_test, norm_meta = apply_dataset_normalization(
            X_source, X_val, X_test, mode=normalize_mode
        )
        mean = np.array(norm_meta["norm_mean"], dtype=np.float32).reshape(1, -1, 1)
        std = np.array(norm_meta["norm_std"], dtype=np.float32).reshape(1, -1, 1)
        X_target = apply_global_norm(X_target, mean, std)
    elif normalize_mode != "none":
        raise ValueError(f"Unknown normalize mode: {normalize_mode}")

    sample_shape = X_source if len(X_source) else X_val
    num_channels = sample_shape.shape[1] if sample_shape.ndim == 3 else 1
    meta = {
        "dataset": "sensitivity_gen_mixed",
        "scheme": "generated_plus_real_mmd",
        "domain_adaptation": True,
        "num_channels": num_channels,
        "num_samples": int(len(X_gen) + len(X_real)),
        "source_samples": int(len(X_source)),
        "generated_source_samples": int(len(X_gen_train)),
        "real_source_samples": int(len(X_train_real)),
        "target_samples": int(len(X_target)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "class_counts": {name: int((y_source == idx).sum()) for idx, name in enumerate(label_names)},
        "generated_root": str(generated_root),
        "real_root": str(real_root),
        "generated_split_file": str(generated_split_file),
        "real_split_file": str(real_split_file),
        "stride": stride,
        "generated_stride": generated_stride,
        "real_stride": real_stride,
        "gen_train_subsample_ratio": gen_subsample_ratio,
        "gen_max_units_per_class": gen_max_units_per_class,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        **gen_meta,
        **{f"real_{k}": v for k, v in real_meta.items() if k not in gen_meta},
        **norm_meta,
    }

    return DiagnosisDataBundle(
        meta=meta,
        label_names=label_names,
        X_source=X_source,
        y_source=y_source,
        X_target=X_target,
        y_target=y_target,
    )


def _load_sensitivity_gen_transfer(cfg: dict[str, Any]) -> DiagnosisDataBundle:
    """Generated sensitivity source + real transfer/test for CNN-BiGRU MMD experiments."""
    sensitivity_classes = {
        "normal": "normal.csv",
        "sensitivity_degradation": "sensitivity_degradation.csv",
        "coupled_severe_fault": "coupled_severe_fault.csv",
    }
    value_columns = cfg.get("value_columns", ["avg_detectivity", "NETD_mK"])
    class_files = cfg.get("class_files", sensitivity_classes)
    sample_length = resolve_sample_length(cfg)
    generated_stride = int(cfg.get("generated_stride", sample_length))
    real_stride = int(cfg.get("real_stride", cfg.get("stride", 32)))
    normalize_mode = resolve_normalize_mode(cfg)

    generated_root = resolve_path(cfg.get("generated_root", "data/sensitivity_generated"))
    real_root = resolve_path(cfg.get("real_root", "data/sensitivity"))
    generated_split_file = generated_root / cfg.get("generated_split_file", "unit_split.csv")
    real_split_file = real_root / cfg.get("real_split_file", "unit_split_transfer_15.csv")

    gen_cfg = {
        "root": str(generated_root),
        "value_columns": value_columns,
        "class_files": class_files,
        "sample_length": sample_length,
        "stride": generated_stride,
    }
    real_cfg = {
        "root": str(real_root),
        "value_columns": value_columns,
        "class_files": class_files,
        "sample_length": sample_length,
        "stride": real_stride,
    }

    X_gen, y_gen, unit_gen, label_names, gen_meta = _collect_multivariate_windows_from_cfg(gen_cfg)
    X_real, y_real, unit_real, _, real_meta = _collect_multivariate_windows_from_cfg(real_cfg)

    gen_parts = _partition_windows_by_split_file(
        X_gen,
        y_gen,
        unit_gen,
        generated_split_file,
        {"source": {"train"}, "val": {"val"}},
        context="generated sensitivity",
    )
    real_parts = _partition_windows_by_split_file(
        X_real,
        y_real,
        unit_real,
        real_split_file,
        {"target": {"transfer"}, "test": {"test"}},
        context="real sensitivity",
    )

    X_source, y_source = gen_parts["source"]
    X_val, y_val = gen_parts["val"]
    X_target, y_target = real_parts["target"]
    X_test, y_test = real_parts["test"]

    norm_meta: dict[str, Any] = {"normalize": normalize_mode}
    if normalize_mode == "global":
        if len(X_source) == 0:
            raise ValueError("Generated train split is empty; run prepare_sensitivity_from_degradation.py first.")
        X_source, X_val, X_test, norm_meta = apply_dataset_normalization(
            X_source, X_val, X_test, mode=normalize_mode
        )
        mean = np.array(norm_meta["norm_mean"], dtype=np.float32).reshape(1, -1, 1)
        std = np.array(norm_meta["norm_std"], dtype=np.float32).reshape(1, -1, 1)
        X_target = apply_global_norm(X_target, mean, std)
    elif normalize_mode == "per_window":
        X_source = normalize_windows_per_sample(X_source)
        X_val = normalize_windows_per_sample(X_val)
        X_target = normalize_windows_per_sample(X_target)
        X_test = normalize_windows_per_sample(X_test)
    elif normalize_mode != "none":
        raise ValueError(f"Unknown normalize mode: {normalize_mode}")

    sample_shape = X_source if len(X_source) else (X_val if len(X_val) else X_test)
    num_channels = sample_shape.shape[1] if sample_shape.ndim == 3 else 1
    meta = {
        "dataset": "sensitivity_gen_transfer",
        "scheme": "generated_source_real_transfer",
        "domain_adaptation": True,
        "num_channels": num_channels,
        "num_samples": int(len(X_gen) + len(X_real)),
        "source_samples": int(len(X_source)),
        "val_samples": int(len(X_val)),
        "target_samples": int(len(X_target)),
        "test_samples": int(len(X_test)),
        "class_counts": {name: int((y_source == idx).sum()) for idx, name in enumerate(label_names)},
        "generated_root": str(generated_root),
        "real_root": str(real_root),
        "generated_split_file": str(generated_split_file),
        "real_split_file": str(real_split_file),
        "generated_stride": generated_stride,
        "real_stride": real_stride,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        **gen_meta,
        **{f"real_{k}": v for k, v in real_meta.items() if k not in gen_meta},
        **norm_meta,
    }

    return DiagnosisDataBundle(
        meta=meta,
        label_names=label_names,
        X_source=X_source,
        y_source=y_source,
        X_target=X_target,
        y_target=y_target,
    )


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
    if name == "sensitivity_gen_transfer":
        return _load_sensitivity_gen_transfer(dataset_cfg)
    if name == "sensitivity_gen_mixed":
        return _load_sensitivity_gen_mixed(dataset_cfg)
    if name == "sensitivity_gen_baseline":
        return _load_sensitivity_gen_baseline(dataset_cfg)
    if name == "sifuqi":
        return _load_sifuqi(dataset_cfg)

    raise ValueError(f"Unknown dataset name: {name}")
