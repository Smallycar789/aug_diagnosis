"""Shared normalization and evaluation helpers for data augmentation models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from data_aug.common import denormalize_from_minus1_1, mean_bias_metrics


def load_norm_params(out_dir: Path) -> dict[str, Any]:
    norm_path = out_dir / "norm_params.json"
    if not norm_path.exists():
        return {}
    with open(norm_path, encoding="utf-8") as f:
        return json.load(f)


def normalize_class_windows(class_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class min-max to [-1, 1] for window tensors (N, T, F)."""
    dmin = class_raw.min(axis=(0, 1), keepdims=True).astype(np.float32)
    dmax = class_raw.max(axis=(0, 1), keepdims=True).astype(np.float32)
    denom = np.where((dmax - dmin) < 1e-8, 1.0, dmax - dmin)
    segments = (2 * ((class_raw - dmin) / denom) - 1).astype(np.float32)
    return segments, dmin, dmax


def normalize_windows_minus1_1(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global per-feature min-max to [-1, 1] for (N, T, F) windows."""
    dmin = raw.min(axis=(0, 1), keepdims=True).astype(np.float32)
    dmax = raw.max(axis=(0, 1), keepdims=True).astype(np.float32)
    denom = np.where((dmax - dmin) < 1e-8, 1.0, dmax - dmin)
    normalized = (2 * ((raw - dmin) / denom) - 1).astype(np.float32)
    return normalized, dmin, dmax


def compute_norm_stats(ts_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min/max for GAN-VAE style [0, 1] normalization."""
    if ts_data.ndim == 3:
        ts_min = ts_data.min(axis=(0, 1), keepdims=True).astype(np.float32)
        ts_max = ts_data.max(axis=(0, 1), keepdims=True).astype(np.float32)
    else:
        ts_min = np.array(np.min(ts_data), dtype=np.float32)
        ts_max = np.array(np.max(ts_data), dtype=np.float32)
    return ts_min, ts_max


def normalize_unit_interval(ts_data: np.ndarray, ts_min, ts_max) -> np.ndarray:
    denom = np.where((ts_max - ts_min) < 1e-8, 1.0, ts_max - ts_min)
    return ((ts_data - ts_min) / denom).astype(np.float32)


def denormalize_minus1_1_array(data: np.ndarray, data_min, data_max) -> np.ndarray:
    """Inverse of [-1, 1] window normalization (RVAE / GAN)."""
    return denormalize_from_minus1_1(np.asarray(data, dtype=np.float32), data_min, data_max).astype(
        np.float32
    )


def denormalize_unit_interval_array(data: np.ndarray, data_min, data_max) -> np.ndarray:
    """Inverse of [0, 1] normalization (GAN-VAE)."""
    arr = np.asarray(data, dtype=np.float32)
    dmin = np.squeeze(np.asarray(data_min, dtype=np.float32))
    dmax = np.squeeze(np.asarray(data_max, dtype=np.float32))
    if dmin.ndim == 0:
        scale = float(dmax - dmin)
        if abs(scale) < 1e-8:
            scale = 1.0
        return arr * scale + float(dmin)
    scale = dmax - dmin
    scale = np.where(scale < 1e-8, 1.0, scale)
    return arr * scale + dmin


def _arrays_from_norm_entry(
    entry: dict[str, Any],
    min_keys: tuple[str, ...] = ("data_min", "ts_min"),
    max_keys: tuple[str, ...] = ("data_max", "ts_max"),
) -> tuple[np.ndarray | None, np.ndarray | None]:
    dmin = dmax = None
    for key in min_keys:
        if key in entry:
            dmin = np.asarray(entry[key], dtype=np.float32)
            break
    for key in max_keys:
        if key in entry:
            dmax = np.asarray(entry[key], dtype=np.float32)
            break
    return dmin, dmax


def resolve_class_norm(
    label_name: str | None,
    norm_params: dict[str, Any],
    fallback_min=None,
    fallback_max=None,
    *,
    min_keys: tuple[str, ...] = ("data_min", "ts_min"),
    max_keys: tuple[str, ...] = ("data_max", "ts_max"),
) -> tuple[Any, Any]:
    """Resolve per-class or global normalization bounds from norm_params.json."""
    per_class_norm = norm_params.get("per_class_norm", {})
    if label_name and label_name in per_class_norm:
        dmin, dmax = _arrays_from_norm_entry(per_class_norm[label_name], min_keys, max_keys)
        if dmin is not None and dmax is not None:
            return dmin, dmax
    if fallback_min is not None and fallback_max is not None:
        return np.asarray(fallback_min, dtype=np.float32), np.asarray(fallback_max, dtype=np.float32)
    for gmin, gmax in zip(min_keys, max_keys):
        if gmin in norm_params and gmax in norm_params:
            return (
                np.asarray(norm_params[gmin], dtype=np.float32),
                np.asarray(norm_params[gmax], dtype=np.float32),
            )
    return None, None


def build_per_class_norm_sequences(
    raw_data: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    *,
    normalize_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] = normalize_class_windows,
    min_key: str = "data_min",
    max_key: str = "data_max",
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    """Concatenate per-class normalized windows and record class-wise bounds."""
    sequences_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    per_class_norm: dict[str, dict[str, Any]] = {}
    for label_id, label_name in enumerate(label_names):
        class_mask = labels == label_id
        class_raw = raw_data[class_mask]
        if len(class_raw) == 0:
            continue
        class_norm, class_min, class_max = normalize_fn(class_raw)
        sequences_list.append(class_norm)
        labels_list.append(np.full(len(class_norm), label_id, dtype=np.int64))
        per_class_norm[label_name] = {
            min_key: np.asarray(class_min).tolist(),
            max_key: np.asarray(class_max).tolist(),
        }
    sequences = np.concatenate(sequences_list, axis=0).astype(np.float32)
    out_labels = np.concatenate(labels_list, axis=0)
    return sequences, out_labels, per_class_norm


def physical_statistics(
    original_phys: np.ndarray,
    generated_phys: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Mean/std and Pearson correlation on denormalized windows."""
    original_phys = np.asarray(original_phys, dtype=np.float32)
    generated_phys = np.asarray(generated_phys, dtype=np.float32)
    stats: dict[str, Any] = {
        "original_mean": float(np.mean(original_phys)),
        "original_std": float(np.std(original_phys)),
        "generated_mean": float(np.mean(generated_phys)),
        "generated_std": float(np.std(generated_phys)),
        "mean_bias": mean_bias_metrics(original_phys, generated_phys, feature_names=feature_names),
        "metric_scale": "physical",
    }
    o_flat = original_phys.reshape(-1)
    g_flat = generated_phys.reshape(-1)
    if len(o_flat) > 1 and len(g_flat) > 1:
        stats["correlation"] = float(np.corrcoef(o_flat, g_flat)[0, 1])
    return stats


def strip_plot_compare_stats(plot_stats: dict[str, Any]) -> dict[str, Any]:
    """Drop scalar stats that are merged via physical_statistics()."""
    exclude = {
        "original_mean",
        "original_std",
        "generated_mean",
        "generated_std",
        "mean_bias",
        "correlation",
    }
    return {k: v for k, v in plot_stats.items() if k not in exclude}


def compute_sample_diversity(
    generated_samples: np.ndarray,
    n_check: int = 10,
) -> dict[str, Any]:
    from scipy.spatial.distance import cdist

    n_check = min(n_check, len(generated_samples))
    selected = np.asarray(generated_samples[:n_check], dtype=np.float32).reshape(n_check, -1)
    distances = cdist(selected, selected, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)
    return {
        "min_nearest_neighbor": float(min_distances.min()),
        "mean_nearest_neighbor": float(min_distances.mean()),
        "max_nearest_neighbor": float(min_distances.max()),
        "diversity_ok": bool(min_distances.mean() >= 0.1),
    }


def calibrate_feature_means(
    generated: np.ndarray,
    reference: np.ndarray,
    feature_columns: list[str],
    calibrate_features: list[str] | None,
) -> np.ndarray:
    if not calibrate_features:
        return generated
    out = np.asarray(generated, dtype=np.float32).copy()
    ref = np.asarray(reference, dtype=np.float32)
    for name in calibrate_features:
        if name not in feature_columns:
            continue
        idx = feature_columns.index(name)
        ref_mean = float(ref[..., idx].mean())
        gen_mean = float(out[..., idx].mean())
        if abs(gen_mean) > 1e-8:
            out[..., idx] = out[..., idx] * (ref_mean / gen_mean)
        else:
            out[..., idx] = out[..., idx] + (ref_mean - gen_mean)
    return out


def denormalize_by_label(
    samples: np.ndarray,
    label_indices: np.ndarray | None,
    label_names: list[str],
    norm_params: dict[str, Any],
    fallback_min,
    fallback_max,
) -> np.ndarray:
    """Denormalize each sample using its class bounds when labels are available."""
    if label_indices is None or not label_names:
        return denormalize_minus1_1_array(samples, fallback_min, fallback_max)
    return np.stack(
        [
            denormalize_minus1_1_array(
                samples[i : i + 1],
                *resolve_class_norm(
                    label_names[int(label_indices[i])],
                    norm_params,
                    fallback_min,
                    fallback_max,
                ),
            )[0]
            for i in range(len(samples))
        ],
        axis=0,
    )
