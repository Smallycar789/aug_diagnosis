"""Shared helpers copied from notebooks (logic unchanged)."""

from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normalize_to_minus1_1(data):
    """归一化到[-1, 1]范围"""
    data_min = np.min(data)
    data_max = np.max(data)
    normalized = 2 * ((data - data_min) / (data_max - data_min)) - 1
    return normalized, data_min, data_max


def denormalize_from_minus1_1(normalized_data, data_min, data_max):
    """从[-1,1]反归一化"""
    return (normalized_data + 1) * (data_max - data_min) / 2 + data_min


def split_data(data, window_size, overlap_ratio):
    step = int(window_size * (1 - overlap_ratio))
    if step <= 0:
        step = 1

    segments = []
    for start in range(0, len(data) - window_size + 1, step):
        segment = data[start : start + window_size]
        segments.append(segment)

    return np.array(segments)


def create_sequences(data, seq_len, stride=1):
    """Split 1D series into windows: (num_samples, seq_len, 1)."""
    sequences = []
    for i in range(0, len(data) - seq_len + 1, stride):
        sequences.append(data[i : i + seq_len])
    return np.array(sequences)[..., np.newaxis]


def evaluate_samples(original, generated):
    """评估生成样本质量 (from VAE.ipynb)"""
    orig = original.flatten()
    gen = generated.flatten()

    metrics = {}

    metrics["mean_diff"] = float(abs(np.mean(orig) - np.mean(gen)))
    metrics["std_diff"] = float(abs(np.std(orig) - np.std(gen)))

    if len(orig) > 1:
        n = min(1000, len(orig), len(gen))
        metrics["correlation"] = float(np.corrcoef(orig[:n], gen[:n])[0, 1])

    hist_orig, bins = np.histogram(orig, bins=50, density=True)
    hist_gen, _ = np.histogram(gen, bins=bins, density=True)
    hist_orig = hist_orig + 1e-10
    hist_gen = hist_gen + 1e-10
    m = 0.5 * (hist_orig + hist_gen)
    js_div = 0.5 * (
        np.sum(hist_orig * np.log(hist_orig / m))
        + np.sum(hist_gen * np.log(hist_gen / m))
    )
    metrics["js_divergence"] = float(js_div)

    metrics["original"] = {
        "mean": float(np.mean(orig)),
        "std": float(np.std(orig)),
        "min": float(np.min(orig)),
        "max": float(np.max(orig)),
    }
    metrics["generated"] = {
        "mean": float(np.mean(gen)),
        "std": float(np.std(gen)),
        "min": float(np.min(gen)),
        "max": float(np.max(gen)),
    }
    return metrics


def plot_loss_curves_vae(history, save_path="loss_curves.png"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history["total_loss"], "b-", linewidth=2)
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history["recon_loss"], "r-", linewidth=2)
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(history["kl_loss"], "g-", linewidth=2)
    plt.title("KL Divergence Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curves_gan(history, save_path="loss_curves.png"):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.plot(history.get("total_loss", []), "b-", linewidth=2)
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 2)
    plt.plot(history.get("recon_loss", []), "r-", linewidth=2)
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 3)
    plt.plot(history.get("g_loss", []), "g-", linewidth=2)
    plt.title("Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 4)
    plt.plot(history.get("d_loss", []), "m-", linewidth=2)
    plt.title("Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _flatten_feature_matrix(data):
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr.reshape(-1, arr.shape[-1])
    return arr.reshape(-1, arr.shape[-1])


def _relative_percent(diff, baseline, eps=1e-12):
    if abs(baseline) < eps:
        return None
    return float(diff / abs(baseline) * 100.0)


def mean_bias_metrics(original, generated, feature_names=None):
    """Mean-bias metrics on raw values, keeping feature scale visible in JSON."""
    orig = _flatten_feature_matrix(original)
    gen = _flatten_feature_matrix(generated)
    if len(orig) == 0 or len(gen) == 0:
        return {}

    n_features = orig.shape[-1]
    if gen.shape[-1] != n_features:
        return {}
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f"feature_{idx}" for idx in range(n_features)]

    per_feature = {}
    for idx, name in enumerate(feature_names):
        original_mean = float(np.mean(orig[:, idx]))
        generated_mean = float(np.mean(gen[:, idx]))
        signed_error = generated_mean - original_mean
        abs_error = abs(signed_error)
        per_feature[name] = {
            "original_mean": original_mean,
            "generated_mean": generated_mean,
            "mean_error": float(signed_error),
            "mean_abs_error": float(abs_error),
            "mean_error_percent": _relative_percent(signed_error, original_mean),
            "mean_abs_error_percent": _relative_percent(abs_error, original_mean),
        }

    original_mean = float(np.mean(orig))
    generated_mean = float(np.mean(gen))
    signed_error = generated_mean - original_mean
    abs_error = abs(signed_error)
    return {
        "overall": {
            "original_mean": original_mean,
            "generated_mean": generated_mean,
            "mean_error": float(signed_error),
            "mean_abs_error": float(abs_error),
            "mean_error_percent": _relative_percent(signed_error, original_mean),
            "mean_abs_error_percent": _relative_percent(abs_error, original_mean),
        },
        "per_feature": per_feature,
    }


def _feature_count(data):
    arr = np.asarray(data)
    if arr.ndim >= 3:
        return int(arr.shape[-1])
    if arr.ndim == 2 and arr.shape[-1] > 1:
        return int(arr.shape[-1])
    return 1


def _normalize_multifeature_for_display(original, generated, feature_names=None):
    """Per-feature min-max on combined orig+gen; used only for plots (v2 style)."""
    original_plot = np.asarray(original, dtype=float).copy()
    generated_plot = np.asarray(generated, dtype=float).copy()
    if original_plot.ndim < 3 or original_plot.shape[-1] <= 1:
        return original_plot, generated_plot, None

    n_features = original_plot.shape[-1]
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f"feature_{idx}" for idx in range(n_features)]

    combined = np.concatenate(
        [
            original_plot.reshape(-1, n_features),
            generated_plot.reshape(-1, n_features),
        ],
        axis=0,
    )
    feature_min = np.min(combined, axis=0)
    feature_max = np.max(combined, axis=0)
    scale = feature_max - feature_min
    scale[scale < 1e-12] = 1.0
    original_plot = (original_plot - feature_min) / scale
    generated_plot = (generated_plot - feature_min) / scale
    return original_plot, generated_plot, {
        "mode": "per_feature_minmax",
        "applied_to": list(feature_names),
        "note": "Only signal/frequency plots use normalized values; metrics stay on raw scale.",
    }


def _save_signal_comparison_plot(
    original_stack,
    generated_stack,
    save_path,
    feature_name=None,
):
    n_plots = len(original_stack)
    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    ylabel = feature_name or "Amplitude"
    for i in range(n_plots):
        orig_y = np.asarray(original_stack[i]).squeeze()
        gen_y = np.asarray(generated_stack[i]).squeeze()
        title_suffix = f" | {feature_name}" if feature_name else ""
        axes[i, 0].plot(orig_y, "b-", linewidth=1)
        axes[i, 0].set_title(f"Original{title_suffix} | window {i + 1}")
        axes[i, 0].set_xlabel("Time step")
        axes[i, 0].set_ylabel(ylabel)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(gen_y, "r-", linewidth=1)
        axes[i, 1].set_title(f"Generated{title_suffix} | window {i + 1}")
        axes[i, 1].set_xlabel("Time step")
        axes[i, 1].set_ylabel(ylabel)
        axes[i, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_class_multifeature_plots(
    original_stack,
    generated_stack,
    save_dir,
    filename_prefix,
    feature_names,
    sample_rate,
    spectrum_plot_style,
    window_idx=0,
):
    """One signal + one frequency figure per class; rows = features (GAN-style combined)."""
    n_features = len(feature_names)
    win = min(window_idx, len(original_stack) - 1)
    signal_path = f"{save_dir}/{filename_prefix}signal_comparison.png"
    freq_path = f"{save_dir}/{filename_prefix}frequency_comparison.png"

    fig, axes = plt.subplots(n_features, 2, figsize=(14, 2.8 * n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)
    for feat_idx, feat_name in enumerate(feature_names):
        orig_y = np.asarray(original_stack[win, ..., feat_idx]).reshape(-1)
        gen_y = np.asarray(generated_stack[win, ..., feat_idx]).reshape(-1)
        axes[feat_idx, 0].plot(orig_y, "b-", linewidth=1)
        axes[feat_idx, 0].set_title(f"Original | {feat_name}")
        axes[feat_idx, 0].set_xlabel("Time step")
        axes[feat_idx, 0].set_ylabel(feat_name)
        axes[feat_idx, 0].grid(True, alpha=0.3)

        axes[feat_idx, 1].plot(gen_y, "r-", linewidth=1)
        axes[feat_idx, 1].set_title(f"Generated | {feat_name}")
        axes[feat_idx, 1].set_xlabel("Time step")
        axes[feat_idx, 1].set_ylabel(feat_name)
        axes[feat_idx, 1].grid(True, alpha=0.3)
    fig.suptitle(f"{filename_prefix.rstrip('_')} — multi-parameter comparison (window {win + 1})", y=1.01)
    plt.tight_layout()
    plt.savefig(signal_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(n_features, 2, figsize=(14, 3.0 * n_features))
    if n_features == 1:
        axes = axes.reshape(1, -1)
    for feat_idx, feat_name in enumerate(feature_names):
        orig_flat = np.asarray(original_stack[win, ..., feat_idx]).reshape(-1)
        gen_flat = np.asarray(generated_stack[win, ..., feat_idx]).reshape(-1)
        for col, flat, color, tag in (
            (0, orig_flat, "b", "Original"),
            (1, gen_flat, "r", "Generated"),
        ):
            n_fft = len(flat)
            fft_vals = np.abs(np.fft.fft(flat))
            freqs = np.fft.fftfreq(n_fft, 1 / sample_rate)
            half = len(freqs) // 2
            x_f = freqs[:half]
            y_f = fft_vals[:half]
            if spectrum_plot_style == "vline":
                axes[feat_idx, col].vlines(x_f, 0, y_f, color=color, alpha=0.7, linewidth=0.8)
            else:
                axes[feat_idx, col].plot(x_f, y_f, f"{color}-", alpha=0.7)
            axes[feat_idx, col].set_title(f"{tag} | {feat_name}")
            axes[feat_idx, col].set_xlabel("Frequency (Hz)")
            axes[feat_idx, col].set_ylabel(feat_name)
            axes[feat_idx, col].grid(True, alpha=0.3)
            if sample_rate > 10:
                axes[feat_idx, col].set_xlim([0, sample_rate / 2])
    fig.suptitle(f"{filename_prefix.rstrip('_')} — frequency per parameter (window {win + 1})", y=1.01)
    plt.tight_layout()
    plt.savefig(freq_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "signal_comparison": f"{filename_prefix}signal_comparison.png",
        "frequency_comparison": f"{filename_prefix}frequency_comparison.png",
        "window_index": int(win),
    }


def _save_frequency_comparison_plot(
    original_flat,
    generated_flat,
    save_path,
    sample_rate,
    spectrum_plot_style,
    ylabel,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    n_fft = len(original_flat)

    fft_original = np.abs(np.fft.fft(original_flat))
    freq_original = np.fft.fftfreq(n_fft, 1 / sample_rate)
    half = len(freq_original) // 2
    x_original = freq_original[:half]
    y_original = fft_original[:half]
    if spectrum_plot_style == "vline":
        axes[0].vlines(x_original, 0, y_original, color="b", alpha=0.7, linewidth=0.8)
    else:
        axes[0].plot(x_original, y_original, "b-", alpha=0.7)
    axes[0].set_title("Frequency Spectrum - Original")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.3)
    if sample_rate > 10:
        axes[0].set_xlim([0, sample_rate / 2])

    fft_generated = np.abs(np.fft.fft(generated_flat))
    freq_generated = np.fft.fftfreq(len(generated_flat), 1 / sample_rate)
    half_g = len(freq_generated) // 2
    x_generated = freq_generated[:half_g]
    y_generated = fft_generated[:half_g]
    if spectrum_plot_style == "vline":
        axes[1].vlines(x_generated, 0, y_generated, color="r", alpha=0.7, linewidth=0.8)
    else:
        axes[1].plot(x_generated, y_generated, "r-", alpha=0.7)
    axes[1].set_title("Frequency Spectrum - Generated")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel(ylabel)
    axes[1].grid(True, alpha=0.3)
    if sample_rate > 10:
        axes[1].set_xlim([0, sample_rate / 2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compare_signals(
    original,
    generated,
    save_dir="./results",
    sample_rate=12000,
    filename_prefix="",
    spectrum_plot_style="line",
    feature_names=None,
    split_feature_plots=None,
    use_physical_plot_scale=True,
    num_compare=None,
):
    """Compare original vs generated windows; multi-feature plots are saved per channel."""
    os.makedirs(save_dir, exist_ok=True)

    n_available = min(len(original), len(generated))
    n_plots = min(num_compare or 5, n_available)
    n_features = _feature_count(original[0] if len(original) else generated[0])
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f"feature_{idx}" for idx in range(n_features)]
    if split_feature_plots is None:
        split_feature_plots = n_features > 1

    original_stack = np.stack([np.asarray(original[i]) for i in range(n_plots)], axis=0)
    generated_stack = np.stack([np.asarray(generated[i]) for i in range(n_plots)], axis=0)
    original_plot_stack = original_stack
    generated_plot_stack = generated_stack
    display_norm = None
    if n_features > 1 and not use_physical_plot_scale:
        original_plot_stack, generated_plot_stack, display_norm = _normalize_multifeature_for_display(
            original_stack, generated_stack, feature_names
        )

    feature_plot_files = {}
    if n_features > 1 and split_feature_plots:
        for feat_idx, feat_name in enumerate(feature_names):
            orig_feat_stack = original_plot_stack[..., feat_idx]
            gen_feat_stack = generated_plot_stack[..., feat_idx]
            signal_name = f"{filename_prefix}{feat_name}_signal_comparison.png"
            freq_name = f"{filename_prefix}{feat_name}_frequency_comparison.png"
            _save_signal_comparison_plot(
                orig_feat_stack,
                gen_feat_stack,
                f"{save_dir}/{signal_name}",
                feature_name=feat_name,
            )
            orig_flat = orig_feat_stack.reshape(-1)
            gen_flat = gen_feat_stack.reshape(-1)
            _save_frequency_comparison_plot(
                orig_flat,
                gen_flat,
                f"{save_dir}/{freq_name}",
                sample_rate,
                spectrum_plot_style,
                ylabel=feat_name,
            )
            feat_corr = None
            if len(orig_flat) > 1 and len(gen_flat) > 1:
                feat_corr = float(np.corrcoef(orig_flat, gen_flat)[0, 1])
            feature_plot_files[feat_name] = {
                "signal_comparison": signal_name,
                "frequency_comparison": freq_name,
                "correlation": feat_corr,
            }
    else:
        signal_name = f"{filename_prefix}signal_comparison.png"
        freq_name = f"{filename_prefix}frequency_comparison.png"
        if n_features > 1:
            combined = _save_class_multifeature_plots(
                original_plot_stack,
                generated_plot_stack,
                save_dir,
                filename_prefix,
                feature_names,
                sample_rate,
                spectrum_plot_style,
                window_idx=0,
            )
            feature_plot_files["combined"] = combined
        else:
            _save_signal_comparison_plot(
                original_plot_stack,
                generated_plot_stack,
                f"{save_dir}/{signal_name}",
            )
            original_plot_all = np.concatenate([original_plot_stack[i] for i in range(n_plots)], axis=0)
            generated_plot_all = np.concatenate([generated_plot_stack[i] for i in range(n_plots)], axis=0)
            amp_label = "Normalized Amplitude" if display_norm else "Amplitude"
            _save_frequency_comparison_plot(
                original_plot_all.reshape(-1),
                generated_plot_all.reshape(-1),
                f"{save_dir}/{freq_name}",
                sample_rate,
                spectrum_plot_style,
                amp_label,
            )

    original_all = np.concatenate([original_stack[i] for i in range(n_plots)], axis=0)
    generated_all = np.concatenate([generated_stack[i] for i in range(n_plots)], axis=0)
    original_flat = original_all.reshape(-1)
    generated_flat = generated_all.reshape(-1)

    if n_features > 1 and split_feature_plots:
        plot_mode = "per_feature_physical_split" if use_physical_plot_scale else "per_feature_split"
    elif n_features > 1 and use_physical_plot_scale:
        plot_mode = "class_multifeature_combined_physical"
    elif display_norm:
        plot_mode = "per_feature_minmax_display"
    elif n_features > 1:
        plot_mode = "multi_feature_combined"
    else:
        plot_mode = "single_feature"

    stats = {
        "original_mean": float(np.mean(original_all)),
        "original_std": float(np.std(original_all)),
        "generated_mean": float(np.mean(generated_all)),
        "generated_std": float(np.std(generated_all)),
        "mean_bias": mean_bias_metrics(original_all, generated_all, feature_names=feature_names),
        "plot_mode": plot_mode,
        "feature_plots": feature_plot_files,
    }
    if display_norm:
        stats["display_normalization"] = display_norm
    if len(original_flat) > 1 and len(generated_flat) > 1:
        stats["correlation"] = float(np.corrcoef(original_flat, generated_flat)[0, 1])
    return stats


def autocorr(x, lag=50):
    return np.array([np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, lag + 1)])
