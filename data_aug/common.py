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


def compare_signals(original, generated, save_dir="./results", sample_rate=12000):
    """对比原始信号和生成信号 (from GAN.ipynb / GAN-VAE.ipynb)"""
    os.makedirs(save_dir, exist_ok=True)

    n_plots = min(5, len(generated))
    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 3 * n_plots))

    if n_plots == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_plots):
        axes[i, 0].plot(original[i].squeeze(), "b-", linewidth=1)
        axes[i, 0].set_title(f"Original Signal {i + 1}")
        axes[i, 0].set_xlabel("Sample")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(generated[i].squeeze(), "r-", linewidth=1)
        axes[i, 1].set_title(f"Generated Signal {i + 1}")
        axes[i, 1].set_xlabel("Sample")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/signal_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    original_all = np.concatenate(original[:n_plots], axis=0)
    generated_all = np.concatenate(generated[:n_plots], axis=0)

    stats = {
        "original_mean": float(np.mean(original_all)),
        "original_std": float(np.std(original_all)),
        "generated_mean": float(np.mean(generated_all)),
        "generated_std": float(np.std(generated_all)),
    }
    if len(original_all.squeeze()) > 1 and len(generated_all.squeeze()) > 1:
        stats["correlation"] = float(
            np.corrcoef(original_all.squeeze(), generated_all.squeeze())[0, 1]
        )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    fft_original = np.abs(np.fft.fft(original_all.squeeze()))
    freq_original = np.fft.fftfreq(len(original_all), 1 / sample_rate)
    axes[0].plot(
        freq_original[: len(freq_original) // 2],
        fft_original[: len(fft_original) // 2],
        "b-",
        alpha=0.7,
    )
    axes[0].set_title("Frequency Spectrum - Original")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    if sample_rate > 10:
        axes[0].set_xlim([0, sample_rate / 2])

    fft_generated = np.abs(np.fft.fft(generated_all.squeeze()))
    freq_generated = np.fft.fftfreq(len(generated_all), 1 / sample_rate)
    axes[1].plot(
        freq_generated[: len(freq_generated) // 2],
        fft_generated[: len(fft_generated) // 2],
        "r-",
        alpha=0.7,
    )
    axes[1].set_title("Frequency Spectrum - Generated")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    if sample_rate > 10:
        axes[1].set_xlim([0, sample_rate / 2])

    plt.tight_layout()
    plt.savefig(f"{save_dir}/frequency_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    return stats


def autocorr(x, lag=50):
    return np.array([np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, lag + 1)])
