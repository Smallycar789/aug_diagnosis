"""VAE data augmentation — migrated from test_pre/jupyter_test/augmetation/VAE.ipynb"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_aug.common import (
    autocorr,
    denormalize_from_minus1_1,
    evaluate_samples,
    normalize_to_minus1_1,
    plot_loss_curves_vae,
)
from data_aug.data_load import AugDataBundle
from data_aug.io_utils import get_device, save_checkpoint_best, save_json, save_loss_history


class VAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=64):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(input_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        self.fc2 = nn.Linear(latent_dim, 400)
        self.fc3 = nn.Linear(400, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return torch.tanh(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=0.01):
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = torch.sum(kl_loss)
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def _prepare_data(bundle: AugDataBundle, cfg: dict[str, Any]):
    raw_data = bundle.raw_data.astype(np.float32)
    data_norm, data_min, data_max = normalize_to_minus1_1(raw_data)
    data_tensor = torch.FloatTensor(data_norm).reshape(-1, 1)
    batch_size = min(cfg["model"].get("batch_size", 256), len(data_tensor))
    train_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)
    return raw_data, data_norm, data_min, data_max, train_loader


def train(bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path):
    device = get_device(cfg)
    model_cfg = cfg["model"]
    raw_data, data_norm, data_min, data_max, train_loader = _prepare_data(bundle, cfg)

    input_dim = model_cfg.get("input_dim", 1)
    latent_dim = model_cfg.get("latent_dim", 64)
    epochs = model_cfg.get("epochs", 50)
    lr = model_cfg.get("lr", 0.01)
    beta = model_cfg.get("beta", 0.01)

    vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    train_history = {"total_loss": [], "recon_loss": [], "kl_loss": []}
    best_recon = float("inf")
    best_epoch = 0

    print(f"\nTraining VAE for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        vae.train()
        for batch in train_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(data)
            total_loss, recon_loss, kl_loss = loss_function(recon, data, mu, logvar, beta=beta)
            total_loss.backward()
            optimizer.step()
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

        avg_total = epoch_total_loss / len(train_loader.dataset)
        avg_recon = epoch_recon_loss / len(train_loader.dataset)
        avg_kl = epoch_kl_loss / len(train_loader.dataset)

        train_history["total_loss"].append(avg_total)
        train_history["recon_loss"].append(avg_recon)
        train_history["kl_loss"].append(avg_kl)

        if avg_recon < best_recon:
            best_recon = avg_recon
            best_epoch = epoch + 1
            save_checkpoint_best(
                out_dir,
                {"model_state_dict": vae.state_dict()},
                epoch=epoch + 1,
                best_metric=best_recon,
                metric_name="train_recon_loss",
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Total={avg_total:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}"
            )

    plot_loss_curves_vae(train_history, str(out_dir / "loss_curves.png"))
    save_loss_history({**train_history, "epochs": epochs, "best_epoch": best_epoch}, out_dir)

    meta = {
        "raw_data": raw_data,
        "data_min": float(data_min),
        "data_max": float(data_max),
        "best_epoch": best_epoch,
        "best_recon_loss": best_recon,
    }
    return vae, meta


def load_checkpoint(path: Path, cfg: dict[str, Any]) -> VAE:
    device = get_device(cfg)
    model_cfg = cfg["model"]
    vae = VAE(
        input_dim=model_cfg.get("input_dim", 1),
        latent_dim=model_cfg.get("latent_dim", 64),
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    return vae


def generate(model: VAE, bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path) -> np.ndarray:
    device = get_device(cfg)
    model = model.to(device)
    model.eval()

    raw_data = bundle.raw_data.astype(np.float32)
    _, data_min, data_max = normalize_to_minus1_1(raw_data)
    num_samples = cfg["model"].get("num_generate", len(raw_data))

    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        generated_norm = model.decode(z).cpu().numpy()

    generated_original = denormalize_from_minus1_1(generated_norm, data_min, data_max)

    np.save(out_dir / "generated_samples.npy", generated_original)
    save_json(
        {"data_min": float(data_min), "data_max": float(data_max), "num_samples": int(num_samples)},
        out_dir / "norm_params.json",
    )
    return generated_original


def evaluate(
    bundle: AugDataBundle,
    out_dir: Path,
    cfg: dict[str, Any],
    generated_samples: np.ndarray | None = None,
) -> dict[str, Any]:
    raw_data = bundle.raw_data.astype(np.float32)

    if generated_samples is None:
        gen_path = out_dir / "generated_samples.npy"
        if not gen_path.exists():
            raise FileNotFoundError(f"Generated samples not found: {gen_path}")
        generated_original = np.load(gen_path)
    else:
        generated_original = generated_samples

    metrics = evaluate_samples(raw_data, generated_original)
    metrics["experiment"] = cfg.get("experiment", {}).get("name")
    metrics["dataset"] = cfg["dataset"]["name"]
    metrics["model"] = "vae"

    save_json(metrics, out_dir / "metrics.json")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].plot(raw_data[:200], "b-", alpha=0.7, linewidth=1)
    axes[0, 0].set_title("Original Data (first 200 points)")
    axes[0, 0].set_xlabel("Index")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(generated_original[:200].flatten(), "r-", alpha=0.7, linewidth=1)
    axes[0, 1].set_title("Generated Data (first 200 points)")
    axes[0, 1].set_xlabel("Index")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].hist(raw_data, bins=50, alpha=0.5, label="Original", density=True)
    axes[0, 2].hist(generated_original.flatten(), bins=50, alpha=0.5, label="Generated", density=True)
    axes[0, 2].set_title("Distribution Comparison")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    gen_flat = generated_original.flatten()
    n_scatter = min(1000, len(raw_data), len(gen_flat))
    n_qq = min(len(raw_data), len(gen_flat))

    axes[1, 0].scatter(raw_data[:n_scatter], gen_flat[:n_scatter], alpha=0.5, s=1)
    axes[1, 0].plot([np.min(raw_data), np.max(raw_data)], [np.min(raw_data), np.max(raw_data)], "r--", alpha=0.5)
    axes[1, 0].set_title("Original vs Generated")
    axes[1, 0].grid(True, alpha=0.3)

    lag = 30
    axes[1, 1].plot(autocorr(raw_data, lag), "b-", label="Original", alpha=0.7)
    axes[1, 1].plot(autocorr(generated_original.flatten(), lag), "r-", label="Generated", alpha=0.7)
    axes[1, 1].set_title(f"Autocorrelation (lag={lag})")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].scatter(np.sort(raw_data[:n_qq]), np.sort(gen_flat[:n_qq]), alpha=0.5, s=1)
    axes[1, 2].plot([np.min(raw_data), np.max(raw_data)], [np.min(raw_data), np.max(raw_data)], "r--", alpha=0.5)
    axes[1, 2].set_title("QQ Plot (Sorted Values)")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "vae_comprehensive_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    output_df = pd.DataFrame({"original": raw_data[:n_qq], "generated": gen_flat[:n_qq]})
    output_df.to_csv(out_dir / "vae_generated_results.csv", index=False)

    return metrics
