"""GAN-VAE 1D data augmentation — migrated from test_pre/jupyter_test/augmetation/GAN-VAE.ipynb"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_aug.common import compare_signals, split_data
from data_aug.data_load import AugDataBundle
from data_aug.io_utils import get_device, save_checkpoint_best, save_json, save_loss_history


class Encoder(nn.Module):
    def __init__(self, latent_dim=1000, input_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = 64 * 16
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        b = x.shape[0]
        x = x.view(b, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def encode(self, x):
        return self.forward(x)


class Generator(nn.Module):
    def __init__(self, latent_dim=1000, output_size=128):
        super().__init__()
        self.output_size = output_size
        self.fc = nn.Linear(latent_dim, 64 * 16)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        b = z.shape[0]
        x = self.fc(z)
        x = x.view(b, 64, 16)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size=128):
        super().__init__()
        self.input_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten_dim = 64 * 32
        self.fc1 = nn.Linear(self.flatten_dim, 1024)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(1024, 4)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        b = x.shape[0]
        x = x.view(b, -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.out_activation(x)
        return x


class VAE_GAN(nn.Module):
    def __init__(self, latent_dim=1000, seq_len=128):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, input_size=seq_len)
        self.generator = Generator(latent_dim=latent_dim, output_size=seq_len)
        self.discriminator = Discriminator(input_size=seq_len)
        self.latent_dim = latent_dim
        self.seq_len = seq_len

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.generator(z)
        d_real = self.discriminator(x)
        d_fake = self.discriminator(recon)
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "d_real": d_real,
            "d_fake": d_fake,
        }


def prepare_for_comparison(segments, model, device, ts_min=None, ts_max=None):
    original_samples = []
    reconstructed_samples = []

    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(segments))):
            orig_segment = segments[i].copy()

            sample_tensor = torch.FloatTensor(orig_segment).unsqueeze(0).unsqueeze(0).to(device)
            mu, logvar = model.encoder(sample_tensor)
            z = model.reparameterize(mu, logvar)
            recon = model.generator(z)
            recon_np = recon.cpu().numpy().squeeze()

            if ts_min is not None and ts_max is not None:
                orig_segment = orig_segment * (ts_max - ts_min) + ts_min
                recon_np = recon_np * (ts_max - ts_min) + ts_min

            original_samples.append(orig_segment.flatten())
            reconstructed_samples.append(recon_np.flatten())

    return original_samples, reconstructed_samples


def _prepare_data(bundle: AugDataBundle, cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    seq_len = model_cfg.get("seq_len", 128)
    overlap_ratio = model_cfg.get("overlap_ratio", 0.5)
    batch_size = model_cfg.get("batch_size", 32)

    ts_data = bundle.raw_data.astype(np.float32)
    ts_min = ts_data.min()
    ts_max = ts_data.max()
    if ts_max - ts_min > 0:
        ts_normalized = (ts_data - ts_min) / (ts_max - ts_min)
    else:
        ts_normalized = ts_data - ts_min

    segments = split_data(ts_normalized, seq_len, overlap_ratio)
    segments_tensor = torch.FloatTensor(segments).unsqueeze(1)
    dataset = TensorDataset(segments_tensor, torch.zeros(len(segments_tensor)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    return ts_data, ts_min, ts_max, ts_normalized, segments, segments_tensor, dataloader


def train(bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path):
    device = get_device(cfg)
    model_cfg = cfg["model"]

    ts_data, ts_min, ts_max, ts_normalized, segments, segments_tensor, dataloader = _prepare_data(bundle, cfg)

    latent_dim = model_cfg.get("latent_dim", 64)
    seq_len = model_cfg.get("seq_len", 128)
    epochs = model_cfg.get("epochs", 50)
    lr = model_cfg.get("lr", 1e-4)
    alpha = model_cfg.get("alpha", 0.01)

    model = VAE_GAN(latent_dim=latent_dim, seq_len=seq_len).to(device)
    optimizer_EG = optim.Adam(
        list(model.encoder.parameters()) + list(model.generator.parameters()), lr=lr
    )
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr)

    history = {"d_loss": [], "g_loss": [], "recon_loss": [], "kl_loss": [], "total_loss": []}
    best_recon = float("inf")
    best_epoch = 0

    print("开始训练...")
    for epoch in range(epochs):
        epoch_d_loss = epoch_g_loss = epoch_recon_loss = epoch_kl_loss = 0.0
        num_batches = 0

        for real_data, _ in dataloader:
            real_data = real_data.to(device)
            batch_size_actual = real_data.size(0)

            model.discriminator.zero_grad()
            d_real = model.discriminator(real_data)

            with torch.no_grad():
                mu, logvar = model.encoder(real_data)
                z = model.reparameterize(mu, logvar)
                fake_data = model.generator(z)

            d_fake = model.discriminator(fake_data.detach())

            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)

            d_loss_real = F.binary_cross_entropy(d_real, real_labels)
            d_loss_fake = F.binary_cross_entropy(d_fake, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_loss.backward()
            optimizer_D.step()

            model.encoder.zero_grad()
            model.generator.zero_grad()

            mu, logvar = model.encoder(real_data)
            z = model.reparameterize(mu, logvar)
            fake_data = model.generator(z)
            d_fake = model.discriminator(fake_data)

            recon_loss = F.mse_loss(fake_data, real_data)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / batch_size_actual

            g_adv_loss = F.binary_cross_entropy(d_fake, real_labels)
            g_loss = recon_loss + g_adv_loss + alpha * kl_loss

            g_loss.backward()
            optimizer_EG.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1

        avg_d = epoch_d_loss / max(num_batches, 1)
        avg_g = epoch_g_loss / max(num_batches, 1)
        avg_recon = epoch_recon_loss / max(num_batches, 1)
        avg_kl = epoch_kl_loss / max(num_batches, 1)

        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)
        history["total_loss"].append(avg_g)

        if avg_recon < best_recon:
            best_recon = avg_recon
            best_epoch = epoch + 1
            save_checkpoint_best(
                out_dir,
                {"model_state_dict": model.state_dict()},
                epoch=best_epoch,
                best_metric=best_recon,
                metric_name="train_recon_loss",
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{epochs}] | "
                f"D_loss: {avg_d:.6f} | G_loss: {avg_g:.6f} | "
                f"Recon: {avg_recon:.6f} | KL: {avg_kl:.6f}"
            )

    _plot_gan_vae_curves(history, out_dir / "loss_curves.png")
    save_loss_history({**history, "epochs": epochs, "best_epoch": best_epoch}, out_dir)

    meta = {
        "ts_min": float(ts_min),
        "ts_max": float(ts_max),
        "segments": segments,
        "segments_tensor": segments_tensor,
        "best_epoch": best_epoch,
        "best_recon_loss": best_recon,
    }
    save_json({"ts_min": float(ts_min), "ts_max": float(ts_max), "seq_len": seq_len, "latent_dim": latent_dim}, out_dir / "norm_params.json")
    return model, meta


def _plot_gan_vae_curves(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history["d_loss"], label="D")
    axes[0, 0].plot(history["g_loss"], label="G")
    axes[0, 0].legend()
    axes[0, 0].set_title("Adversarial Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["recon_loss"], "r-")
    axes[0, 1].set_title("Reconstruction Loss")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history["kl_loss"], "g-")
    axes[1, 0].set_title("KL Loss")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["total_loss"], "b-")
    axes[1, 1].set_title("Total G Loss")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def load_checkpoint(path: Path, cfg: dict[str, Any]) -> VAE_GAN:
    device = get_device(cfg)
    model_cfg = cfg["model"]
    model = VAE_GAN(
        latent_dim=model_cfg.get("latent_dim", 64),
        seq_len=model_cfg.get("seq_len", 128),
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate(model: VAE_GAN, bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path) -> np.ndarray:
    device = get_device(cfg)
    model = model.to(device)
    model_cfg = cfg["model"]

    _, ts_min, ts_max, _, segments, _, _ = _prepare_data(bundle, cfg)
    num_generate = model_cfg.get("num_generate", 5)

    random_generated_samples = []
    with torch.no_grad():
        for _ in range(num_generate):
            random_z = torch.randn(1, model.latent_dim).to(device)
            random_gen = model.generator(random_z)
            random_gen_np = random_gen.cpu().numpy().squeeze()
            random_gen_np = random_gen_np * (ts_max - ts_min) + ts_min
            random_generated_samples.append(random_gen_np.reshape(1, -1))

    generated = np.concatenate(random_generated_samples, axis=0)
    np.save(out_dir / "generated_samples.npy", generated)
    return generated


def evaluate(
    bundle: AugDataBundle,
    out_dir: Path,
    cfg: dict[str, Any],
    generated_samples: np.ndarray | None = None,
    model: VAE_GAN | None = None,
    meta: dict | None = None,
) -> dict[str, Any]:
    device = get_device(cfg)
    sample_rate = float(cfg["dataset"].get("sample_rate", bundle.meta.get("sample_rate", 12000)))

    _, ts_min, ts_max, _, segments, _, _ = _prepare_data(bundle, cfg)

    if model is None:
        model = load_checkpoint(out_dir / "checkpoint_best.pth", cfg)
    model = model.to(device)

    original_samples, reconstructed_samples = prepare_for_comparison(
        segments, model, device, ts_min, ts_max
    )
    spectrum_stats = compare_signals(
        [s.reshape(1, -1) for s in original_samples],
        [s.reshape(1, -1) for s in reconstructed_samples],
        save_dir=str(out_dir),
        sample_rate=sample_rate,
    )

    if generated_samples is None:
        gen_path = out_dir / "generated_samples.npy"
        generated_samples = np.load(gen_path) if gen_path.exists() else np.array([])

    all_random = generated_samples if len(generated_samples) else np.array(reconstructed_samples)
    all_original = np.concatenate([s.reshape(1, -1) for s in original_samples], axis=0)

    metrics = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "dataset": cfg["dataset"]["name"],
        "model": "gan_vae",
        "statistics": {
            **spectrum_stats,
            "random_generated": {
                "mean": float(np.mean(all_random)),
                "std": float(np.std(all_random)),
                "min": float(np.min(all_random)),
                "max": float(np.max(all_random)),
            },
            "original": {
                "mean": float(np.mean(all_original)),
                "std": float(np.std(all_original)),
                "min": float(np.min(all_original)),
                "max": float(np.max(all_original)),
            },
        },
    }

    if meta:
        metrics["training_best"] = {
            "best_epoch": meta.get("best_epoch"),
            "best_recon_loss": meta.get("best_recon_loss"),
        }

    save_json(metrics, out_dir / "metrics.json")
    np.save(out_dir / "original_samples.npy", original_samples)
    np.save(out_dir / "reconstructed_samples.npy", reconstructed_samples)
    return metrics
