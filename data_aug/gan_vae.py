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

from data_aug.common import compare_signals, mean_bias_metrics, split_data
from data_aug.data_load import AugDataBundle
from data_aug.io_utils import get_device, save_checkpoint_best, save_json, save_loss_history
from data_aug.shared import (
    calibrate_feature_means,
    compute_norm_stats,
    denormalize_unit_interval_array,
    load_norm_params,
    normalize_unit_interval,
    resolve_class_norm,
)


class Encoder(nn.Module):
    def __init__(self, latent_dim=1000, input_size=128, input_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.input_channels = input_channels

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=4, stride=2, padding=1),
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

        conv_len = input_size
        for _ in range(3):
            conv_len = (conv_len + 2 - 4) // 2 + 1
        self.flatten_dim = 64 * conv_len
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
    def __init__(self, latent_dim=1000, output_size=128, output_channels=1):
        super().__init__()
        self.output_size = output_size
        self.output_channels = output_channels
        self.base_len = max(1, output_size // 8)
        self.fc = nn.Linear(latent_dim, 64 * self.base_len)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(16, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        b = z.shape[0]
        x = self.fc(z)
        x = x.view(b, 64, self.base_len)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        if x.shape[-1] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode="linear", align_corners=False)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size=128, input_channels=1):
        super().__init__()
        self.input_size = input_size
        self.input_channels = input_channels

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten_dim = 64 * 32
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        self.fc1 = nn.Linear(self.flatten_dim, 1024)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(1024, 4)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.adaptive_pool(x)
        b = x.shape[0]
        x = x.view(b, -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.out_activation(x)
        return x


class VAE_GAN(nn.Module):
    def __init__(self, latent_dim=1000, seq_len=128, input_channels=1):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, input_size=seq_len, input_channels=input_channels)
        self.generator = Generator(latent_dim=latent_dim, output_size=seq_len, output_channels=input_channels)
        self.discriminator = Discriminator(input_size=seq_len, input_channels=input_channels)
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.input_channels = input_channels

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


def _to_model_tensor(segments):
    if segments.ndim == 3:
        return torch.FloatTensor(segments).permute(0, 2, 1)
    return torch.FloatTensor(segments).unsqueeze(1)


def _from_model_array(data):
    if data.ndim == 3 and data.shape[1] > 1:
        return np.transpose(data, (0, 2, 1))
    return data.squeeze(1)


def prepare_for_comparison(segments, model, device, ts_min=None, ts_max=None):
    original_samples = []
    reconstructed_samples = []

    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(segments))):
            orig_segment = segments[i].copy()

            sample_tensor = _to_model_tensor(orig_segment[np.newaxis, ...]).to(device)
            mu, logvar = model.encoder(sample_tensor)
            z = model.reparameterize(mu, logvar)
            recon = model.generator(z)
            recon_np = _from_model_array(recon.cpu().numpy())[0]

            if ts_min is not None and ts_max is not None:
                orig_segment = denormalize_unit_interval_array(orig_segment, ts_min, ts_max)
                recon_np = denormalize_unit_interval_array(recon_np, ts_min, ts_max)

            original_samples.append(orig_segment)
            reconstructed_samples.append(recon_np)

    return original_samples, reconstructed_samples


def _prepare_data(bundle: AugDataBundle, cfg: dict[str, Any], label_mask: np.ndarray | None = None):
    model_cfg = cfg["model"]
    seq_len = model_cfg.get("seq_len", 128)
    overlap_ratio = model_cfg.get("overlap_ratio", 0.5)
    batch_size = model_cfg.get("batch_size", 32)

    ts_data = bundle.raw_data.astype(np.float32)
    norm_source = ts_data[label_mask] if label_mask is not None else ts_data
    if norm_source.size == 0:
        norm_source = ts_data

    if ts_data.ndim == 3:
        ts_min, ts_max = compute_norm_stats(norm_source)
        ts_normalized = normalize_unit_interval(ts_data, ts_min, ts_max)
        segments = ts_normalized.astype(np.float32)
    else:
        ts_min, ts_max = compute_norm_stats(norm_source)
        if np.squeeze(ts_max - ts_min) > 0:
            ts_normalized = normalize_unit_interval(ts_data, ts_min, ts_max)
        else:
            ts_normalized = ts_data - ts_min
        segments = split_data(ts_normalized, seq_len, overlap_ratio).astype(np.float32)

    segments_tensor = _to_model_tensor(segments)
    dataset = TensorDataset(segments_tensor, torch.zeros(len(segments_tensor)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    return ts_data, ts_min, ts_max, ts_normalized, segments, segments_tensor, dataloader


def _build_feature_weight_tensor(
    feature_columns: list[str],
    weight_cfg: dict[str, float] | None,
    device: torch.device,
):
    if not weight_cfg:
        return None
    weights = [float(weight_cfg.get(name, 1.0)) for name in feature_columns]
    return torch.tensor(weights, dtype=torch.float32, device=device).view(1, -1, 1)


def _weighted_recon_loss(fake_data, real_data, feature_weights):
    if feature_weights is None:
        return F.mse_loss(fake_data, real_data)
    return torch.mean((fake_data - real_data) ** 2 * feature_weights)


def _feature_mean_match_loss(
    fake_data,
    real_data,
    feature_columns: list[str],
    mean_match_cfg: dict[str, float] | None,
):
    if not mean_match_cfg:
        return fake_data.new_tensor(0.0)
    losses = []
    for idx, name in enumerate(feature_columns):
        weight = float(mean_match_cfg.get(name, 0.0))
        if weight <= 0:
            continue
        fake_mean = fake_data[:, idx, :].mean(dim=1)
        real_mean = real_data[:, idx, :].mean(dim=1)
        losses.append(weight * F.mse_loss(fake_mean, real_mean))
    if not losses:
        return fake_data.new_tensor(0.0)
    return torch.stack(losses).sum()


def _train_single_gan_vae(
    dataloader,
    latent_dim,
    seq_len,
    input_channels,
    epochs,
    lr,
    alpha,
    device,
    feature_columns=None,
    feature_recon_weights=None,
    feature_mean_match_weights=None,
):
    model = VAE_GAN(latent_dim=latent_dim, seq_len=seq_len, input_channels=input_channels).to(device)
    optimizer_EG = optim.Adam(
        list(model.encoder.parameters()) + list(model.generator.parameters()), lr=lr
    )
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr)
    feature_columns = feature_columns or []
    recon_weights = _build_feature_weight_tensor(feature_columns, feature_recon_weights, device)

    history = {
        "d_loss": [],
        "g_loss": [],
        "recon_loss": [],
        "mean_match_loss": [],
        "random_mean_match_loss": [],
        "kl_loss": [],
        "total_loss": [],
    }
    best_recon = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        epoch_d_loss = epoch_g_loss = epoch_recon_loss = epoch_mean_match_loss = epoch_random_mean_match_loss = epoch_kl_loss = 0.0
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

            recon_loss = _weighted_recon_loss(fake_data, real_data, recon_weights)
            mean_match_loss = _feature_mean_match_loss(
                fake_data,
                real_data,
                feature_columns,
                feature_mean_match_weights,
            )
            random_z = torch.randn(batch_size_actual, model.latent_dim, device=device)
            random_gen = model.generator(random_z)
            random_mean_match_loss = _feature_mean_match_loss(
                random_gen,
                real_data,
                feature_columns,
                feature_mean_match_weights,
            )
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / batch_size_actual

            g_adv_loss = F.binary_cross_entropy(d_fake, real_labels)
            g_loss = recon_loss + mean_match_loss + random_mean_match_loss + g_adv_loss + alpha * kl_loss

            g_loss.backward()
            optimizer_EG.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_mean_match_loss += mean_match_loss.item()
            epoch_random_mean_match_loss += random_mean_match_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1

        avg_d = epoch_d_loss / max(num_batches, 1)
        avg_g = epoch_g_loss / max(num_batches, 1)
        avg_recon = epoch_recon_loss / max(num_batches, 1)
        avg_mean_match = epoch_mean_match_loss / max(num_batches, 1)
        avg_random_mean_match = epoch_random_mean_match_loss / max(num_batches, 1)
        avg_kl = epoch_kl_loss / max(num_batches, 1)

        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        history["recon_loss"].append(avg_recon)
        history["mean_match_loss"].append(avg_mean_match)
        history["random_mean_match_loss"].append(avg_random_mean_match)
        history["kl_loss"].append(avg_kl)
        history["total_loss"].append(avg_g)

        train_metric = avg_recon + avg_mean_match + avg_random_mean_match
        if train_metric < best_recon:
            best_recon = train_metric
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{epochs}] | "
                f"D_loss: {avg_d:.6f} | G_loss: {avg_g:.6f} | "
                f"Recon: {avg_recon:.6f} | MeanMatch: {avg_mean_match:.6f} | "
                f"RandMeanMatch: {avg_random_mean_match:.6f} | KL: {avg_kl:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, best_recon


def train(bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path):
    device = get_device(cfg)
    model_cfg = cfg["model"]

    ts_data, ts_min, ts_max, ts_normalized, segments, segments_tensor, dataloader = _prepare_data(bundle, cfg)

    latent_dim = model_cfg.get("latent_dim", 64)
    seq_len = model_cfg.get("seq_len", 128)
    input_channels = int(segments_tensor.shape[1])
    epochs = model_cfg.get("epochs", 50)
    lr = model_cfg.get("lr", 1e-4)
    alpha = model_cfg.get("alpha", 0.01)
    feature_recon_weights = model_cfg.get("feature_recon_weights")
    feature_mean_match_weights = model_cfg.get("feature_mean_match_weights")
    train_kwargs = {
        "feature_columns": bundle.feature_columns,
        "feature_recon_weights": feature_recon_weights,
        "feature_mean_match_weights": feature_mean_match_weights,
    }

    if model_cfg.get("per_class", False) and bundle.labels is not None:
        models: dict[str, VAE_GAN] = {}
        checkpoint_by_class: dict[str, dict[str, Any]] = {}
        history_by_class: dict[str, Any] = {}
        best_by_class: dict[str, Any] = {}
        per_class_norm: dict[str, dict[str, Any]] = {}

        for label_id, label_name in enumerate(bundle.label_names):
            class_mask = bundle.labels == label_id
            class_raw = ts_data[class_mask]
            if len(class_raw) == 0:
                continue
            class_min, class_max = compute_norm_stats(class_raw)
            class_segments = normalize_unit_interval(class_raw, class_min, class_max).astype(np.float32)
            per_class_norm[label_name] = {
                "ts_min": class_min.tolist() if hasattr(class_min, "tolist") else float(class_min),
                "ts_max": class_max.tolist() if hasattr(class_max, "tolist") else float(class_max),
            }
            if len(class_segments) < model_cfg.get("batch_size", 32):
                continue

            print(f"\n--- VAE-GAN class: {label_name} ({len(class_segments)} windows) ---")
            class_tensor = _to_model_tensor(class_segments)
            class_loader = DataLoader(
                TensorDataset(class_tensor, torch.zeros(len(class_tensor))),
                batch_size=model_cfg.get("batch_size", 32),
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            class_model, class_history, best_epoch, best_recon = _train_single_gan_vae(
                class_loader,
                latent_dim=latent_dim,
                seq_len=seq_len,
                input_channels=input_channels,
                epochs=epochs,
                lr=lr,
                alpha=alpha,
                device=device,
                **train_kwargs,
            )
            models[label_name] = class_model
            checkpoint_by_class[label_name] = class_model.state_dict()
            history_by_class[label_name] = {
                **class_history,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "best_recon_loss": best_recon,
            }
            best_by_class[label_name] = {
                "best_epoch": best_epoch,
                "best_recon_loss": best_recon,
                "num_windows": int(len(class_segments)),
            }
            _plot_gan_vae_curves(class_history, out_dir / f"{label_name}_loss_curves.png")

        save_checkpoint_best(
            out_dir,
            {
                "per_class": checkpoint_by_class,
                "latent_dim": latent_dim,
                "seq_len": seq_len,
                "input_channels": input_channels,
                "feature_columns": bundle.feature_columns,
                "label_names": bundle.label_names,
            },
            epoch=max((item["best_epoch"] for item in best_by_class.values()), default=0),
            best_metric=float(np.mean([item["best_recon_loss"] for item in best_by_class.values()])),
            metric_name="mean_train_recon_loss",
        )
        save_loss_history({"per_class": history_by_class}, out_dir)
        save_json(
            {
                "ts_min": ts_min,
                "ts_max": ts_max,
                "per_class_norm": per_class_norm,
                "normalization": "per_class_minmax",
                "seq_len": seq_len,
                "latent_dim": latent_dim,
                "input_channels": input_channels,
                "feature_columns": bundle.feature_columns,
                "label_names": bundle.label_names,
                "per_class": best_by_class,
            },
            out_dir / "norm_params.json",
        )
        meta = {
            "ts_min": ts_min,
            "ts_max": ts_max,
            "per_class_norm": per_class_norm,
            "segments": segments,
            "labels": bundle.labels,
            "best_epoch": None,
            "best_recon_loss": float(np.mean([item["best_recon_loss"] for item in best_by_class.values()])),
            "input_channels": input_channels,
            "feature_columns": bundle.feature_columns,
            "label_names": bundle.label_names,
            "per_class": best_by_class,
        }
        return models, meta

    model = VAE_GAN(latent_dim=latent_dim, seq_len=seq_len, input_channels=input_channels).to(device)
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
                {
                    "model_state_dict": model.state_dict(),
                    "input_channels": input_channels,
                    "feature_columns": bundle.feature_columns,
                    "label_names": bundle.label_names,
                },
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
        "ts_min": ts_min,
        "ts_max": ts_max,
        "segments": segments,
        "segments_tensor": segments_tensor,
        "best_epoch": best_epoch,
        "best_recon_loss": best_recon,
        "input_channels": input_channels,
        "feature_columns": bundle.feature_columns,
        "label_names": bundle.label_names,
    }
    save_json(
        {
            "ts_min": ts_min,
            "ts_max": ts_max,
            "seq_len": seq_len,
            "latent_dim": latent_dim,
            "input_channels": input_channels,
            "feature_columns": bundle.feature_columns,
            "label_names": bundle.label_names,
        },
        out_dir / "norm_params.json",
    )
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
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "per_class" in checkpoint:
        models = {}
        latent_dim = int(checkpoint.get("latent_dim", model_cfg.get("latent_dim", 64)))
        seq_len = int(checkpoint.get("seq_len", model_cfg.get("seq_len", 128)))
        input_channels = int(checkpoint.get("input_channels", model_cfg.get("input_channels", 1)))
        for label_name, state in checkpoint["per_class"].items():
            model = VAE_GAN(
                latent_dim=latent_dim,
                seq_len=seq_len,
                input_channels=input_channels,
            ).to(device)
            model.load_state_dict(state)
            model.eval()
            models[label_name] = model
        return models
    model = VAE_GAN(
        latent_dim=model_cfg.get("latent_dim", 64),
        seq_len=model_cfg.get("seq_len", 128),
        input_channels=int(checkpoint.get("input_channels", model_cfg.get("input_channels", 1))),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate(model: VAE_GAN, bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path) -> np.ndarray:
    device = get_device(cfg)
    model_cfg = cfg["model"]

    _, ts_min, ts_max, _, segments, _, _ = _prepare_data(bundle, cfg)
    norm_params = load_norm_params(out_dir)
    calibrate_features = model_cfg.get("feature_mean_calibration", [])
    if isinstance(model, dict):
        per_class = int(model_cfg.get("num_generate_per_class", model_cfg.get("num_generate", 5)))
        generated_by_class = []
        generated_labels = []
        with torch.no_grad():
            for label_id, label_name in enumerate(bundle.label_names):
                class_model = model.get(label_name)
                if class_model is None:
                    continue
                class_model = class_model.to(device)
                class_mask = bundle.labels == label_id
                class_raw = bundle.raw_data[class_mask]
                class_ts_min, class_ts_max = resolve_class_norm(
                    label_name, norm_params, fallback_min=ts_min, fallback_max=ts_max
                )
                random_generated_samples = []
                for _ in range(per_class):
                    random_z = torch.randn(1, class_model.latent_dim).to(device)
                    random_gen = class_model.generator(random_z)
                    random_gen_np = _from_model_array(random_gen.cpu().numpy())[0]
                    random_gen_np = denormalize_unit_interval_array(random_gen_np, class_ts_min, class_ts_max)
                    random_generated_samples.append(random_gen_np)
                generated_class = np.stack(random_generated_samples, axis=0)
                generated_class = calibrate_feature_means(
                    generated_class,
                    class_raw,
                    bundle.feature_columns,
                    calibrate_features,
                )
                np.save(out_dir / f"generated_{label_name}.npy", generated_class)
                generated_by_class.append(generated_class)
                generated_labels.extend([label_id] * len(generated_class))

        generated = np.concatenate(generated_by_class, axis=0)
        np.save(out_dir / "generated_samples.npy", generated)
        np.save(out_dir / "generated_labels.npy", np.asarray(generated_labels, dtype=np.int64))
        return generated

    model = model.to(device)
    num_generate = model_cfg.get("num_generate", 5)
    random_generated_samples = []
    with torch.no_grad():
        for _ in range(num_generate):
            random_z = torch.randn(1, model.latent_dim).to(device)
            random_gen = model.generator(random_z)
            random_gen_np = _from_model_array(random_gen.cpu().numpy())[0]
            random_gen_np = denormalize_unit_interval_array(random_gen_np, ts_min, ts_max)
            random_generated_samples.append(random_gen_np)

    generated = np.stack(random_generated_samples, axis=0)
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
    norm_params = load_norm_params(out_dir)
    per_class_metrics = {}
    if isinstance(model, dict) and bundle.labels is not None:
        generated_labels_path = out_dir / "generated_labels.npy"
        generated_labels = np.load(generated_labels_path) if generated_labels_path.exists() else None
        for label_id, label_name in enumerate(bundle.label_names):
            class_model = model.get(label_name)
            if class_model is None:
                continue
            class_model = class_model.to(device)
            class_mask = bundle.labels == label_id
            class_raw = bundle.raw_data[class_mask]
            if len(class_raw) == 0:
                continue
            class_ts_min, class_ts_max = resolve_class_norm(
                label_name, norm_params, fallback_min=ts_min, fallback_max=ts_max
            )
            class_segments = normalize_unit_interval(class_raw, class_ts_min, class_ts_max).astype(np.float32)
            original_denorm_class = class_raw
            item = {
                "num_original_windows": int(len(class_raw)),
                "generated_file": f"generated_{label_name}.npy",
                "signal_comparison": f"{label_name}_signal_comparison.png",
                "frequency_comparison": f"{label_name}_frequency_comparison.png",
            }
            if generated_labels is not None and generated_samples is not None and len(generated_samples):
                generated_class = generated_samples[generated_labels == label_id]
                if len(generated_class):
                    n_compare = min(5, len(original_denorm_class), len(generated_class))
                    class_stats = compare_signals(
                        original_denorm_class[:n_compare],
                        generated_class[:n_compare],
                        save_dir=str(out_dir),
                        sample_rate=sample_rate,
                        filename_prefix=f"{label_name}_",
                        feature_names=bundle.feature_columns,
                    )
                    item["statistics"] = class_stats
                    item["num_generated_windows"] = int(len(generated_class))
                    item["random_generated"] = {
                        "mean": float(np.mean(generated_class)),
                        "std": float(np.std(generated_class)),
                        "min": float(np.min(generated_class)),
                        "max": float(np.max(generated_class)),
                    }
                    item["generated_vs_original_mean_bias"] = mean_bias_metrics(
                        original_denorm_class,
                        generated_class,
                        feature_names=bundle.feature_columns,
                    )
            original_class, recon_class = prepare_for_comparison(
                class_segments, class_model, device, class_ts_min, class_ts_max
            )
            item["reconstruction_mean_bias"] = mean_bias_metrics(
                np.stack(original_class, axis=0),
                np.stack(recon_class, axis=0),
                feature_names=bundle.feature_columns,
            )
            per_class_metrics[label_name] = item

        first_label = next(iter(per_class_metrics), None)
        spectrum_stats = per_class_metrics[first_label].get("statistics", {}) if first_label else {}
        original_samples = []
        reconstructed_samples = []
    else:
        model = model.to(device)

        original_samples, reconstructed_samples = prepare_for_comparison(
            segments, model, device, ts_min, ts_max
        )
        spectrum_stats = compare_signals(
            original_samples,
            reconstructed_samples,
            save_dir=str(out_dir),
            sample_rate=sample_rate,
            feature_names=bundle.feature_columns,
        )

    if generated_samples is None:
        gen_path = out_dir / "generated_samples.npy"
        generated_samples = np.load(gen_path) if gen_path.exists() else np.array([])

    all_random = generated_samples if len(generated_samples) else np.array(reconstructed_samples)
    all_original = np.asarray(original_samples)
    if all_original.size == 0 and bundle.labels is not None:
        original_parts = []
        for label_id, label_name in enumerate(bundle.label_names):
            class_mask = bundle.labels == label_id
            class_raw = bundle.raw_data[class_mask]
            if len(class_raw) == 0:
                continue
            original_parts.append(class_raw)
        if original_parts:
            all_original = np.concatenate(original_parts, axis=0)
    elif all_original.size == 0:
        all_original = denormalize_unit_interval_array(segments, ts_min, ts_max)

    metrics = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "dataset": cfg["dataset"]["name"],
        "model": "gan_vae",
        "feature_columns": bundle.feature_columns,
        "label_names": bundle.label_names,
        "input_shape": list(segments.shape[1:]),
        "per_class": per_class_metrics,
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
            "generated_vs_original_mean_bias": mean_bias_metrics(
                all_original,
                all_random,
                feature_names=bundle.feature_columns,
            ),
        },
    }

    if meta:
        metrics["training_best"] = {
            "best_epoch": meta.get("best_epoch"),
            "best_recon_loss": meta.get("best_recon_loss"),
        }

    save_json(metrics, out_dir / "metrics.json")
    if len(original_samples):
        np.save(out_dir / "original_samples.npy", original_samples)
    if len(reconstructed_samples):
        np.save(out_dir / "reconstructed_samples.npy", reconstructed_samples)
    return metrics
