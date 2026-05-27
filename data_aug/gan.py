from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_aug.common import compare_signals, normalize_to_minus1_1, plot_loss_curves_gan, split_data
from data_aug.data_load import AugDataBundle
from data_aug.io_utils import get_device, save_checkpoint_best, save_json, save_loss_history


class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, label_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=-1)
        z = self.fc(inp)
        return z


class Generator(nn.Module):
    def __init__(self, z_dim, output_shape):
        super().__init__()
        self.output_shape = output_shape
        H, W = output_shape

        self.fc = nn.Linear(z_dim, 256)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((H, W)),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.lstm = nn.LSTM(input_size=W, hidden_size=64, batch_first=True)
        self.final_fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, H * W),
            nn.Tanh(),
        )

    def forward(self, z):
        batch_size = z.shape[0]
        H, W = self.output_shape

        x = self.fc(z)
        x = x.view(batch_size, 256, 1, 1)
        x = self.deconv_layers(x)
        x_flat = x.squeeze(1)
        x_lstm = x_flat.view(batch_size, H, W)
        lstm_out, _ = self.lstm(x_lstm)
        lstm_out = lstm_out[:, -1, :]
        output = self.final_fc(lstm_out)
        output = output.view(batch_size, H, W)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        H, W = input_shape

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.lstm = nn.LSTM(input_size=256 * 4, hidden_size=128, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        batch_size = x.shape[0]
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(batch_size, 4, -1)
        lstm_out, _ = self.lstm(conv_out)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output


def train_gan(data, labels=None, z_dim=100, epochs=100, batch_size=32, g_lr=0.002, d_lr=0.0001, lr=0.0001):
    """
    通用GAN训练函数
    data: 形状为 (n_samples, H, W) 或 (n_samples, C, H, W)
    """
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
    elif len(data.shape) == 4 and data.shape[1] == 1:
        pass
    else:
        raise ValueError(f"数据形状 {data.shape} 不正确，应为 (n, H, W) 或 (n, 1, H, W)")

    n_samples, _, H, W = data.shape

    if labels is None:
        label_dim = z_dim
        labels = torch.randn(n_samples, label_dim)
    else:
        label_dim = labels.shape[1]

    enc = Encoder(input_dim=H * W, z_dim=z_dim, label_dim=label_dim)
    gen = Generator(z_dim=z_dim, output_shape=(H, W))
    disc = Discriminator(input_shape=(H, W))

    optim_disc = torch.optim.Adam(disc.parameters(), lr=d_lr, betas=(0.5, 0.999))
    optim_gen = torch.optim.Adam(gen.parameters(), lr=g_lr, betas=(0.5, 0.999))
    optim_enc = torch.optim.Adam(enc.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()
    mse_loss = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"d_loss": [], "g_loss": [], "recon_loss": [], "total_loss": []}
    best_recon = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        epoch_d = epoch_g = epoch_recon = 0.0
        n_batches = 0

        for batch_data, batch_labels in dataloader:
            batch_size_actual = batch_data.shape[0]
            real_data = batch_data.squeeze(1)

            disc.zero_grad()
            real_output = disc(real_data)
            real_loss = criterion(real_output, torch.ones_like(real_output))

            noise = torch.randn(batch_size_actual, z_dim)
            fake_data = gen(noise)
            fake_output = disc(fake_data.detach())
            fake_loss = criterion(fake_output, torch.zeros_like(fake_output))

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optim_disc.step()

            gen.zero_grad()
            enc.zero_grad()
            # 生成器损失
            fake_output2 = disc(fake_data)
            g_loss = criterion(fake_output2, torch.ones_like(fake_output2))

            # 编码器重构损失
            real_flat = real_data.view(batch_size_actual, -1)
            z_encoded = enc(real_flat, batch_labels)
            reconstructed = gen(z_encoded)
            recon_loss = mse_loss(reconstructed, real_data)

            total_loss = g_loss + 0.1 * recon_loss
            total_loss.backward()
            optim_gen.step()
            optim_enc.step()

            epoch_d += d_loss.item()
            epoch_g += g_loss.item()
            epoch_recon += recon_loss.item()
            n_batches += 1

        avg_d = epoch_d / max(n_batches, 1)
        avg_g = epoch_g / max(n_batches, 1)
        avg_recon = epoch_recon / max(n_batches, 1)
        history["d_loss"].append(avg_d)
        history["g_loss"].append(avg_g)
        history["recon_loss"].append(avg_recon)
        history["total_loss"].append(avg_g + 0.1 * avg_recon)

        if avg_recon < best_recon:
            best_recon = avg_recon
            best_epoch = epoch + 1
            best_state = {
                "encoder_state_dict": enc.state_dict(),
                "generator_state_dict": gen.state_dict(),
                "discriminator_state_dict": disc.state_dict(),
            }

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] D_loss: {avg_d:.4f}, "
                f"G_loss: {avg_g:.4f}, Recon: {avg_recon:.4f}"
            )

    return enc, gen, disc, history, best_state, best_epoch, best_recon


def _prepare_segments(bundle: AugDataBundle, cfg: dict[str, Any]):
    raw_data = bundle.raw_data.astype(np.float32).reshape(-1, 1)
    column_data, data_min, data_max = normalize_to_minus1_1(raw_data)

    window_size = cfg["model"].get("window_size", 50)
    overlap_ratio = cfg["model"].get("overlap_ratio", 0.2)
    segments = split_data(column_data, window_size=window_size, overlap_ratio=overlap_ratio)
    data = torch.tensor(segments, dtype=torch.float32)
    return raw_data, data_min, data_max, segments, data


def train(bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path):
    model_cfg = cfg["model"]
    _, data_min, data_max, segments, data = _prepare_segments(bundle, cfg)

    z_dim = model_cfg.get("z_dim", 100)
    enc, gen, disc, history, best_state, best_epoch, best_recon = train_gan(
        data=data,
        z_dim=z_dim,
        epochs=model_cfg.get("epochs", 50),
        batch_size=model_cfg.get("batch_size", 64),
        g_lr=model_cfg.get("g_lr", 0.002),
        d_lr=model_cfg.get("d_lr", 0.0001),
        lr=model_cfg.get("enc_lr", 0.001),
    )

    if best_state is not None:
        save_checkpoint_best(
            out_dir,
            best_state,
            epoch=best_epoch,
            best_metric=best_recon,
            metric_name="train_recon_loss",
        )

    plot_loss_curves_gan(history, str(out_dir / "loss_curves.png"))
    save_loss_history({**history, "epochs": model_cfg.get("epochs", 50), "best_epoch": best_epoch}, out_dir)

    H, W = data.shape[1], data.shape[2]
    meta = {
        "segments": segments,
        "data_min": float(data_min),
        "data_max": float(data_max),
        "z_dim": z_dim,
        "H": int(H),
        "W": int(W),
        "best_epoch": best_epoch,
        "best_recon_loss": best_recon,
    }
    save_json(
        {
            "z_dim": z_dim,
            "H": int(H),
            "W": int(W),
            "data_min": float(data_min),
            "data_max": float(data_max),
        },
        out_dir / "norm_params.json",
    )
    return enc, gen, disc, meta


def load_checkpoint(path: Path, cfg: dict[str, Any], out_dir: Path | None = None):
    device = get_device(cfg)
    model_cfg = cfg["model"]

    if out_dir is not None:
        norm_path = out_dir / "norm_params.json"
        if norm_path.exists():
            import json

            with open(norm_path, encoding="utf-8") as f:
                norm = json.load(f)
            H, W = norm["H"], norm["W"]
            z_dim = norm.get("z_dim", model_cfg.get("z_dim", 100))
        else:
            raise FileNotFoundError(f"GAN norm params not found: {norm_path}")
    else:
        z_dim = model_cfg.get("z_dim", 100)
        window_size = model_cfg.get("window_size", 50)
        H, W = window_size, 1

    enc = Encoder(input_dim=H * W, z_dim=z_dim, label_dim=z_dim)
    gen = Generator(z_dim=z_dim, output_shape=(H, W))
    disc = Discriminator(input_shape=(H, W))

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    enc.load_state_dict(checkpoint["encoder_state_dict"])
    gen.load_state_dict(checkpoint["generator_state_dict"])
    disc.load_state_dict(checkpoint["discriminator_state_dict"])
    enc.eval()
    gen.eval()
    disc.eval()
    return enc, gen, disc


def generate(enc, gen, disc, bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path) -> np.ndarray:
    model_cfg = cfg["model"]
    _, _, _, segments, _ = _prepare_segments(bundle, cfg)

    n_generate = model_cfg.get("num_generate", 20)
    z_dim = model_cfg.get("z_dim", 100)

    with torch.no_grad():
        noise = torch.randn(n_generate, z_dim)
        generated_segments = gen(noise).numpy()

    np.save(out_dir / "generated_samples.npy", generated_segments)
    return generated_segments


def evaluate(
    bundle: AugDataBundle,
    out_dir: Path,
    cfg: dict[str, Any],
    generated_samples: np.ndarray | None = None,
    segments: np.ndarray | None = None,
) -> dict[str, Any]:
    from scipy.spatial.distance import cdist
    from sklearn.metrics import mean_squared_error

    model_cfg = cfg["model"]
    sample_rate = float(cfg["dataset"].get("sample_rate", bundle.meta.get("sample_rate", 12000)))

    if segments is None:
        _, _, _, segments, _ = _prepare_segments(bundle, cfg)

    if generated_samples is None:
        generated_samples = np.load(out_dir / "generated_samples.npy")

    n_compare = min(5, len(segments), len(generated_samples))
    original_for_comparison = segments[:n_compare]
    generated_for_comparison = generated_samples[:n_compare]

    spectrum_stats = compare_signals(
        original_for_comparison,
        generated_for_comparison,
        save_dir=str(out_dir),
        sample_rate=sample_rate,
    )

    mse_list = []
    mse_per_list = []
    for i in range(n_compare):
        mse = mean_squared_error(
            original_for_comparison[i].squeeze(),
            generated_for_comparison[i].squeeze(),
        )
        mse_list.append(mse)
        denom = np.max(generated_for_comparison[i]) - np.min(generated_for_comparison[i])
        mse_per = mse / denom if denom != 0 else 0.0
        mse_per_list.append(mse_per)

    n_check = min(10, len(generated_samples))
    selected_generated = generated_samples[:n_check].reshape(n_check, -1)
    distances = cdist(selected_generated, selected_generated, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    min_distances = distances.min(axis=1)

    metrics = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "dataset": cfg["dataset"]["name"],
        "model": "gan",
        "statistics": spectrum_stats,
        "mse": {
            "avg_mse": float(np.mean(mse_list)),
            "avg_mse_percent": float(np.mean(mse_per_list)),
        },
        "diversity": {
            "min_nearest_neighbor": float(min_distances.min()),
            "mean_nearest_neighbor": float(min_distances.mean()),
            "max_nearest_neighbor": float(min_distances.max()),
            "diversity_ok": bool(min_distances.mean() >= 0.1),
        },
    }
    save_json(metrics, out_dir / "metrics.json")
    return metrics
