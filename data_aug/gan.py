from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_aug.common import (
    compare_signals,
    mean_bias_metrics,
    normalize_to_minus1_1,
    plot_loss_curves_gan,
    split_data,
)
from data_aug.data_load import AugDataBundle
from data_aug.io_utils import get_device, save_checkpoint_best, save_json, save_loss_history
from data_aug.shared import (
    compute_sample_diversity,
    denormalize_by_label,
    denormalize_minus1_1_array,
    load_norm_params,
    normalize_class_windows,
    normalize_windows_minus1_1,
    physical_statistics,
    resolve_class_norm,
    strip_plot_compare_stats,
)


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


def train_gan(
    data,
    labels=None,
    z_dim=100,
    epochs=100,
    batch_size=32,
    g_lr=0.002,
    d_lr=0.0001,
    lr=0.0001,
    recon_weight=0.1,
    device=None,
):
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

    device = device or torch.device("cpu")
    data = data.to(device)
    n_samples, _, H, W = data.shape

    if labels is None:
        label_dim = z_dim
        labels = torch.randn(n_samples, label_dim, device=device)
    else:
        labels = labels.to(device)
        label_dim = labels.shape[1]

    enc = Encoder(input_dim=H * W, z_dim=z_dim, label_dim=label_dim).to(device)
    gen = Generator(z_dim=z_dim, output_shape=(H, W)).to(device)
    disc = Discriminator(input_shape=(H, W)).to(device)

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

            noise = torch.randn(batch_size_actual, z_dim, device=device)
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

            total_loss = g_loss + recon_weight * recon_loss
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
    window_size = cfg["model"].get("window_size", 50)
    overlap_ratio = cfg["model"].get("overlap_ratio", 0.2)
    raw_data = bundle.raw_data.astype(np.float32)

    if raw_data.ndim == 3:
        segments, data_min, data_max = normalize_windows_minus1_1(raw_data)
    elif raw_data.ndim == 2:
        data_min = raw_data.min(axis=0, keepdims=True)
        data_max = raw_data.max(axis=0, keepdims=True)
        denom = np.where((data_max - data_min) < 1e-8, 1.0, data_max - data_min)
        data_norm = 2 * ((raw_data - data_min) / denom) - 1
        stride = int(cfg["model"].get("step_length", max(1, int(window_size * (1 - overlap_ratio)))))
        segments = np.asarray(
            [data_norm[i : i + window_size] for i in range(0, len(data_norm) - window_size + 1, stride)],
            dtype=np.float32,
        )
    else:
        raw_data = raw_data.reshape(-1, 1)
        column_data, data_min, data_max = normalize_to_minus1_1(raw_data)
        segments = split_data(column_data, window_size=window_size, overlap_ratio=overlap_ratio)

    segments = segments.astype(np.float32)
    data = torch.tensor(segments, dtype=torch.float32)
    return raw_data, data_min, data_max, segments, data


def train(bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path):
    model_cfg = cfg["model"]
    _, data_min, data_max, segments, data = _prepare_segments(bundle, cfg)
    device = get_device(cfg)
    z_dim = model_cfg.get("z_dim", 100)
    H, W = data.shape[1], data.shape[2]

    use_per_class_norm = bool(model_cfg.get("per_class_norm", False))
    recon_weight = float(model_cfg.get("recon_weight", 0.1))
    per_class_norm: dict[str, dict[str, Any]] = {}

    if model_cfg.get("per_class", False) and bundle.labels is not None:
        models: dict[str, tuple[Encoder, Generator, Discriminator]] = {}
        checkpoint_by_class: dict[str, dict[str, Any]] = {}
        history_by_class: dict[str, Any] = {}
        best_by_class: dict[str, Any] = {}

        for label_id, label_name in enumerate(bundle.label_names):
            class_mask = bundle.labels == label_id
            if use_per_class_norm:
                class_raw = bundle.raw_data[class_mask]
                if len(class_raw) == 0:
                    continue
                class_segments, class_min, class_max = normalize_class_windows(class_raw)
                per_class_norm[label_name] = {
                    "data_min": np.asarray(class_min).tolist(),
                    "data_max": np.asarray(class_max).tolist(),
                }
            else:
                class_segments = segments[class_mask]
            if len(class_segments) == 0:
                continue

            print(f"\n--- GAN class: {label_name} ({len(class_segments)} windows) ---")
            class_data = torch.tensor(class_segments, dtype=torch.float32)
            class_labels = F.one_hot(
                torch.full((len(class_segments),), label_id, dtype=torch.long),
                num_classes=len(bundle.label_names),
            ).float()

            enc, gen, disc, history, best_state, best_epoch, best_recon = train_gan(
                data=class_data,
                labels=class_labels,
                z_dim=z_dim,
                epochs=model_cfg.get("epochs", 50),
                batch_size=model_cfg.get("batch_size", 64),
                g_lr=model_cfg.get("g_lr", 0.002),
                d_lr=model_cfg.get("d_lr", 0.0001),
                lr=model_cfg.get("enc_lr", 0.001),
                recon_weight=recon_weight,
                device=device,
            )
            models[label_name] = (enc, gen, disc)
            checkpoint_by_class[label_name] = best_state or {
                "encoder_state_dict": enc.state_dict(),
                "generator_state_dict": gen.state_dict(),
                "discriminator_state_dict": disc.state_dict(),
            }
            history_by_class[label_name] = {
                **history,
                "epochs": model_cfg.get("epochs", 50),
                "best_epoch": best_epoch,
                "best_recon_loss": best_recon,
            }
            best_by_class[label_name] = {
                "best_epoch": best_epoch,
                "best_recon_loss": best_recon,
                "num_windows": int(len(class_segments)),
            }
            plot_loss_curves_gan(history, str(out_dir / f"{label_name}_loss_curves.png"))

        save_checkpoint_best(
            out_dir,
            {
                "per_class": checkpoint_by_class,
                "z_dim": z_dim,
                "H": int(H),
                "W": int(W),
                "label_dim": len(bundle.label_names),
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
                "z_dim": z_dim,
                "H": int(H),
                "W": int(W),
                "label_dim": len(bundle.label_names),
                "data_min": data_min,
                "data_max": data_max,
                "feature_columns": bundle.feature_columns,
                "label_names": bundle.label_names,
                "per_class": best_by_class,
                "per_class_norm": per_class_norm,
            },
            out_dir / "norm_params.json",
        )
        meta = {
            "segments": segments,
            "labels": bundle.labels,
            "data_min": data_min,
            "data_max": data_max,
            "z_dim": z_dim,
            "H": int(H),
            "W": int(W),
            "label_dim": len(bundle.label_names),
            "feature_columns": bundle.feature_columns,
            "label_names": bundle.label_names,
            "per_class": best_by_class,
        }
        return models, None, None, meta

    label_tensor = None
    label_dim = model_cfg.get("label_dim")
    if bundle.labels is not None and len(bundle.labels) == len(segments):
        num_classes = len(bundle.label_names) if bundle.label_names else int(np.max(bundle.labels)) + 1
        label_tensor = F.one_hot(torch.tensor(bundle.labels, dtype=torch.long), num_classes=num_classes).float()
        label_dim = num_classes

    enc, gen, disc, history, best_state, best_epoch, best_recon = train_gan(
        data=data,
        labels=label_tensor,
        z_dim=z_dim,
        epochs=model_cfg.get("epochs", 50),
        batch_size=model_cfg.get("batch_size", 64),
        g_lr=model_cfg.get("g_lr", 0.002),
        d_lr=model_cfg.get("d_lr", 0.0001),
        lr=model_cfg.get("enc_lr", 0.001),
        recon_weight=recon_weight,
        device=device,
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

    meta = {
        "segments": segments,
        "data_min": data_min,
        "data_max": data_max,
        "z_dim": z_dim,
        "H": int(H),
        "W": int(W),
        "label_dim": int(label_dim or z_dim),
        "feature_columns": bundle.feature_columns,
        "label_names": bundle.label_names,
        "best_epoch": best_epoch,
        "best_recon_loss": best_recon,
    }
    save_json(
        {
            "z_dim": z_dim,
            "H": int(H),
            "W": int(W),
            "label_dim": int(label_dim or z_dim),
            "data_min": data_min,
            "data_max": data_max,
            "feature_columns": bundle.feature_columns,
            "label_names": bundle.label_names,
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
            label_dim = norm.get("label_dim", model_cfg.get("label_dim", z_dim))
        else:
            raise FileNotFoundError(f"GAN norm params not found: {norm_path}")
    else:
        z_dim = model_cfg.get("z_dim", 100)
        label_dim = model_cfg.get("label_dim", z_dim)
        window_size = model_cfg.get("window_size", 50)
        H, W = window_size, 1

    enc = Encoder(input_dim=H * W, z_dim=z_dim, label_dim=label_dim).to(device)
    gen = Generator(z_dim=z_dim, output_shape=(H, W)).to(device)
    disc = Discriminator(input_shape=(H, W)).to(device)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "per_class" in checkpoint:
        models = {}
        label_names = checkpoint.get("label_names", [])
        for label_name in label_names:
            class_state = checkpoint["per_class"].get(label_name)
            if class_state is None:
                continue
            enc = Encoder(input_dim=H * W, z_dim=z_dim, label_dim=label_dim).to(device)
            gen = Generator(z_dim=z_dim, output_shape=(H, W)).to(device)
            disc = Discriminator(input_shape=(H, W)).to(device)
            enc.load_state_dict(class_state["encoder_state_dict"])
            gen.load_state_dict(class_state["generator_state_dict"])
            disc.load_state_dict(class_state["discriminator_state_dict"])
            enc.eval()
            gen.eval()
            disc.eval()
            models[label_name] = (enc, gen, disc)
        return models, None, None

    enc.load_state_dict(checkpoint["encoder_state_dict"])
    gen.load_state_dict(checkpoint["generator_state_dict"])
    disc.load_state_dict(checkpoint["discriminator_state_dict"])
    enc.eval()
    gen.eval()
    disc.eval()
    return enc, gen, disc


def generate(enc, gen, disc, bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path) -> np.ndarray:
    model_cfg = cfg["model"]
    device = get_device(cfg)
    _, _, _, segments, _ = _prepare_segments(bundle, cfg)

    z_dim = model_cfg.get("z_dim", 100)
    if isinstance(enc, dict):
        per_class = int(model_cfg.get("num_generate_per_class", model_cfg.get("num_generate", 20)))
        generated_by_class = []
        generated_labels = []
        for label_id, label_name in enumerate(bundle.label_names):
            if label_name not in enc:
                continue
            _, class_gen, _ = enc[label_name]
            class_gen = class_gen.to(device)
            n_generate = per_class
            with torch.no_grad():
                noise = torch.randn(n_generate, z_dim, device=device)
                generated_class = class_gen(noise).cpu().numpy()
            np.save(out_dir / f"generated_{label_name}.npy", generated_class)
            generated_by_class.append(generated_class)
            generated_labels.extend([label_id] * len(generated_class))

        generated_segments = np.concatenate(generated_by_class, axis=0)
        np.save(out_dir / "generated_samples.npy", generated_segments)
        np.save(out_dir / "generated_labels.npy", np.asarray(generated_labels, dtype=np.int64))
        return generated_segments

    gen = gen.to(device)
    n_generate = model_cfg.get("num_generate", 20)
    with torch.no_grad():
        noise = torch.randn(n_generate, z_dim, device=device)
        generated_segments = gen(noise).cpu().numpy()

    np.save(out_dir / "generated_samples.npy", generated_segments)
    return generated_segments


def evaluate(
    bundle: AugDataBundle,
    out_dir: Path,
    cfg: dict[str, Any],
    generated_samples: np.ndarray | None = None,
    segments: np.ndarray | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import mean_squared_error

    model_cfg = cfg["model"]
    sample_rate = float(cfg["dataset"].get("sample_rate", bundle.meta.get("sample_rate", 12000)))

    if segments is None:
        _, data_min, data_max, segments, _ = _prepare_segments(bundle, cfg)
    else:
        _, data_min, data_max, _, _ = _prepare_segments(bundle, cfg)

    norm_params = load_norm_params(out_dir)

    if generated_samples is None:
        generated_samples = np.load(out_dir / "generated_samples.npy")

    spectrum_plot_style = model_cfg.get("spectrum_plot_style", "line")
    split_feature_plots = bool(model_cfg.get("split_feature_plots", True))
    num_compare = int(model_cfg.get("num_compare", 5))
    feature_names = bundle.feature_columns
    per_class_metrics = {}
    generated_labels_path = out_dir / "generated_labels.npy"
    generated_labels = None
    if bundle.labels is not None and generated_labels_path.exists():
        generated_labels = np.load(generated_labels_path)
        for label_id, label_name in enumerate(bundle.label_names):
            class_mask = bundle.labels == label_id
            original_class = segments[class_mask]
            original_phys_class = bundle.raw_data[class_mask]
            generated_class = generated_samples[generated_labels == label_id]
            if len(original_class) == 0 or len(generated_class) == 0:
                continue

            n_class_compare = min(num_compare, len(original_class), len(generated_class))
            original_phys_compare = original_phys_class[:n_class_compare]
            class_dmin, class_dmax = resolve_class_norm(
                label_name, norm_params, data_min, data_max
            )
            generated_phys_compare = denormalize_minus1_1_array(
                generated_class[:n_class_compare], class_dmin, class_dmax
            )
            plot_stats = compare_signals(
                original_phys_compare,
                generated_phys_compare,
                save_dir=str(out_dir),
                sample_rate=sample_rate,
                filename_prefix=f"{label_name}_",
                spectrum_plot_style=spectrum_plot_style,
                feature_names=feature_names,
                split_feature_plots=split_feature_plots,
                use_physical_plot_scale=True,
                num_compare=n_class_compare,
            )
            class_stats = strip_plot_compare_stats(plot_stats)
            class_stats.update(
                physical_statistics(
                    original_phys_compare, generated_phys_compare, feature_names
                )
            )

            class_mse = [
                mean_squared_error(
                    original_phys_compare[i].squeeze(),
                    generated_phys_compare[i].squeeze(),
                )
                for i in range(n_class_compare)
            ]
            per_class_metrics[label_name] = {
                "statistics": class_stats,
                "mse": {"avg_mse": float(np.mean(class_mse))},
                "diversity": compute_sample_diversity(generated_class),
                "num_original_windows": int(len(original_class)),
                "num_generated_windows": int(len(generated_class)),
                "generated_file": f"generated_{label_name}.npy",
                "feature_plots": plot_stats.get("feature_plots", {}),
                "generated_vs_original_mean_bias": mean_bias_metrics(
                    original_phys_class,
                    denormalize_minus1_1_array(
                        generated_class,
                        *resolve_class_norm(label_name, norm_params, data_min, data_max),
                    ),
                    feature_names=feature_names,
                ),
            }

    n_compare = min(num_compare, len(segments), len(generated_samples))
    original_phys_compare = bundle.raw_data[:n_compare]
    if generated_labels_path.exists() and bundle.labels is not None:
        generated_labels = np.load(generated_labels_path)
        generated_phys_compare = denormalize_by_label(
            generated_samples[:n_compare],
            generated_labels[:n_compare],
            bundle.label_names,
            norm_params,
            data_min,
            data_max,
        )
    else:
        generated_phys_compare = denormalize_minus1_1_array(
            generated_samples[:n_compare], data_min, data_max
        )

    plot_stats = compare_signals(
        original_phys_compare,
        generated_phys_compare,
        save_dir=str(out_dir),
        sample_rate=sample_rate,
        spectrum_plot_style=spectrum_plot_style,
        feature_names=feature_names,
        split_feature_plots=split_feature_plots,
        use_physical_plot_scale=True,
        num_compare=n_compare,
    )
    spectrum_stats = strip_plot_compare_stats(plot_stats)
    spectrum_stats.update(
        physical_statistics(
            original_phys_compare, generated_phys_compare, feature_names
        )
    )

    mse_list = []
    mse_per_list = []
    for i in range(n_compare):
        mse = mean_squared_error(
            original_phys_compare[i].squeeze(),
            generated_phys_compare[i].squeeze(),
        )
        mse_list.append(mse)
        denom = np.max(original_phys_compare[i]) - np.min(original_phys_compare[i])
        mse_per = mse / denom if denom != 0 else 0.0
        mse_per_list.append(mse_per)

    metrics = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "dataset": cfg["dataset"]["name"],
        "model": "gan",
        "feature_columns": bundle.feature_columns,
        "label_names": bundle.label_names,
        "input_shape": list(segments.shape[1:]),
        "per_class": per_class_metrics,
        "statistics": spectrum_stats,
        "mse": {
            "avg_mse": float(np.mean(mse_list)),
            "avg_mse_percent": float(np.mean(mse_per_list)),
        },
        "diversity": compute_sample_diversity(generated_samples),
        "feature_plots": plot_stats.get("feature_plots", {}),
        "generated_vs_original_mean_bias": mean_bias_metrics(
            bundle.raw_data,
            denormalize_by_label(
                generated_samples,
                np.load(generated_labels_path)
                if generated_labels_path.exists() and bundle.labels is not None
                else None,
                bundle.label_names,
                norm_params,
                data_min,
                data_max,
            ),
            feature_names=feature_names,
        ),
        "metric_scale": "physical",
    }
    save_json(metrics, out_dir / "metrics.json")
    return metrics
