from __future__ import annotations

import math
import random
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

from data_aug.common import create_sequences, denormalize_from_minus1_1, normalize_to_minus1_1
from data_aug.data_load import AugDataBundle
from data_aug.io_utils import get_device, save_checkpoint_best, save_json, save_loss_history


class RVAE(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, latent_dim, num_layers=1, dec_dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dec_dropout = dec_dropout

        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.fc_mu = nn.Linear(2 * hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)

        dec_in = input_dim + latent_dim
        self.decoder_lstm = nn.LSTM(dec_in, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        nn.init.constant_(self.fc_logvar.bias, -1.0)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n.view(self.num_layers, 2, x.size(0), self.hidden_dim)[-1]
        context = torch.cat([h[0], h[1]], dim=-1)
        return self.fc_mu(context), self.fc_logvar(context)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _dec_inp(self, x_part, z):
        if self.training and self.dec_dropout > 0:
            x_part = F.dropout(x_part, self.dec_dropout)
        z1 = z.unsqueeze(1).expand(-1, 1, -1)
        return torch.cat([x_part, z1], dim=-1)

    def decode(self, z, target_seq=None, teacher_force_ratio=1.0, first_inputs=None):
        batch_size = z.size(0)
        hidden = None

        if target_seq is not None:
            outputs = []
            x_t = target_seq[:, 0:1, :]
            for t in range(self.seq_len):
                inp = self._dec_inp(x_t, z)
                out_t, hidden = self.decoder_lstm(inp, hidden)
                out_t = self.fc_out(out_t)
                outputs.append(out_t)
                if t < self.seq_len - 1:
                    b = x_t.size(0)
                    coin = torch.rand(b, 1, 1, device=x_t.device) < teacher_force_ratio
                    x_t = torch.where(coin, target_seq[:, t + 1 : t + 2, :], out_t.detach())
            return torch.cat(outputs, dim=1)

        if first_inputs is None:
            x_t = torch.zeros(batch_size, 1, self.input_dim, device=z.device, dtype=z.dtype)
        else:
            x_t = first_inputs
        outputs = []
        for _ in range(self.seq_len):
            inp = self._dec_inp(x_t, z)
            out_t, hidden = self.decoder_lstm(inp, hidden)
            out_t = self.fc_out(out_t)
            outputs.append(out_t)
            x_t = out_t
        return torch.cat(outputs, dim=1)

    def forward(self, x, teacher_force_ratio=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, target_seq=x, teacher_force_ratio=teacher_force_ratio)
        return recon_x, mu, logvar

    def generate(self, num_samples, device, first_inputs=None):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, target_seq=None, first_inputs=first_inputs)


def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
    return recon_loss + kl_loss, recon_loss, kl_loss


def reference_kld_scale(step, total_steps, mid_frac=0.45, width_frac=0.14):
    center = mid_frac * total_steps
    width = max(width_frac * total_steps, 1.0)
    return (math.tanh((step - center) / width) + 1.0) / 2.0


def validate_and_visualize(
    model,
    original_sequences,
    latent_dim,
    device,
    out_dir: Path,
    num_gen=500,
    fs=12000,
    use_data_start=True,
):
    model.eval()
    num_gen = min(num_gen, len(original_sequences))

    with torch.no_grad():
        z = torch.randn(num_gen, latent_dim, device=device)
        if use_data_start:
            idx = np.random.randint(0, len(original_sequences), size=num_gen)
            fi = torch.from_numpy(original_sequences[idx, 0:1, :].astype(np.float32)).to(device)
            gen_seq = model.decode(z, target_seq=None, first_inputs=fi)
        else:
            gen_seq = model.decode(z, target_seq=None)

    gen_seq = gen_seq.cpu().numpy().squeeze(-1)
    orig_seq = (
        original_sequences.squeeze(-1)
        if original_sequences.ndim == 3
        else original_sequences
    )

    num_orig = len(orig_seq)
    if num_orig > num_gen:
        pick = np.random.choice(num_orig, num_gen, replace=False)
        orig_sample = orig_seq[pick]
    else:
        orig_sample = orig_seq
        gen_seq = gen_seq[:num_orig]

    n_plots = min(5, len(orig_sample), len(gen_seq))
    fig, axes = plt.subplots(n_plots, 2, figsize=(12, 3 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_plots):
        axes[i, 0].plot(orig_sample[i], "b-", linewidth=1)
        axes[i, 0].set_title(f"Original {i + 1}")
        axes[i, 0].set_xlabel("Time step")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].plot(gen_seq[i], "r-", linewidth=1)
        axes[i, 1].set_title(f"Generated {i + 1}")
        axes[i, 1].set_xlabel("Time step")
        axes[i, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "signal_comparison.png", dpi=150)
    plt.close()

    orig_flat = np.concatenate(orig_sample, axis=0)
    gen_flat = np.concatenate(gen_seq, axis=0)

    flat_corr = float(np.corrcoef(orig_flat, gen_flat)[0, 1])
    stats = {
        "original_mean": float(np.mean(orig_flat)),
        "original_std": float(np.std(orig_flat)),
        "generated_mean": float(np.mean(gen_flat)),
        "generated_std": float(np.std(gen_flat)),
        "concatenated_correlation": flat_corr,
    }

    n_diag = min(512, len(orig_seq))
    x_np = orig_seq[:n_diag][:, :, np.newaxis] if orig_seq.ndim == 2 else orig_seq[:n_diag]
    x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
    with torch.no_grad():
        recon_tf, _, _ = model(x_t, teacher_force_ratio=1.0)
        mse_tf = F.mse_loss(recon_tf, x_t).item()
        mu, _ = model.encode(x_t)
        z_det = mu
        fi = x_t[:, 0:1, :]
        recon_ol = model.decode(z_det, target_seq=None, first_inputs=fi)
        mse_ol = F.mse_loss(recon_ol, x_t).item()

    diagnostic = {
        "mse_teacher_forced": float(mse_tf),
        "mse_open_loop": float(mse_ol),
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    freqs = np.fft.fftfreq(len(orig_flat), 1 / fs)
    axes[0].plot(freqs[: len(freqs) // 2], np.abs(np.fft.fft(orig_flat))[: len(freqs) // 2], "b-", alpha=0.7)
    axes[0].set_title("Spectrum — original")
    axes[0].set_xlim(0, fs / 2)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(freqs[: len(freqs) // 2], np.abs(np.fft.fft(gen_flat))[: len(freqs) // 2], "r-", alpha=0.7)
    axes[1].set_title("Spectrum — generated")
    axes[1].set_xlim(0, fs / 2)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "frequency_comparison.png", dpi=150)
    plt.close()

    return gen_seq, stats, diagnostic


def _prepare_data(bundle: AugDataBundle, cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    seq_len = model_cfg.get("seq_len", 128)
    stride = model_cfg.get("stride", 32)
    batch_size = model_cfg.get("batch_size", 256)

    raw_data = bundle.raw_data.astype(np.float32)
    data_norm, data_min, data_max = normalize_to_minus1_1(raw_data)
    sequences = create_sequences(data_norm, seq_len, stride=stride)
    data_tensor = torch.from_numpy(sequences.astype(np.float32))
    train_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

    return raw_data, data_norm, data_min, data_max, sequences, train_loader


def train(bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path):
    device = get_device(cfg)
    model_cfg = cfg["model"]
    exp_cfg = cfg.get("experiment", {})

    seed = exp_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    raw_data, data_norm, data_min, data_max, sequences, train_loader = _prepare_data(bundle, cfg)

    SEQ_LEN = model_cfg.get("seq_len", 128)
    INPUT_DIM = model_cfg.get("input_dim", 1)
    LATENT_DIM = model_cfg.get("latent_dim", 32)
    HIDDEN_DIM = model_cfg.get("hidden_dim", 128)
    NUM_LAYERS = model_cfg.get("num_layers", 1)
    EPOCHS = model_cfg.get("epochs", 100)
    LR = model_cfg.get("lr", 1e-3)

    PHASE1_TF_EPOCHS = model_cfg.get("phase1_tf_epochs", 40)
    RAMP_TFR_EPOCHS = model_cfg.get("ramp_tfr_epochs", 45)
    USE_REFERENCE_KL_SCHEDULE = model_cfg.get("use_reference_kl_schedule", True)
    BETA_DELAY_FRAC = model_cfg.get("beta_delay_frac", 0.42)
    BETA_END = model_cfg.get("beta_end", 0.04)
    FREE_BITS = model_cfg.get("free_bits", 0.0)
    TFR_MIN = model_cfg.get("tfr_min", 0.2)
    OPENLOOP_AUX_WEIGHT = model_cfg.get("openloop_aux_weight", 0.25)
    DEC_DROPOUT = model_cfg.get("dec_dropout", 0.05)
    WEIGHT_DECAY = model_cfg.get("weight_decay", 1e-5)
    LR_PATIENCE = model_cfg.get("lr_patience", 10)

    model = RVAE(SEQ_LEN, INPUT_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, dec_dropout=DEC_DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE, min_lr=1e-6
    )

    epochs_list = []
    loss_history = []
    recon_history = []
    kl_history = []
    ol_history = []
    beta_history = []
    tfr_history = []

    total_steps = max(1, EPOCHS * len(train_loader))
    delay_steps = int(BETA_DELAY_FRAC * total_steps)
    global_step = 0
    beta_after_delay = max(1, total_steps - delay_steps)

    best_recon = float("inf")
    best_epoch = 0

    print("Training on", device, "| steps", total_steps, "| KL delay", delay_steps)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        if epoch <= PHASE1_TF_EPOCHS:
            teacher_force_ratio = 1.0
        else:
            p = (epoch - PHASE1_TF_EPOCHS) / max(1, RAMP_TFR_EPOCHS)
            p = min(1.0, p)
            teacher_force_ratio = 1.0 - p * (1.0 - TFR_MIN)

        total_loss = total_recon = total_kl = total_ol = beta_epoch_sum = 0.0
        linear_wu = PHASE1_TF_EPOCHS + RAMP_TFR_EPOCHS

        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            if USE_REFERENCE_KL_SCHEDULE:
                if global_step < delay_steps:
                    beta = 0.0
                else:
                    adj = global_step - delay_steps
                    beta = BETA_END * reference_kld_scale(adj, beta_after_delay)
            else:
                if epoch <= PHASE1_TF_EPOCHS:
                    beta = 0.0
                elif epoch <= linear_wu:
                    beta = BETA_END * (epoch - PHASE1_TF_EPOCHS) / max(1, linear_wu - PHASE1_TF_EPOCHS)
                else:
                    beta = BETA_END
            global_step += 1
            beta_epoch_sum += beta

            recon_x, mu, logvar = model(x, teacher_force_ratio=teacher_force_ratio)
            recon_loss = F.mse_loss(recon_x, x, reduction="mean")
            kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
            if FREE_BITS > 0:
                kl_loss = torch.clamp(kl_loss, min=FREE_BITS)

            ol_w = OPENLOOP_AUX_WEIGHT if epoch > PHASE1_TF_EPOCHS else 0.0
            if ol_w > 0:
                ol_recon = model.decode(mu, target_seq=None, first_inputs=x[:, :1, :])
                ol_loss = F.mse_loss(ol_recon, x, reduction="mean")
            else:
                ol_loss = torch.zeros((), device=x.device)

            loss = recon_loss + beta * kl_loss + ol_w * ol_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_ol += float(ol_loss.detach().cpu())

        n_batch = len(train_loader)
        avg_recon = total_recon / n_batch
        scheduler.step(avg_recon)

        beta_avg = beta_epoch_sum / n_batch
        epochs_list.append(epoch)
        loss_history.append(total_loss / n_batch)
        recon_history.append(avg_recon)
        kl_history.append(total_kl / n_batch)
        ol_history.append(total_ol / n_batch)
        beta_history.append(beta_avg)
        tfr_history.append(teacher_force_ratio)

        if avg_recon < best_recon:
            best_recon = avg_recon
            best_epoch = epoch
            save_checkpoint_best(
                out_dir,
                {"model_state_dict": model.state_dict()},
                epoch=best_epoch,
                best_metric=best_recon,
                metric_name="train_recon_loss",
            )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/n_batch:.5f}  recon={avg_recon:.5f}  "
                f"kl={total_kl/n_batch:.5f}  ol={total_ol/n_batch:.5f}  beta~{beta_avg:.4f}  "
                f"tfr={teacher_force_ratio:.3f}  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    history = {
        "epochs": epochs_list,
        "total_loss": loss_history,
        "recon_loss": recon_history,
        "kl_loss": kl_history,
        "ol_loss": ol_history,
        "beta": beta_history,
        "teacher_force_ratio": tfr_history,
        "best_epoch": best_epoch,
    }
    save_loss_history(history, out_dir)
    _plot_rvae_curves(history, out_dir / "loss_curves.png")

    meta = {
        "sequences": sequences,
        "data_min": float(data_min),
        "data_max": float(data_max),
        "best_epoch": best_epoch,
        "best_recon_loss": best_recon,
    }
    save_json({"data_min": float(data_min), "data_max": float(data_max)}, out_dir / "norm_params.json")
    return model, meta


def _plot_rvae_curves(history, save_path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(history["epochs"], history["total_loss"], label="total")
    ax[0, 0].plot(history["epochs"], history["recon_loss"], label="recon (mean MSE)")
    ax[0, 0].plot(history["epochs"], history["ol_loss"], label="open-loop aux")
    ax[0, 0].legend()
    ax[0, 0].set_title("Loss")
    ax[0, 0].grid(True, alpha=0.3)
    ax[1, 0].plot(history["epochs"], history["kl_loss"])
    ax[1, 0].set_title("KL (mean per sequence)")
    ax[1, 0].grid(True, alpha=0.3)
    ax[0, 1].plot(history["epochs"], history["beta"])
    ax[0, 1].set_title("KL weight beta (avg / epoch)")
    ax[0, 1].grid(True, alpha=0.3)
    ax[1, 1].plot(history["epochs"], history["teacher_force_ratio"])
    ax[1, 1].set_title("Teacher force ratio")
    ax[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def load_checkpoint(path: Path, cfg: dict[str, Any]) -> RVAE:
    device = get_device(cfg)
    model_cfg = cfg["model"]
    model = RVAE(
        model_cfg.get("seq_len", 128),
        model_cfg.get("input_dim", 1),
        model_cfg.get("hidden_dim", 128),
        model_cfg.get("latent_dim", 32),
        model_cfg.get("num_layers", 1),
        dec_dropout=model_cfg.get("dec_dropout", 0.05),
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate(model: RVAE, bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path, meta: dict) -> np.ndarray:
    device = get_device(cfg)
    model = model.to(device)
    model_cfg = cfg["model"]

    sequences = meta["sequences"]
    num_gen = min(model_cfg.get("num_generate", 500), len(sequences))
    fs = float(cfg["dataset"].get("sample_rate", bundle.meta.get("sample_rate", 12000)))

    gen_norm, _, _ = validate_and_visualize(
        model,
        sequences,
        model.latent_dim,
        device,
        out_dir,
        num_gen=num_gen,
        fs=fs,
        use_data_start=model_cfg.get("use_data_start", True),
    )

    np.save(out_dir / "generated_samples.npy", gen_norm)
    data_min = meta["data_min"]
    data_max = meta["data_max"]
    gen_raw = denormalize_from_minus1_1(gen_norm, data_min, data_max)
    np.save(out_dir / "generated_samples_denorm.npy", gen_raw)
    return gen_norm


def evaluate(
    bundle: AugDataBundle,
    out_dir: Path,
    cfg: dict[str, Any],
    generated_samples: np.ndarray | None = None,
    model: RVAE | None = None,
    meta: dict | None = None,
) -> dict[str, Any]:
    device = get_device(cfg)
    model_cfg = cfg["model"]
    fs = float(cfg["dataset"].get("sample_rate", bundle.meta.get("sample_rate", 12000)))

    if meta is None:
        _, _, data_min, data_max, sequences, _ = _prepare_data(bundle, cfg)
        meta = {"sequences": sequences, "data_min": float(data_min), "data_max": float(data_max)}
    else:
        sequences = meta["sequences"]

    if model is None:
        model = load_checkpoint(out_dir / "checkpoint_best.pth", cfg)
    model = model.to(device)

    num_gen = min(model_cfg.get("num_generate", 500), len(sequences))
    _, stats, diagnostic = validate_and_visualize(
        model,
        sequences,
        model.latent_dim,
        device,
        out_dir,
        num_gen=num_gen,
        fs=fs,
        use_data_start=model_cfg.get("use_data_start", True),
    )

    metrics = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "dataset": cfg["dataset"]["name"],
        "model": "rvae",
        "statistics": stats,
        "reconstruction_diagnostic": diagnostic,
        "spectrum": {"sample_rate": fs},
    }
    if meta.get("best_epoch"):
        metrics["training_best"] = {
            "best_epoch": meta.get("best_epoch"),
            "best_recon_loss": meta.get("best_recon_loss"),
        }

    save_json(metrics, out_dir / "metrics.json")
    return metrics
