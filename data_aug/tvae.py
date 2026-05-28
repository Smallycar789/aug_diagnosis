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

from data_aug.common import (
    compare_signals,
    create_sequences,
    mean_bias_metrics,
    normalize_to_minus1_1,
)
from data_aug.shared import (
    build_per_class_norm_sequences,
    compute_sample_diversity,
    denormalize_by_label,
    denormalize_minus1_1_array,
    normalize_windows_minus1_1,
    physical_statistics,
    resolve_class_norm,
    strip_plot_compare_stats,
)
from data_aug.data_load import AugDataBundle
from data_aug.io_utils import get_device, save_checkpoint_best, save_json, save_loss_history


def _sinusoidal_position_encoding(seq_len: int, dim: int) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class TVAE(nn.Module):
    def __init__(
        self,
        seq_len,
        input_dim,
        hidden_dim,
        latent_dim,
        num_layers=1,
        dec_dropout=0.0,
        num_classes=0,
        label_embed_dim=0,
        use_attention_encoder=False,
        attention_heads=8,
        attention_layers=1,
        structured_decoder=False,
        trend_degree=2,
        seasonal_periods=None,
        sample_decoder_output=True,
        decoder_sample_scale=1.0,
        min_decoder_logvar=-7.0,
        max_decoder_logvar=1.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dec_dropout = dec_dropout
        self.use_attention_encoder = bool(use_attention_encoder)
        self.structured_decoder = bool(structured_decoder)
        self.trend_degree = int(trend_degree)
        self.seasonal_periods = [int(p) for p in (seasonal_periods or []) if int(p) > 1]
        self.sample_decoder_output = bool(sample_decoder_output)
        self.decoder_sample_scale = float(decoder_sample_scale)
        self.min_decoder_logvar = float(min_decoder_logvar)
        self.max_decoder_logvar = float(max_decoder_logvar)
        self.decoder_mean = None
        self.decoder_logvar = None
        self.num_classes = int(num_classes or 0)
        requested_label_embed_dim = int(label_embed_dim or 0)
        self.conditional = self.num_classes > 0 and requested_label_embed_dim > 0
        self.label_embed_dim = requested_label_embed_dim if self.conditional else 0
        self.label_embedding = (
            nn.Embedding(self.num_classes, self.label_embed_dim) if self.conditional else None
        )

        enc_in_dim = input_dim + self.label_embed_dim
        if self.use_attention_encoder:
            self.input_projection = nn.Linear(enc_in_dim, hidden_dim)
            self.register_buffer(
                "position_encoding",
                _sinusoidal_position_encoding(seq_len, hidden_dim),
                persistent=False,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=attention_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dec_dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.attention_encoder = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)
            enc_out_dim = seq_len * hidden_dim
            self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
            self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        else:
            self.encoder_lstm = nn.LSTM(
                enc_in_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.fc_mu = nn.Linear(2 * hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(2 * hidden_dim, latent_dim)

        dec_context_dim = latent_dim + self.label_embed_dim
        if self.structured_decoder:
            self.trend_head = nn.Linear(dec_context_dim, input_dim * (self.trend_degree + 1))
            self.num_seasonal_terms = 2 * len(self.seasonal_periods)
            self.seasonal_head = (
                nn.Linear(dec_context_dim, input_dim * self.num_seasonal_terms)
                if self.num_seasonal_terms
                else None
            )
            self.residual_head = nn.Sequential(
                nn.Linear(dec_context_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, seq_len * input_dim),
            )
            self.residual_refine = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1),
            )
            self.logvar_head = nn.Sequential(
                nn.Linear(dec_context_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, seq_len * input_dim),
            )
            t = torch.linspace(0.0, 1.0, seq_len)
            trend_basis = torch.stack([t.pow(i) for i in range(self.trend_degree + 1)], dim=0)
            self.register_buffer("trend_basis", trend_basis, persistent=False)
            if self.num_seasonal_terms:
                steps = torch.arange(seq_len, dtype=torch.float32)
                basis = []
                for period in self.seasonal_periods:
                    angle = 2.0 * math.pi * steps / float(period)
                    basis.extend([torch.sin(angle), torch.cos(angle)])
                self.register_buffer("seasonal_basis", torch.stack(basis, dim=0), persistent=False)
        else:
            dec_in = input_dim + latent_dim + self.label_embed_dim
            self.decoder_lstm = nn.LSTM(dec_in, hidden_dim, num_layers, batch_first=True)
            self.fc_out = nn.Linear(hidden_dim, input_dim)
        nn.init.constant_(self.fc_logvar.bias, -1.0)

    def _label_context(self, labels, batch_size, device):
        if not self.conditional:
            return None
        if labels is None:
            raise ValueError("Conditional T-VAE requires labels.")
        labels = labels.to(device).long()
        return self.label_embedding(labels)

    def _append_label_context(self, x, labels):
        ctx = self._label_context(labels, x.size(0), x.device)
        if ctx is None:
            return x
        ctx_seq = ctx.unsqueeze(1).expand(-1, x.size(1), -1)
        return torch.cat([x, ctx_seq], dim=-1)

    def encode(self, x, labels=None):
        enc_in = self._append_label_context(x, labels)
        if self.use_attention_encoder:
            h = self.input_projection(enc_in) + self.position_encoding[:, : enc_in.size(1), :].to(enc_in.device)
            h = self.attention_encoder(h)
            context = h.reshape(h.size(0), -1)
        else:
            _, (h_n, _) = self.encoder_lstm(enc_in)
            h = h_n.view(self.num_layers, 2, x.size(0), self.hidden_dim)[-1]
            context = torch.cat([h[0], h[1]], dim=-1)
        return self.fc_mu(context), self.fc_logvar(context)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _dec_inp(self, x_part, z, labels=None):
        if self.training and self.dec_dropout > 0:
            x_part = F.dropout(x_part, self.dec_dropout)
        z1 = z.unsqueeze(1).expand(-1, 1, -1)
        parts = [x_part, z1]
        ctx = self._label_context(labels, z.size(0), z.device)
        if ctx is not None:
            parts.append(ctx.unsqueeze(1))
        return torch.cat(parts, dim=-1)

    def _decoder_context(self, z, labels=None):
        parts = [z]
        ctx = self._label_context(labels, z.size(0), z.device)
        if ctx is not None:
            parts.append(ctx)
        return torch.cat(parts, dim=-1)

    def _decode_structured(self, z, labels=None, sample=None):
        context = self._decoder_context(z, labels)
        trend_coeff = self.trend_head(context).view(z.size(0), self.input_dim, self.trend_degree + 1)
        trend = torch.matmul(trend_coeff, self.trend_basis.to(z.device)).transpose(1, 2)

        if self.seasonal_head is not None:
            seasonal_coeff = self.seasonal_head(context).view(z.size(0), self.input_dim, self.num_seasonal_terms)
            seasonal = torch.matmul(seasonal_coeff, self.seasonal_basis.to(z.device)).transpose(1, 2)
        else:
            seasonal = torch.zeros_like(trend)

        residual = self.residual_head(context).view(z.size(0), self.seq_len, self.input_dim)
        residual = residual + self.residual_refine(residual.transpose(1, 2)).transpose(1, 2)
        mean = torch.tanh(trend + seasonal + residual)
        logvar = self.logvar_head(context).view(z.size(0), self.seq_len, self.input_dim)
        logvar = torch.clamp(logvar, self.min_decoder_logvar, self.max_decoder_logvar)
        self.decoder_mean = mean
        self.decoder_logvar = logvar

        if sample is None:
            sample = self.training or self.sample_decoder_output
        if sample:
            noise = torch.randn_like(mean) * torch.exp(0.5 * logvar) * self.decoder_sample_scale
            return torch.clamp(mean + noise, -1.2, 1.2)
        return mean

    def decode(self, z, target_seq=None, teacher_force_ratio=1.0, first_inputs=None, labels=None):
        if self.structured_decoder:
            return self._decode_structured(z, labels=labels)

        batch_size = z.size(0)
        hidden = None

        if target_seq is not None:
            outputs = []
            x_t = target_seq[:, 0:1, :]
            for t in range(self.seq_len):
                inp = self._dec_inp(x_t, z, labels)
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
            inp = self._dec_inp(x_t, z, labels)
            out_t, hidden = self.decoder_lstm(inp, hidden)
            out_t = self.fc_out(out_t)
            outputs.append(out_t)
            x_t = out_t
        return torch.cat(outputs, dim=1)

    def forward(self, x, labels=None, teacher_force_ratio=1.0):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, target_seq=x, teacher_force_ratio=teacher_force_ratio, labels=labels)
        return recon_x, mu, logvar

    def generate(self, num_samples, device, first_inputs=None, labels=None):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, target_seq=None, first_inputs=first_inputs, labels=labels)


def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
    return recon_loss + kl_loss, recon_loss, kl_loss


def reconstruction_loss(model, recon_x, x):
    if getattr(model, "structured_decoder", False) and model.decoder_mean is not None:
        dec_logvar = model.decoder_logvar
        dec_mean = model.decoder_mean
        return 0.5 * (((x - dec_mean).pow(2) / dec_logvar.exp()) + dec_logvar).mean()
    return F.mse_loss(recon_x, x, reduction="mean")


def reference_kld_scale(step, total_steps, mid_frac=0.45, width_frac=0.14):
    center = mid_frac * total_steps
    width = max(width_frac * total_steps, 1.0)
    return (math.tanh((step - center) / width) + 1.0) / 2.0


def _reconstruction_diagnostic(
    model,
    sequences,
    device,
    labels=None,
    label_id=None,
):
    n_diag = min(512, len(sequences))
    x_np = sequences[:n_diag]
    x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
    diag_labels = None
    if labels is not None:
        diag_labels_np = labels[:n_diag]
        if label_id is not None:
            diag_labels_np = np.full(n_diag, label_id, dtype=np.int64)
        diag_labels = torch.from_numpy(diag_labels_np.astype(np.int64)).to(device)
    with torch.no_grad():
        recon_tf, _, _ = model(x_t, labels=diag_labels, teacher_force_ratio=1.0)
        mse_tf = F.mse_loss(recon_tf, x_t).item()
        mu, _ = model.encode(x_t, diag_labels)
        fi = x_t[:, 0:1, :]
        recon_ol = model.decode(mu, target_seq=None, first_inputs=fi, labels=diag_labels)
        mse_ol = F.mse_loss(recon_ol, x_t).item()
    return {"mse_teacher_forced": float(mse_tf), "mse_open_loop": float(mse_ol)}


def validate_and_visualize(
    model,
    original_sequences,
    latent_dim,
    device,
    out_dir: Path,
    num_gen=500,
    fs=12000,
    use_data_start=True,
    labels=None,
    label_id=None,
    filename_prefix="",
    save_plots=False,
):
    model.eval()
    if (
        use_data_start
        and len(original_sequences) > 0
        and not getattr(model, "structured_decoder", False)
    ):
        num_gen = min(num_gen, len(original_sequences))
    label_tensor = None

    with torch.no_grad():
        z = torch.randn(num_gen, latent_dim, device=device)
        if use_data_start:
            idx = np.random.randint(0, len(original_sequences), size=num_gen)
            fi = torch.from_numpy(original_sequences[idx, 0:1, :].astype(np.float32)).to(device)
            if labels is not None:
                picked_labels = labels[idx]
                if label_id is not None:
                    picked_labels = np.full(num_gen, label_id, dtype=np.int64)
                label_tensor = torch.from_numpy(picked_labels.astype(np.int64)).to(device)
            gen_seq = model.decode(z, target_seq=None, first_inputs=fi, labels=label_tensor)
        else:
            if labels is not None:
                fill_label = 0 if label_id is None else int(label_id)
                label_tensor = torch.full((num_gen,), fill_label, dtype=torch.long, device=device)
            gen_seq = model.decode(z, target_seq=None, labels=label_tensor)

    gen_seq = gen_seq.cpu().numpy()
    diagnostic = _reconstruction_diagnostic(
        model,
        original_sequences,
        device,
        labels=labels,
        label_id=label_id,
    )
    return gen_seq, {}, diagnostic


def _prepare_data(bundle: AugDataBundle, cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    seq_len = model_cfg.get("seq_len", 128)
    stride = model_cfg.get("stride", 32)
    batch_size = model_cfg.get("batch_size", 256)
    use_per_class_norm = bool(model_cfg.get("per_class_norm", False))

    raw_data = bundle.raw_data.astype(np.float32)
    labels = bundle.labels
    per_class_norm: dict[str, dict[str, Any]] = {}
    if raw_data.ndim == 3 and use_per_class_norm and labels is not None:
        sequences, labels, per_class_norm = build_per_class_norm_sequences(
            raw_data, labels, bundle.label_names
        )
        data_min = raw_data.min(axis=(0, 1), keepdims=True)
        data_max = raw_data.max(axis=(0, 1), keepdims=True)
        data_norm = sequences
    elif raw_data.ndim == 3:
        data_norm, data_min, data_max = normalize_windows_minus1_1(raw_data)
        sequences = data_norm
    elif raw_data.ndim == 2:
        data_min = raw_data.min(axis=0, keepdims=True)
        data_max = raw_data.max(axis=0, keepdims=True)
        denom = np.where((data_max - data_min) < 1e-8, 1.0, data_max - data_min)
        data_norm = 2 * ((raw_data - data_min) / denom) - 1
        sequences = np.array(
            [data_norm[i : i + seq_len] for i in range(0, len(data_norm) - seq_len + 1, stride)],
            dtype=np.float32,
        )
    else:
        data_norm, data_min, data_max = normalize_to_minus1_1(raw_data)
        sequences = create_sequences(data_norm, seq_len, stride=stride)

    data_tensor = torch.from_numpy(sequences.astype(np.float32))
    if labels is not None:
        label_tensor = torch.from_numpy(labels.astype(np.int64))
        train_loader = DataLoader(TensorDataset(data_tensor, label_tensor), batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

    return raw_data, data_norm, data_min, data_max, sequences, labels, train_loader, per_class_norm


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

    raw_data, data_norm, data_min, data_max, sequences, labels, train_loader, per_class_norm = _prepare_data(
        bundle, cfg
    )

    SEQ_LEN = model_cfg.get("seq_len", 128)
    INPUT_DIM = model_cfg.get("input_dim", sequences.shape[-1])
    LATENT_DIM = model_cfg.get("latent_dim", 32)
    HIDDEN_DIM = model_cfg.get("hidden_dim", 128)
    NUM_LAYERS = model_cfg.get("num_layers", 1)
    EPOCHS = model_cfg.get("epochs", 100)
    LR = model_cfg.get("lr", 1e-3)
    CONDITIONAL = bool(model_cfg.get("conditional", labels is not None))
    NUM_CLASSES = len(bundle.label_names) if CONDITIONAL and labels is not None else 0
    LABEL_EMBED_DIM = model_cfg.get("label_embed_dim", 8 if NUM_CLASSES else 0)

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
    USE_ATTENTION_ENCODER = bool(model_cfg.get("use_attention_encoder", False))
    ATTENTION_HEADS = int(model_cfg.get("attention_heads", 8))
    ATTENTION_LAYERS = int(model_cfg.get("attention_layers", 1))
    STRUCTURED_DECODER = bool(model_cfg.get("structured_decoder", False))
    TREND_DEGREE = int(model_cfg.get("trend_degree", 2))
    SEASONAL_PERIODS = model_cfg.get("seasonal_periods", [])
    SAMPLE_DECODER_OUTPUT = bool(model_cfg.get("sample_decoder_output", True))
    DECODER_SAMPLE_SCALE = float(model_cfg.get("decoder_sample_scale", 1.0))
    MIN_DECODER_LOGVAR = float(model_cfg.get("min_decoder_logvar", -7.0))
    MAX_DECODER_LOGVAR = float(model_cfg.get("max_decoder_logvar", 1.0))

    model = TVAE(
        SEQ_LEN,
        INPUT_DIM,
        HIDDEN_DIM,
        LATENT_DIM,
        NUM_LAYERS,
        dec_dropout=DEC_DROPOUT,
        num_classes=NUM_CLASSES,
        label_embed_dim=LABEL_EMBED_DIM,
        use_attention_encoder=USE_ATTENTION_ENCODER,
        attention_heads=ATTENTION_HEADS,
        attention_layers=ATTENTION_LAYERS,
        structured_decoder=STRUCTURED_DECODER,
        trend_degree=TREND_DEGREE,
        seasonal_periods=SEASONAL_PERIODS,
        sample_decoder_output=SAMPLE_DECODER_OUTPUT,
        decoder_sample_scale=DECODER_SAMPLE_SCALE,
        min_decoder_logvar=MIN_DECODER_LOGVAR,
        max_decoder_logvar=MAX_DECODER_LOGVAR,
    ).to(device)
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

        for batch in train_loader:
            if len(batch) == 2:
                x, y = batch
                y = y.to(device)
            else:
                (x,) = batch
                y = None
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

            recon_x, mu, logvar = model(x, labels=y, teacher_force_ratio=teacher_force_ratio)
            recon_loss = reconstruction_loss(model, recon_x, x)
            kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
            if FREE_BITS > 0:
                kl_loss = torch.clamp(kl_loss, min=FREE_BITS)

            ol_w = OPENLOOP_AUX_WEIGHT if epoch > PHASE1_TF_EPOCHS else 0.0
            if ol_w > 0 and not model.structured_decoder:
                ol_recon = model.decode(mu, target_seq=None, first_inputs=x[:, :1, :], labels=y)
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
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": INPUT_DIM,
                    "num_classes": NUM_CLASSES,
                    "label_names": bundle.label_names,
                    "feature_columns": bundle.feature_columns,
                    "use_attention_encoder": USE_ATTENTION_ENCODER,
                    "attention_heads": ATTENTION_HEADS,
                    "attention_layers": ATTENTION_LAYERS,
                    "structured_decoder": STRUCTURED_DECODER,
                    "trend_degree": TREND_DEGREE,
                    "seasonal_periods": SEASONAL_PERIODS,
                    "sample_decoder_output": SAMPLE_DECODER_OUTPUT,
                    "decoder_sample_scale": DECODER_SAMPLE_SCALE,
                    "min_decoder_logvar": MIN_DECODER_LOGVAR,
                    "max_decoder_logvar": MAX_DECODER_LOGVAR,
                },
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
    _plot_tvae_curves(history, out_dir / "loss_curves.png")

    meta = {
        "sequences": sequences,
        "labels": labels,
        "label_names": bundle.label_names,
        "feature_columns": bundle.feature_columns,
        "data_min": data_min.tolist() if hasattr(data_min, "tolist") else float(data_min),
        "data_max": data_max.tolist() if hasattr(data_max, "tolist") else float(data_max),
        "best_epoch": best_epoch,
        "best_recon_loss": best_recon,
        "per_class_norm": per_class_norm,
    }
    norm_payload = {
        "data_min": meta["data_min"],
        "data_max": meta["data_max"],
        "label_names": bundle.label_names,
        "feature_columns": bundle.feature_columns,
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
    }
    if per_class_norm:
        norm_payload["per_class_norm"] = per_class_norm
    save_json(norm_payload, out_dir / "norm_params.json")
    return model, meta


def _plot_tvae_curves(history, save_path):
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


def load_checkpoint(path: Path, cfg: dict[str, Any]) -> TVAE:
    device = get_device(cfg)
    model_cfg = cfg["model"]
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    num_classes = int(checkpoint.get("num_classes", model_cfg.get("num_classes", 0)))
    label_embed_dim = int(model_cfg.get("label_embed_dim", 8 if num_classes else 0))
    model = TVAE(
        model_cfg.get("seq_len", 128),
        checkpoint.get("input_dim", model_cfg.get("input_dim", 1)),
        model_cfg.get("hidden_dim", 128),
        model_cfg.get("latent_dim", 32),
        model_cfg.get("num_layers", 1),
        dec_dropout=model_cfg.get("dec_dropout", 0.05),
        num_classes=num_classes,
        label_embed_dim=label_embed_dim,
        use_attention_encoder=bool(
            checkpoint.get("use_attention_encoder", model_cfg.get("use_attention_encoder", False))
        ),
        attention_heads=int(checkpoint.get("attention_heads", model_cfg.get("attention_heads", 8))),
        attention_layers=int(checkpoint.get("attention_layers", model_cfg.get("attention_layers", 1))),
        structured_decoder=bool(
            checkpoint.get("structured_decoder", model_cfg.get("structured_decoder", False))
        ),
        trend_degree=int(checkpoint.get("trend_degree", model_cfg.get("trend_degree", 2))),
        seasonal_periods=checkpoint.get("seasonal_periods", model_cfg.get("seasonal_periods", [])),
        sample_decoder_output=bool(
            checkpoint.get("sample_decoder_output", model_cfg.get("sample_decoder_output", True))
        ),
        decoder_sample_scale=float(
            checkpoint.get("decoder_sample_scale", model_cfg.get("decoder_sample_scale", 1.0))
        ),
        min_decoder_logvar=float(
            checkpoint.get("min_decoder_logvar", model_cfg.get("min_decoder_logvar", -7.0))
        ),
        max_decoder_logvar=float(
            checkpoint.get("max_decoder_logvar", model_cfg.get("max_decoder_logvar", 1.0))
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate(model: TVAE, bundle: AugDataBundle, cfg: dict[str, Any], out_dir: Path, meta: dict) -> np.ndarray:
    device = get_device(cfg)
    model = model.to(device)
    model_cfg = cfg["model"]

    sequences = meta["sequences"]
    labels = meta.get("labels")
    label_names = meta.get("label_names", [])
    fs = float(cfg["dataset"].get("sample_rate", bundle.meta.get("sample_rate", 12000)))
    data_min = np.asarray(meta["data_min"], dtype=np.float32)
    data_max = np.asarray(meta["data_max"], dtype=np.float32)
    norm_params = {"per_class_norm": meta.get("per_class_norm", {})}

    if labels is not None and model.conditional:
        generated_by_class = []
        generated_labels = []
        generated_denorm = []
        per_class = int(model_cfg.get("num_generate_per_class", model_cfg.get("num_generate", 200)))
        for label_id, label_name in enumerate(label_names):
            class_mask = labels == label_id
            class_sequences = sequences[class_mask]
            if len(class_sequences) == 0:
                continue
            gen_class, _, _ = validate_and_visualize(
                model,
                class_sequences,
                model.latent_dim,
                device,
                out_dir,
                num_gen=per_class,
                fs=fs,
                use_data_start=model_cfg.get("use_data_start", True),
                labels=np.full(len(class_sequences), label_id, dtype=np.int64),
                label_id=label_id,
                save_plots=False,
            )
            class_dmin, class_dmax = resolve_class_norm(label_name, norm_params, data_min, data_max)
            gen_denorm = denormalize_minus1_1_array(gen_class, class_dmin, class_dmax)
            generated_by_class.append(gen_class)
            generated_denorm.append(gen_denorm)
            generated_labels.extend([label_id] * len(gen_class))
            np.save(out_dir / f"generated_{label_name}.npy", gen_class)
            np.save(out_dir / f"generated_{label_name}_denorm.npy", gen_denorm)

        gen_norm = np.concatenate(generated_by_class, axis=0)
        gen_raw = np.concatenate(generated_denorm, axis=0)
        np.save(out_dir / "generated_labels.npy", np.asarray(generated_labels, dtype=np.int64))
    else:
        num_gen = min(model_cfg.get("num_generate", 500), len(sequences))
        gen_norm, _, _ = validate_and_visualize(
            model,
            sequences,
            model.latent_dim,
            device,
            out_dir,
            num_gen=num_gen,
            fs=fs,
            use_data_start=model_cfg.get("use_data_start", True),
            save_plots=False,
        )

    np.save(out_dir / "generated_samples.npy", gen_norm)
    if labels is None or not model.conditional:
        gen_raw = denormalize_minus1_1_array(gen_norm, data_min, data_max)
        np.save(out_dir / "generated_samples_denorm.npy", gen_raw)
    else:
        np.save(out_dir / "generated_samples_denorm.npy", gen_raw)
    return gen_norm


def evaluate(
    bundle: AugDataBundle,
    out_dir: Path,
    cfg: dict[str, Any],
    generated_samples: np.ndarray | None = None,
    model: TVAE | None = None,
    meta: dict | None = None,
) -> dict[str, Any]:
    from sklearn.metrics import mean_squared_error

    device = get_device(cfg)
    model_cfg = cfg["model"]
    sample_rate = float(cfg["dataset"].get("sample_rate", bundle.meta.get("sample_rate", 12000)))
    feature_names = bundle.feature_columns
    num_compare = int(model_cfg.get("num_compare", 5))
    spectrum_plot_style = model_cfg.get("spectrum_plot_style", "line")
    split_feature_plots = bool(model_cfg.get("split_feature_plots", True))

    if meta is None:
        _, _, data_min, data_max, sequences, labels, _, per_class_norm = _prepare_data(bundle, cfg)
        meta = {
            "sequences": sequences,
            "labels": labels,
            "label_names": bundle.label_names,
            "feature_columns": bundle.feature_columns,
            "data_min": data_min,
            "data_max": data_max,
            "per_class_norm": per_class_norm,
        }
    else:
        sequences = meta["sequences"]
        labels = meta.get("labels")
        data_min = np.asarray(meta["data_min"], dtype=np.float32)
        data_max = np.asarray(meta["data_max"], dtype=np.float32)
        per_class_norm = meta.get("per_class_norm", {})

    if model is None:
        model = load_checkpoint(out_dir / "checkpoint_best.pth", cfg)
    model = model.to(device)

    if generated_samples is None:
        generated_samples = np.load(out_dir / "generated_samples.npy")

    original_phys = bundle.raw_data.astype(np.float32)
    generated_labels_path = out_dir / "generated_labels.npy"
    generated_labels = (
        np.load(generated_labels_path)
        if generated_labels_path.exists() and bundle.labels is not None
        else None
    )

    norm_params = {"per_class_norm": per_class_norm}

    def _denorm_generated(samples: np.ndarray, label_indices: np.ndarray | None) -> np.ndarray:
        if label_indices is None or bundle.label_names is None:
            return denormalize_minus1_1_array(samples, data_min, data_max)
        return denormalize_by_label(
            samples,
            label_indices,
            bundle.label_names,
            norm_params,
            data_min,
            data_max,
        )

    per_class_metrics = {}
    if bundle.labels is not None and generated_labels is not None:
        for label_id, label_name in enumerate(bundle.label_names):
            class_mask = bundle.labels == label_id
            original_class = original_phys[class_mask]
            generated_class_norm = generated_samples[generated_labels == label_id]
            if len(original_class) == 0 or len(generated_class_norm) == 0:
                continue

            generated_class = _denorm_generated(generated_class_norm, None)
            n_class_compare = min(num_compare, len(original_class), len(generated_class))
            original_for_class = original_class[:n_class_compare]
            generated_for_class = generated_class[:n_class_compare]

            plot_stats = compare_signals(
                original_for_class,
                generated_for_class,
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
                physical_statistics(original_for_class, generated_for_class, feature_names)
            )

            class_mse = [
                mean_squared_error(
                    original_for_class[i].squeeze(),
                    generated_for_class[i].squeeze(),
                )
                for i in range(n_class_compare)
            ]
            class_sequences = sequences[class_mask]
            per_class_metrics[label_name] = {
                "statistics": class_stats,
                "reconstruction_diagnostic": _reconstruction_diagnostic(
                    model,
                    class_sequences,
                    device,
                    labels=np.full(len(class_sequences), label_id, dtype=np.int64),
                    label_id=label_id,
                ),
                "mse": {"avg_mse": float(np.mean(class_mse))},
                "diversity": compute_sample_diversity(generated_class),
                "num_original_windows": int(len(original_class)),
                "num_generated_windows": int(len(generated_class)),
                "generated_file": f"generated_{label_name}.npy",
                "signal_comparison": f"{label_name}_signal_comparison.png",
                "frequency_comparison": f"{label_name}_frequency_comparison.png",
                "feature_plots": plot_stats.get("feature_plots", {}),
                "generated_vs_original_mean_bias": mean_bias_metrics(
                    original_class,
                    generated_class,
                    feature_names=feature_names,
                ),
            }

    n_compare = min(num_compare, len(original_phys), len(generated_samples))
    original_for_comparison = original_phys[:n_compare]
    generated_for_comparison = _denorm_generated(generated_samples[:n_compare], None)

    plot_stats = compare_signals(
        original_for_comparison,
        generated_for_comparison,
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
        physical_statistics(original_for_comparison, generated_for_comparison, feature_names)
    )

    mse_list = []
    mse_per_list = []
    for i in range(n_compare):
        mse = mean_squared_error(
            original_for_comparison[i].squeeze(),
            generated_for_comparison[i].squeeze(),
        )
        mse_list.append(mse)
        denom = np.max(original_for_comparison[i]) - np.min(original_for_comparison[i])
        mse_per_list.append(mse / denom if denom != 0 else 0.0)

    generated_phys_all = _denorm_generated(generated_samples, generated_labels)

    metrics = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "dataset": cfg["dataset"]["name"],
        "model": "tvae",
        "feature_columns": bundle.feature_columns,
        "label_names": bundle.label_names,
        "input_shape": list(original_phys.shape[1:]),
        "per_class": per_class_metrics,
        "statistics": spectrum_stats,
        "reconstruction_diagnostic": _reconstruction_diagnostic(
            model, sequences, device, labels=labels
        ),
        "mse": {
            "avg_mse": float(np.mean(mse_list)) if mse_list else None,
            "avg_mse_percent": float(np.mean(mse_per_list)) if mse_per_list else None,
        },
        "diversity": compute_sample_diversity(generated_phys_all),
        "feature_plots": plot_stats.get("feature_plots", {}),
        "generated_vs_original_mean_bias": mean_bias_metrics(
            original_phys,
            generated_phys_all,
            feature_names=feature_names,
        ),
        "metric_scale": "physical",
    }
    if meta.get("best_epoch"):
        metrics["training_best"] = {
            "best_epoch": meta.get("best_epoch"),
            "best_recon_loss": meta.get("best_recon_loss"),
        }

    save_json(metrics, out_dir / "metrics.json")
    return metrics
