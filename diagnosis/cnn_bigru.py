"""1DCNN-BiGRU + MMD domain adaptation — from 1DCNN-BiGRU.ipynb"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from diagnosis.data_preprocess import DiagnosisDataBundle, load_data
from diagnosis.io_utils import get_device, save_checkpoint_best, save_json, save_loss_history


class SoftPool1D(nn.Module):
    def __init__(self, kernel_size=2, stride=None):
        super(SoftPool1D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        unfolded = x.unfold(2, self.kernel_size, self.stride)
        weights = torch.softmax(unfolded, dim=-1)
        pooled = (unfolded * weights).sum(dim=-1)
        return pooled


class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, hidden_dims=64, gru_hidden=128, num_layers=2):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.pool1 = SoftPool1D(kernel_size=2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.pool2 = SoftPool1D(kernel_size=2)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, hidden_dims, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
        )
        self.pool3 = SoftPool1D(kernel_size=2)

        self.gru = nn.GRU(
            input_size=hidden_dims,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_hidden = gru_hidden * 2

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        out = out.mean(dim=1)
        return out


class Classifier(nn.Module):
    def __init__(self, in_features, num_classes, hidden=64):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def mmd_rbf(x, y, sigma_list=None):
    if sigma_list is None:
        sigma_list = [1e-2, 1e-1, 1, 5, 10]
    x = nn.functional.normalize(x, p=2, dim=1)
    y = nn.functional.normalize(y, p=2, dim=1)

    batch_s = x.size(0)
    batch_t = y.size(0)

    xx = torch.mm(x, x.t())
    xy = torch.mm(x, y.t())
    yy = torch.mm(y, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    d_xx = rx.t() + rx - 2 * xx
    d_yy = ry.t() + ry - 2 * yy

    rx_expand = xx.diag().unsqueeze(0).expand(batch_s, batch_t)
    ry_expand = yy.diag().unsqueeze(0).expand(batch_s, batch_t)
    d_xy = rx_expand + ry_expand - 2 * xy

    d_xx = torch.clamp(d_xx, min=0)
    d_xy = torch.clamp(d_xy, min=0)
    d_yy = torch.clamp(d_yy, min=0)

    mmd = torch.tensor(0.0, device=x.device)
    for sigma in sigma_list:
        exp_arg_xx = torch.clamp(-0.5 * d_xx / (sigma**2), max=50)
        exp_arg_xy = torch.clamp(-0.5 * d_xy / (sigma**2), max=50)
        exp_arg_yy = torch.clamp(-0.5 * d_yy / (sigma**2), max=50)

        kernel_xx = torch.exp(exp_arg_xx)
        kernel_xy = torch.exp(exp_arg_xy)
        kernel_yy = torch.exp(exp_arg_yy)

        mmd += kernel_xx.mean() + kernel_yy.mean() - 2 * kernel_xy.mean()

    mmd = torch.clamp(mmd, min=1e-8)
    return mmd.sqrt()


class DomainAdaptationModel(nn.Module):
    def __init__(self, input_channels, num_classes, seq_len=1024, gru_hidden=128, num_layers=2, hidden_dims=64):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = FeatureExtractor(
            input_channels=input_channels,
            hidden_dims=hidden_dims,
            gru_hidden=gru_hidden,
            num_layers=num_layers,
        )
        with torch.no_grad():
            dummy = torch.randn(1, input_channels, seq_len)
            feat_dim = self.feature_extractor(dummy).shape[1]
        self.classifier = Classifier(in_features=feat_dim, num_classes=num_classes)

    def forward(self, x_s, x_t=None):
        feat_s = self.feature_extractor(x_s)
        logits_s = self.classifier(feat_s)

        mmd_loss = torch.tensor(0.0, device=x_s.device)
        if x_t is not None:
            feat_t = self.feature_extractor(x_t)
            mmd_loss = mmd_rbf(feat_s, feat_t)
        return logits_s, mmd_loss


def train_model(model, source_loader, target_loader, val_loader, epochs, alpha, device, lr=5e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_ces, train_mmds, val_accs = [], [], [], []
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss, total_ce, total_mmd = 0.0, 0.0, 0.0

        if alpha > 0:
            if target_loader is None:
                raise ValueError("alpha > 0 requires target domain data (bundle.X_target). For sifuqi scheme A, set alpha: 0.")
            batch_pairs = zip(source_loader, target_loader)
            n_batches = min(len(source_loader), len(target_loader))
        else:
            batch_pairs = ((batch, (None, None)) for batch in source_loader)
            n_batches = len(source_loader)

        for batch_s, batch_t in batch_pairs:
            x_s, y_s = batch_s
            x_s, y_s = x_s.to(device), y_s.to(device).long()

            optimizer.zero_grad()
            if alpha > 0:
                x_t, _ = batch_t
                x_t = x_t.to(device)
                logits_s, mmd_loss = model(x_s, x_t)
            else:
                logits_s, mmd_loss = model(x_s, x_t=None)
                mmd_loss = torch.tensor(0.0, device=device)

            ce_loss = criterion(logits_s, y_s)
            loss = ce_loss + alpha * mmd_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_mmd += mmd_loss.item()

        avg_loss = total_loss / max(n_batches, 1)
        avg_ce = total_ce / max(n_batches, 1)
        avg_mmd = total_mmd / max(n_batches, 1)
        train_losses.append(avg_loss)
        train_ces.append(avg_ce)
        train_mmds.append(avg_mmd)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x_s, y_s in val_loader:
                x_s = x_s.to(device)
                logits_s, _ = model(x_s, x_t=None)
                pred = torch.argmax(logits_s, dim=1).cpu()
                preds.extend(pred.numpy())
                trues.extend(y_s.numpy())

        val_acc = accuracy_score(trues, preds)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d} | Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, MMD: {avg_mmd:.4f}) | Val Acc: {val_acc:.4f}"
            )

    history = {
        "train_loss": train_losses,
        "train_ce": train_ces,
        "train_mmd": train_mmds,
        "val_accuracy": val_accs,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
    }
    return history, best_state, best_epoch, best_val_acc


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0, 0].plot(epochs, history["train_loss"], "b-")
    axes[0, 0].set_title("Total Loss")
    axes[0, 1].plot(epochs, history["train_ce"], "r-")
    axes[0, 1].set_title("Cross Entropy Loss")
    axes[1, 0].plot(epochs, history["train_mmd"], "g-")
    axes[1, 0].set_title("MMD Loss")
    axes[1, 1].plot(epochs, history["val_accuracy"], "m-")
    axes[1, 1].set_title("Validation Accuracy")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, label_names, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return cm


def _build_loaders(bundle: DiagnosisDataBundle, cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    batch_size = int(model_cfg.get("batch_size", 64))
    val_ratio = float(model_cfg.get("val_ratio", 0.2))
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    alpha = float(model_cfg.get("alpha", 1.0))

    X_train = bundle.X_source
    y_train = bundle.y_source
    X_val = bundle.meta.get("X_val")
    y_val = bundle.meta.get("y_val")

    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, random_state=seed, stratify=y_train
        )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    target_loader = None
    X_target, y_target = None, None
    if alpha > 0 and bundle.X_target is not None and bundle.y_target is not None:
        X_target = bundle.X_target
        y_target = bundle.y_target
        target_loader = DataLoader(
            TensorDataset(torch.tensor(X_target), torch.tensor(y_target)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    return train_loader, val_loader, target_loader, X_val, y_val, X_target, y_target


def train(bundle: DiagnosisDataBundle, cfg: dict[str, Any], out_dir: Path):
    device = get_device(cfg)
    model_cfg = cfg["model"]

    expected_channels = int(bundle.meta.get("num_channels", model_cfg.get("input_channels", 1)))
    input_channels = int(model_cfg.get("input_channels", expected_channels))
    if input_channels != expected_channels:
        raise ValueError(
            f"model.input_channels={input_channels} but dataset has {expected_channels} channel(s). "
            f"Check value_columns in config."
        )

    train_loader, val_loader, target_loader, _, _, _, _ = _build_loaders(bundle, cfg)

    seq_len = int(model_cfg.get("sample_length", bundle.X_source.shape[-1]))
    model = DomainAdaptationModel(
        input_channels=int(model_cfg.get("input_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", len(bundle.label_names))),
        seq_len=seq_len,
        gru_hidden=int(model_cfg.get("gru_hidden", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        hidden_dims=int(model_cfg.get("hidden_dims", 64)),
    ).to(device)

    history, best_state, best_epoch, best_val_acc = train_model(
        model,
        train_loader,
        target_loader,
        val_loader,
        epochs=int(model_cfg.get("epochs", 50)),
        alpha=float(model_cfg.get("alpha", 1.0)),
        device=device,
        lr=float(model_cfg.get("lr", 5e-4)),
    )

    if best_state is not None:
        save_checkpoint_best(
            out_dir,
            {"model_state_dict": best_state, "label_names": bundle.label_names},
            epoch=best_epoch,
            best_metric=best_val_acc,
            metric_name="val_accuracy",
        )
        model.load_state_dict(best_state)

    plot_training_curves(history, out_dir / "loss_curves.png")
    save_loss_history(history, out_dir)

    meta = {"label_names": bundle.label_names, "best_epoch": best_epoch, "best_val_accuracy": best_val_acc}
    return model, meta


def load_checkpoint(path: Path, cfg: dict[str, Any], label_names: Optional[List[str]] = None):
    device = get_device(cfg)
    model_cfg = cfg["model"]
    checkpoint = torch.load(path, map_location=device)
    if label_names is None:
        label_names = checkpoint.get("label_names", [])

    seq_len = int(model_cfg.get("sample_length", 1024))
    model = DomainAdaptationModel(
        input_channels=int(model_cfg.get("input_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", len(label_names) or 10)),
        seq_len=seq_len,
        gru_hidden=int(model_cfg.get("gru_hidden", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        hidden_dims=int(model_cfg.get("hidden_dims", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate(model, bundle: DiagnosisDataBundle, cfg: dict[str, Any], out_dir: Path, split: str = "test") -> dict:
    device = get_device(cfg)
    model = model.to(device)
    model.eval()
    model_cfg = cfg["model"]
    batch_size = int(model_cfg.get("batch_size", 64))
    label_names = bundle.label_names

    if split == "val":
        X = bundle.meta.get("X_val", bundle.X_source)
        y = bundle.meta.get("y_val", bundle.y_source)
        title = "Confusion Matrix on Validation Set"
    else:
        X = bundle.meta.get("X_test", bundle.X_target)
        y = bundle.meta.get("y_test", bundle.y_target)
        title = "Confusion Matrix on Test Set"

    loader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(y)),
        batch_size=batch_size,
        shuffle=False,
    )

    preds, trues = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            logits, _ = model(x_b, x_t=None)
            pred = torch.argmax(logits, dim=1).cpu()
            preds.extend(pred.numpy())
            trues.extend(y_b.numpy())

    preds = np.array(preds)
    trues = np.array(trues)

    metrics = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "dataset": cfg["dataset"]["name"],
        "model": "cnn_bigru",
        "split": split,
        "accuracy": float(accuracy_score(trues, preds)),
        "precision_macro": float(precision_score(trues, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(trues, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(trues, preds, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(trues, preds)),
        "classification_report": classification_report(trues, preds, target_names=label_names, output_dict=True),
    }

    cm = plot_confusion_matrix(trues, preds, label_names, out_dir / "confusion_matrix.png", title=title)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["label_names"] = label_names

    save_json(metrics, out_dir / "test_metrics.json")
    return metrics
