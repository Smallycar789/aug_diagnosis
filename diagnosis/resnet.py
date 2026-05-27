"""TL-ResNet18 + STFT — from ResNet.ipynb"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy.signal import stft
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

from diagnosis.common import (
    apply_global_norm,
    collect_multivariate_windows,
    load_class_files,
    prepare_signal_segment,
    resolve_class_window_region,
    resolve_normalize_mode,
    resolve_sample_length,
    resolve_stride,
    resolve_value_columns,
    zscore_window_channels,
)
from diagnosis.data_preprocess import DiagnosisDataBundle
from diagnosis.io_utils import get_device, resolve_path, save_checkpoint_best, save_json, save_loss_history


def _stft_magnitude_norm(signal, nperseg, noverlap, nfft):
    signal = np.asarray(signal)
    _, _, Zxx = stft(
        signal,
        fs=1.0,
        window="hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=True,
    )
    magnitude = np.abs(Zxx)
    return (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)


def _magnitude_map_to_rgb_image(mag_norm, target_size):
    mag_uint8 = (mag_norm * 255).astype(np.uint8)
    cmap = plt.get_cmap("jet")
    colored = cmap(mag_uint8 / 255.0)[:, :, :3]
    colored_uint8 = (colored * 255).astype(np.uint8)
    img = Image.fromarray(colored_uint8)
    return img.resize((target_size, target_size), Image.BILINEAR)


def signal_to_stft_rgb(signal, nperseg=64, noverlap=32, nfft=224, target_size=224):
    return _magnitude_map_to_rgb_image(_stft_magnitude_norm(signal, nperseg, noverlap, nfft), target_size)


def multivariate_window_to_stft_rgb(
    window,
    nperseg=64,
    noverlap=32,
    nfft=224,
    target_size=224,
    *,
    normalize: str = "per_window",
    norm_mean=None,
    norm_std=None,
):
    """Map each sensor channel to an STFT plane, then fuse into RGB for ResNet input."""
    window = np.asarray(window, dtype=np.float32)
    if window.ndim == 1:
        window = window[np.newaxis, :]

    mode = resolve_normalize_mode({"normalize": normalize})
    if mode == "none":
        normalized = window
    elif mode == "per_window":
        normalized = zscore_window_channels(window)
    elif mode == "global":
        if norm_mean is None or norm_std is None:
            raise ValueError("global normalization requires norm_mean and norm_std")
        normalized = apply_global_norm(window[np.newaxis, ...], norm_mean, norm_std)[0]
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")

    maps = [_stft_magnitude_norm(normalized[c], nperseg, noverlap, nfft) for c in range(normalized.shape[0])]
    if len(maps) == 1:
        rgb = np.stack([maps[0], maps[0], maps[0]], axis=-1)
    elif len(maps) == 2:
        blend = (maps[0] + maps[1]) / 2.0
        rgb = np.stack([maps[0], maps[1], blend], axis=-1)
    else:
        blend = maps[2] if len(maps) >= 3 else (maps[0] + maps[1]) / 2.0
        rgb = np.stack([maps[0], maps[1], blend], axis=-1)

    rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb_uint8)
    return img.resize((target_size, target_size), Image.BILINEAR)


def load_data_from_csv(csv_path, sample_len, nperseg, noverlap, nfft, target_size, has_header=False):
    df = pd.read_csv(csv_path, header=0 if has_header else None)
    num_classes = df.shape[1]
    images, labels = [], []
    for label in range(num_classes):
        signal_series = pd.to_numeric(df.iloc[:, label], errors="coerce").dropna().values.astype(np.float32)
        num_samples = len(signal_series) // sample_len
        for i in range(num_samples):
            start = i * sample_len
            end = start + sample_len
            segment = signal_series[start:end]
            img = signal_to_stft_rgb(segment, nperseg, noverlap, nfft, target_size)
            images.append(img)
            labels.append(label)
    return images, labels


def _segments_to_stft_images(
    segments,
    labels,
    nperseg,
    noverlap,
    nfft,
    target_size,
    *,
    normalize: str = "per_window",
    stft_input: str = "flat",
    norm_mean=None,
    norm_std=None,
):
    images, out_labels = [], []
    use_channel_stft = stft_input == "channel"
    for segment, label in zip(segments, labels):
        segment_arr = np.asarray(segment)
        if use_channel_stft and segment_arr.ndim == 2:
            img = multivariate_window_to_stft_rgb(
                segment_arr,
                nperseg,
                noverlap,
                nfft,
                target_size,
                normalize=normalize,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )
        else:
            segment = prepare_signal_segment(
                segment,
                normalize,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )
            if len(segment) == 0:
                continue
            img = signal_to_stft_rgb(segment, nperseg, noverlap, nfft, target_size)
        images.append(img)
        out_labels.append(int(label))
    return images, out_labels


def _stft_preprocess_kwargs(ds_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalize": ds_cfg.get("normalize", "per_window"),
        "stft_input": ds_cfg.get("stft_input", "flat"),
    }


def _load_multivariate_class_segments(ds_cfg, model_cfg):
    root = resolve_path(ds_cfg.get("root"))
    value_columns = resolve_value_columns(ds_cfg)
    sample_len = int(model_cfg.get("sample_len", resolve_sample_length(ds_cfg)))
    stride = resolve_stride(ds_cfg)
    class_files = load_class_files(ds_cfg, root)
    region_fraction = float(ds_cfg.get("region_fraction", 1.0))
    use_regions = bool(ds_cfg.get("class_window_regions"))

    segments, labels, _, label_names = collect_multivariate_windows(
        root,
        class_files,
        value_columns,
        sample_len,
        stride,
        unit_col=ds_cfg.get("unit_column", "unit"),
        cycle_col=ds_cfg.get("cycle_column", "cycle"),
        region_resolver=(lambda name: resolve_class_window_region(name, ds_cfg)) if use_regions else None,
        region_fraction=region_fraction,
        whole_file_as_series=bool(ds_cfg.get("whole_file_as_series", False)),
    )
    return segments, labels, label_names


def _load_cwru_images(ds_cfg, model_cfg, stft_kwargs):
    csv_path = str(resolve_path(ds_cfg.get("csv", "data/CWRU/CWRU_12k_1797_10c.csv")))
    sample_len = int(model_cfg.get("sample_len", 1024))
    has_header = bool(ds_cfg.get("has_header", True))
    df = pd.read_csv(csv_path, header=0 if has_header else None)
    label_names = df.columns.tolist()
    segments, labels = [], []
    for label_idx, col_name in enumerate(label_names):
        signal = pd.to_numeric(df[col_name], errors="coerce").dropna().values.astype(np.float32)
        num_samples = len(signal) // sample_len
        for i in range(num_samples):
            start = i * sample_len
            segments.append(signal[start : start + sample_len])
            labels.append(label_idx)
    images, labels = _segments_to_stft_images(segments, labels, **_stft_preprocess_kwargs(ds_cfg), **stft_kwargs)
    return images, labels, label_names


def _load_image_quality_images(ds_cfg, model_cfg, stft_kwargs):
    ds_cfg = {
        **ds_cfg,
        "root": ds_cfg.get("root", "data/image_quality"),
        "value_columns": ds_cfg.get(
            "value_columns",
            ["MTF50", "response_nonuniformity", "bad_pixel_rate"],
        ),
        "class_files": ds_cfg.get(
            "class_files",
            {
                "normal": "normal.csv",
                "mtf_degradation": "mtf_degradation.csv",
                "nonuniformity_degradation": "nonuniformity_degradation.csv",
                "bad_pixel_degradation": "bad_pixel_degradation.csv",
                "coupled_severe_fault": "coupled_severe_fault.csv",
            },
        ),
    }
    segments, labels, label_names = _load_multivariate_class_segments(ds_cfg, model_cfg)
    images, labels = _segments_to_stft_images(segments, labels, **_stft_preprocess_kwargs(ds_cfg), **stft_kwargs)
    return images, labels, label_names


def _load_sensitivity_images(ds_cfg, model_cfg, stft_kwargs):
    ds_cfg = {
        **ds_cfg,
        "root": ds_cfg.get("root", "data/sensitivity"),
        "value_columns": ds_cfg.get("value_columns", ["avg_detectivity", "NETD_mK"]),
        "class_files": ds_cfg.get(
            "class_files",
            {
                "normal": "normal.csv",
                "sensitivity_degradation": "sensitivity_degradation.csv",
                "coupled_severe_fault": "coupled_severe_fault.csv",
            },
        ),
    }
    segments, labels, label_names = _load_multivariate_class_segments(ds_cfg, model_cfg)
    images, labels = _segments_to_stft_images(segments, labels, **_stft_preprocess_kwargs(ds_cfg), **stft_kwargs)
    return images, labels, label_names


def _load_cooler_images(ds_cfg, model_cfg, stft_kwargs):
    root = resolve_path(ds_cfg.get("root", "data/cooler"))
    head_csv = root / ds_cfg.get("head_csv", "cooler_simulation_results/all_head_30.csv")
    tail_csv = root / ds_cfg.get("tail_csv", "cooler_simulation_results/all_tail_30.csv")
    value_columns = resolve_value_columns(
        {
            **ds_cfg,
            "value_columns": ds_cfg.get(
                "value_columns",
                ["T_stable_K", "t_cool_hours", "sigma_T_K"],
            ),
        }
    )
    sample_len = int(model_cfg.get("sample_len", ds_cfg.get("sample_length", 30)))
    group_window_len = int(ds_cfg.get("sample_length", 30))
    group_col = ds_cfg.get("group_column", "group_id")
    cycle_col = ds_cfg.get("cycle_column", "work_cycle")
    label_names = list(ds_cfg.get("label_names", ["normal", "degraded"]))

    segments, labels = [], []
    for label_id, csv_path in enumerate([head_csv, tail_csv]):
        df = pd.read_csv(csv_path)
        for _, group in df.groupby(group_col):
            group = group.sort_values(cycle_col)
            if len(group) != group_window_len:
                continue
            window = np.stack([group[column].values.astype(np.float32) for column in value_columns], axis=0)
            segments.append(window)
            labels.append(label_id)

    if sample_len != group_window_len * len(value_columns):
        raise ValueError(
            f"model.sample_len={sample_len} should equal "
            f"sample_length({group_window_len}) * channels({len(value_columns)}) "
            f"for cooler multivariate STFT input."
        )

    images, labels = _segments_to_stft_images(segments, labels, **_stft_preprocess_kwargs(ds_cfg), **stft_kwargs)
    return images, labels, label_names


def _load_sifuqi_images(ds_cfg, model_cfg, stft_kwargs):
    ds_cfg = {
        **ds_cfg,
        "root": ds_cfg.get("root", "data/sifuqi"),
        "value_columns": ds_cfg.get("value_columns", ["azimuth_error", "pitch_error"]),
        "levels": ds_cfg.get(
            "levels",
            {
                "normal": "servo_normal.csv",
                "mild": "servo_mild.csv",
                "moderate": "servo_moderate.csv",
                "severe": "servo_severe.csv",
            },
        ),
        "whole_file_as_series": True,
    }
    segments, labels, label_names = _load_multivariate_class_segments(
        {**ds_cfg, "class_files": ds_cfg["levels"]},
        model_cfg,
    )
    images, labels = _segments_to_stft_images(segments, labels, **_stft_preprocess_kwargs(ds_cfg), **stft_kwargs)
    return images, labels, label_names


def _load_images_and_labels(cfg: dict[str, Any]):
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    stft_kwargs = {
        "nperseg": int(model_cfg.get("nperseg", 64)),
        "noverlap": int(model_cfg.get("noverlap", 32)),
        "nfft": int(model_cfg.get("nfft", 224)),
        "target_size": int(model_cfg.get("target_size", 224)),
    }
    name = ds_cfg.get("name", "cwru_csv")

    if name in ("cwru_csv", "cwru"):
        return _load_cwru_images(ds_cfg, model_cfg, stft_kwargs)
    if name in ("image_quality", "degradation"):
        return _load_image_quality_images(ds_cfg, model_cfg, stft_kwargs)
    if name == "sensitivity":
        return _load_sensitivity_images(ds_cfg, model_cfg, stft_kwargs)
    if name == "cooler":
        return _load_cooler_images(ds_cfg, model_cfg, stft_kwargs)
    if name == "sifuqi":
        return _load_sifuqi_images(ds_cfg, model_cfg, stft_kwargs)

    csv_path = str(resolve_path(ds_cfg.get("csv", "data/CWRU/CWRU_12k_1797_10c.csv")))
    sample_len = int(model_cfg.get("sample_len", 1024))
    has_header = bool(ds_cfg.get("has_header", False))
    images, labels = load_data_from_csv(
        csv_path, sample_len, stft_kwargs["nperseg"], stft_kwargs["noverlap"], stft_kwargs["nfft"], stft_kwargs["target_size"], has_header
    )
    label_names = [f"class_{i}" for i in range(len(set(labels)))]
    return images, labels, label_names


class Wrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def build_tl_resnet18(num_classes, freeze_layers=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_layers:
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        model.layer4.apply(init_weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def _prepare_dataloaders(cfg: dict[str, Any], bundle: DiagnosisDataBundle | None = None):
    model_cfg = cfg["model"]
    batch_size = int(model_cfg.get("batch_size", 32))

    images, labels, label_names = _load_images_and_labels(cfg)
    if not images:
        raise ValueError("No STFT samples were generated. Check dataset paths and sample_len/stride settings.")

    num_classes = int(model_cfg.get("num_classes", len(label_names)))
    if num_classes != len(label_names):
        raise ValueError(
            f"model.num_classes={num_classes} but dataset has {len(label_names)} classes: {label_names}"
        )

    total = len(images)
    train_ratio = float(model_cfg.get("train_ratio", 0.6))
    val_ratio = float(model_cfg.get("val_ratio", 0.2))
    train_len = int(train_ratio * total)
    val_len = int(val_ratio * total)
    test_len = total - train_len - val_len
    if test_len <= 0:
        raise ValueError(f"Not enough samples ({total}) for train/val/test split.")

    generator = torch.Generator().manual_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    train_dataset, val_dataset, test_dataset = random_split(
        list(zip(images, labels)), [train_len, val_len, test_len], generator=generator
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader = DataLoader(Wrapper(train_dataset, transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Wrapper(val_dataset, transform), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Wrapper(test_dataset, transform), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, label_names, num_classes


def train(bundle: DiagnosisDataBundle | None, cfg: dict[str, Any], out_dir: Path):
    device = get_device(cfg)
    model_cfg = cfg["model"]

    train_loader, val_loader, test_loader, label_names, num_classes = _prepare_dataloaders(cfg, bundle)

    model = build_tl_resnet18(num_classes, freeze_layers=bool(model_cfg.get("freeze_layers", True))).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(model_cfg.get("lr", 0.001)),
        weight_decay=float(model_cfg.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(model_cfg.get("epochs", 30))
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = model.state_dict()

    if best_state is not None:
        save_checkpoint_best(
            out_dir,
            {"model_state_dict": best_state, "label_names": label_names, "num_classes": num_classes},
            epoch=best_epoch,
            best_metric=float(best_val_acc),
        )
        model.load_state_dict(best_state)

    _plot_curves(history, out_dir / "loss_curves.png")
    save_loss_history({**history, "best_epoch": best_epoch, "best_val_accuracy": float(best_val_acc)}, out_dir)

    meta = {
        "label_names": label_names,
        "num_classes": num_classes,
        "test_loader": test_loader,
        "best_epoch": best_epoch,
        "best_val_accuracy": float(best_val_acc),
    }
    return model, meta


def _plot_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title("Loss Curves")

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title("Accuracy Curves")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def load_checkpoint(path: Path, cfg: dict[str, Any]):
    device = get_device(cfg)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", int(cfg["model"].get("num_classes", 10)))
    model = build_tl_resnet18(num_classes, freeze_layers=bool(cfg["model"].get("freeze_layers", True))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate(model, bundle: DiagnosisDataBundle | None, cfg: dict[str, Any], out_dir: Path, split: str = "test", meta: dict | None = None) -> dict:
    device = get_device(cfg)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if meta and meta.get("test_loader") is not None and split == "test":
        test_loader = meta["test_loader"]
        label_names = meta.get("label_names", [])
    else:
        _, _, test_loader, label_names, _ = _prepare_dataloaders(cfg, bundle)

    test_loss, test_acc, preds, trues = validate(model, test_loader, criterion, device)

    metrics = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "dataset": cfg["dataset"]["name"],
        "model": "resnet",
        "split": split,
        "accuracy": float(test_acc),
        "loss": float(test_loss),
        "precision_macro": float(precision_score(trues, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(trues, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(trues, preds, average="macro", zero_division=0)),
        "classification_report": classification_report(trues, preds, output_dict=True),
        "label_names": label_names,
    }

    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix on Test Set (Acc={test_acc:.4f})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    metrics["confusion_matrix"] = cm.tolist()
    save_json(metrics, out_dir / "test_metrics.json")
    return metrics
