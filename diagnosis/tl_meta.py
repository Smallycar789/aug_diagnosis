"""SSMN semi-supervised meta-learning — from TL-Meta.ipynb"""

from __future__ import annotations

from collections import defaultdict
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from diagnosis.common import (
    collect_multivariate_windows,
    flatten_multivariate_window,
    load_class_files,
    resolve_class_window_region,
    resolve_sample_length,
    resolve_stride,
    resolve_value_columns,
    zscore_window_channels,
)
from diagnosis.data_preprocess import DiagnosisDataBundle
from diagnosis.io_utils import get_device, resolve_path, save_checkpoint_best, save_json, save_loss_history


class TLMetaSignalDataset(Dataset):
    """1D meta-learning samples for SSMN encoder input."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, class_names: list[str], indices: np.ndarray | None = None):
        self.data = np.asarray(data, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.class_names = list(class_names)
        if indices is not None:
            self.data = self.data[indices]
            self.labels = self.labels[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]).unsqueeze(0), torch.LongTensor([self.labels[idx]])[0]


class CWRUDataset(Dataset):
    def __init__(
        self,
        csv_path,
        class_names=None,
        samples_per_class=200,
        signal_length=2048,
        overlap=0.5,
        normalize=True,
        is_train=True,
        indices=None,
    ):
        self.csv_path = csv_path
        self.signal_length = signal_length
        self.overlap = overlap
        self.normalize = normalize
        self.is_train = is_train

        df = pd.read_csv(csv_path)
        if class_names is None:
            self.class_names = df.columns.tolist()
        else:
            self.class_names = class_names

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples_per_class = samples_per_class
        self.data, self.labels = self._load_data_from_csv(df)

        if indices is not None:
            self.data = self.data[indices]
            self.labels = self.labels[indices]

    def _load_data_from_csv(self, df):
        all_signals, all_labels = [], []
        for col_idx, class_name in enumerate(self.class_names):
            raw_signal = df[class_name].values.flatten()
            raw_signal = raw_signal[~np.isnan(raw_signal)]
            signals = self._split_signal(raw_signal)
            signals = signals[: self.samples_per_class]
            labels = [col_idx] * len(signals)
            all_signals.extend(signals)
            all_labels.extend(labels)

        all_signals = np.array(all_signals)
        all_labels = np.array(all_labels)
        if self.normalize:
            all_signals = self._normalize(all_signals)
        return all_signals, all_labels

    def _split_signal(self, signal):
        step = int(self.signal_length * (1 - self.overlap))
        samples = []
        for start in range(0, len(signal) - self.signal_length + 1, step):
            sample = signal[start : start + self.signal_length]
            samples.append(sample)
            if len(samples) >= self.samples_per_class:
                break
        return samples

    def _normalize(self, signals):
        norms = np.linalg.norm(signals, axis=1, keepdims=True) ** 2
        norms[norms == 0] = 1
        return signals / norms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]).unsqueeze(0), torch.LongTensor([self.labels[idx]])[0]


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=32, stride=1, padding=16, pool_size=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool_size, stride=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(1, 64, kernel_size=32, padding=16)
        self.conv2 = ConvBlock(64, 64, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(64, 64, kernel_size=3, padding=1)
        self.conv4 = ConvBlock(64, 64, kernel_size=3, padding=1)
        self.se1 = SEBlock(64, reduction=4)
        self.se2 = SEBlock(64, reduction=4)
        self.flatten_dim = 64 * 128
        self.fc = nn.Linear(self.flatten_dim, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class SSMN(nn.Module):
    def __init__(self, encoder, n_way=5, embedding_dim=128, refine_iterations=2):
        super(SSMN, self).__init__()
        self.encoder = encoder
        self.n_way = n_way
        self.embedding_dim = embedding_dim
        self.refine_iterations = refine_iterations

    def euclidean_distance(self, x, y):
        return torch.cdist(x, y, p=2) ** 2

    def compute_prototypes(self, support_embeddings, support_labels, unlabeled_embeddings=None, prototypes=None):
        n_support = support_embeddings.shape[0]
        n_way = self.n_way

        if prototypes is None:
            prototypes = torch.zeros(n_way, self.embedding_dim, device=support_embeddings.device)
            for c in range(n_way):
                mask = support_labels == c
                if mask.sum() > 0:
                    prototypes[c] = support_embeddings[mask].mean(dim=0)

        if unlabeled_embeddings is None or unlabeled_embeddings.shape[0] == 0:
            return prototypes

        for _ in range(self.refine_iterations):
            distances = self.euclidean_distance(unlabeled_embeddings, prototypes)
            unlabeled_probs = F.softmax(-distances, dim=1)

            label_probs = torch.zeros(n_support, n_way, device=support_embeddings.device)
            label_probs[torch.arange(n_support), support_labels] = 1.0

            sum_label = label_probs.sum(dim=0)
            sum_unlabeled = unlabeled_probs.sum(dim=0)
            total_sum = sum_label + sum_unlabeled + 1e-8

            k_labeled = label_probs / total_sum.unsqueeze(0)
            k_unlabeled = unlabeled_probs / total_sum.unsqueeze(0)
            prototypes = (k_labeled.T @ support_embeddings) + (k_unlabeled.T @ unlabeled_embeddings)

        return prototypes

    def forward(self, support_embeddings, support_labels, unlabeled_embeddings, query_embeddings):
        prototypes = self.compute_prototypes(support_embeddings, support_labels, unlabeled_embeddings)
        distances = self.euclidean_distance(query_embeddings, prototypes)
        query_probs = F.softmax(-distances, dim=1)
        return query_probs, prototypes


class CombinatorialOptimizer:
    def __init__(
        self,
        model_params,
        l_skip=0.15,
        sgd_lr=0.2,
        sgd_gamma=0.9,
        sgd_step_decay=500,
        adam_lr=0.001,
        adam_betas=(0.9, 0.999),
    ):
        self.l_skip = l_skip
        self.sgd_lr = sgd_lr
        self.sgd_gamma = sgd_gamma
        self.sgd_step_decay = sgd_step_decay
        self.adam_lr = adam_lr
        self.adam_betas = adam_betas
        self.epoch = 0
        self.use_sgd = True

        params_list = list(model_params)
        if len(params_list) == 0:
            raise ValueError("No parameters received.")

        self.sgd_optimizer = torch.optim.SGD(params_list, lr=sgd_lr)
        self.adam_optimizer = torch.optim.Adam(params_list, lr=adam_lr, betas=adam_betas)

    def step(self):
        if self.use_sgd:
            if self.epoch > 0 and self.epoch % self.sgd_step_decay == 0:
                current_lr = self.sgd_optimizer.param_groups[0]["lr"]
                new_lr = current_lr * self.sgd_gamma
                for param_group in self.sgd_optimizer.param_groups:
                    param_group["lr"] = new_lr
            self.sgd_optimizer.step()
        else:
            self.adam_optimizer.step()
        self.epoch += 1

    def update_optimizer(self, loss):
        if self.use_sgd and loss.item() <= self.l_skip:
            self.use_sgd = False

    def zero_grad(self):
        self.sgd_optimizer.zero_grad()
        self.adam_optimizer.zero_grad()


class EpisodeSampler:
    def __init__(self, dataset, n_way, n_support, n_unlabeled, n_query, allow_replace: bool = False):
        self.dataset = dataset
        self.n_way = n_way
        self.n_support = n_support
        self.n_unlabeled = n_unlabeled
        self.n_query = n_query
        self.allow_replace = allow_replace

        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            self.class_to_indices[label].append(idx)
        self.classes = list(self.class_to_indices.keys())

    def sample_episode(self):
        selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
        support_indices, support_labels = [], []
        unlabeled_indices, query_indices, query_labels = [], [], []

        for class_idx, class_id in enumerate(selected_classes):
            indices = self.class_to_indices[class_id].copy()
            np.random.shuffle(indices)
            total_needed = self.n_support + self.n_unlabeled + self.n_query
            if len(indices) < total_needed:
                if not self.allow_replace:
                    raise ValueError(f"Class {class_id} has only {len(indices)} samples, need {total_needed}")
                picked = np.random.choice(indices, size=total_needed, replace=True)
            else:
                picked = indices[:total_needed]

            for i in range(self.n_support):
                support_indices.append(int(picked[i]))
                support_labels.append(class_idx)
            for i in range(self.n_support, self.n_support + self.n_unlabeled):
                unlabeled_indices.append(int(picked[i]))
            offset = self.n_support + self.n_unlabeled
            for i in range(offset, offset + self.n_query):
                query_indices.append(int(picked[i]))
                query_labels.append(class_idx)

        return {
            "support_indices": support_indices,
            "support_labels": torch.LongTensor(support_labels),
            "unlabeled_indices": unlabeled_indices,
            "query_indices": query_indices,
            "query_labels": torch.LongTensor(query_labels),
            "selected_classes": selected_classes,
        }


def train_ssmn(
    model,
    train_dataset,
    val_dataset,
    device,
    n_way,
    n_support,
    n_unlabeled,
    n_query,
    n_episodes=5000,
    val_interval=200,
    l_skip=0.15,
    sgd_lr=0.2,
    adam_lr=0.001,
    allow_episode_replace: bool = False,
):
    train_sampler = EpisodeSampler(
        train_dataset, n_way, n_support, n_unlabeled, n_query, allow_replace=allow_episode_replace
    )
    val_sampler = (
        EpisodeSampler(val_dataset, n_way, n_support, n_unlabeled, n_query, allow_replace=allow_episode_replace)
        if val_dataset is not None
        else None
    )

    optimizer = CombinatorialOptimizer(model.parameters(), l_skip=l_skip, sgd_lr=sgd_lr, adam_lr=adam_lr)

    train_losses, val_accuracies = [], []
    best_val_acc = 0.0
    best_state = None
    best_episode = 0

    for episode in tqdm(range(n_episodes), desc="Training Episodes"):
        episode_data = train_sampler.sample_episode()

        support_signals = torch.stack([train_dataset[i][0] for i in episode_data["support_indices"]]).to(device)
        unlabeled_signals = torch.stack([train_dataset[i][0] for i in episode_data["unlabeled_indices"]]).to(device)
        query_signals = torch.stack([train_dataset[i][0] for i in episode_data["query_indices"]]).to(device)

        support_labels = episode_data["support_labels"].to(device)
        query_labels = episode_data["query_labels"].to(device)

        support_embeddings = model.encoder(support_signals)
        unlabeled_embeddings = model.encoder(unlabeled_signals)
        query_embeddings = model.encoder(query_signals)

        query_probs, _ = model(support_embeddings, support_labels, unlabeled_embeddings, query_embeddings)
        # query_probs is already normalized; use NLL on log-probabilities for stable optimization.
        loss = F.nll_loss(torch.log(query_probs.clamp_min(1e-8)), query_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.update_optimizer(loss)
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(loss.item())

        if val_sampler is not None and (episode + 1) % val_interval == 0:
            val_acc, _, _ = evaluate_ssmn(
                model,
                val_dataset,
                device,
                n_way,
                n_support,
                n_unlabeled,
                n_query,
                n_episodes=100,
                allow_episode_replace=allow_episode_replace,
            )
            val_accuracies.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_episode = episode + 1
                best_state = model.state_dict()
                print(f"\nEpisode {episode + 1}: Loss={loss.item():.4f}, Val Acc={val_acc:.2f}% (Best)")
            else:
                print(f"\nEpisode {episode + 1}: Loss={loss.item():.4f}, Val Acc={val_acc:.2f}%")

    history = {
        "train_loss": train_losses,
        "val_accuracy": val_accuracies,
        "best_episode": best_episode,
        "best_val_accuracy": best_val_acc,
        "val_interval": val_interval,
    }
    return history, best_state, best_episode, best_val_acc


@torch.no_grad()
def evaluate_ssmn(
    model,
    test_dataset,
    device,
    n_way,
    n_support,
    n_unlabeled,
    n_query,
    n_episodes=500,
    allow_episode_replace: bool = False,
):
    model.eval()
    sampler = EpisodeSampler(
        test_dataset, n_way, n_support, n_unlabeled, n_query, allow_replace=allow_episode_replace
    )

    correct, total = 0, 0
    all_predictions, all_labels = [], []

    for _ in range(n_episodes):
        episode_data = sampler.sample_episode()
        support_signals = torch.stack([test_dataset[i][0] for i in episode_data["support_indices"]]).to(device)
        unlabeled_signals = torch.stack([test_dataset[i][0] for i in episode_data["unlabeled_indices"]]).to(device)
        query_signals = torch.stack([test_dataset[i][0] for i in episode_data["query_indices"]]).to(device)

        support_labels = episode_data["support_labels"].to(device)
        query_labels = episode_data["query_labels"].to(device)

        support_embeddings = model.encoder(support_signals)
        unlabeled_embeddings = model.encoder(unlabeled_signals)
        query_embeddings = model.encoder(query_signals)

        query_probs, _ = model(support_embeddings, support_labels, unlabeled_embeddings, query_embeddings)
        predictions = torch.argmax(query_probs, dim=1)

        correct += (predictions == query_labels).sum().item()
        total += len(query_labels)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(query_labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    model.train()
    return accuracy, np.array(all_predictions), np.array(all_labels)


def plot_training_curves(train_losses, val_accuracies, val_interval, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, alpha=0.7, linewidth=0.5)
    if len(train_losses) > 100:
        smoothed = np.convolve(train_losses, np.ones(50) / 50, mode="valid")
        axes[0].plot(np.arange(25, 25 + len(smoothed)), smoothed, "r", linewidth=2, label="Smoothed")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    if val_accuracies:
        val_episodes = np.arange(val_interval, len(val_accuracies) * val_interval + 1, val_interval)
        axes[1].plot(val_episodes, val_accuracies, "o-", linewidth=2, markersize=4)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return cm


def _pad_or_truncate_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if len(signal) == target_length:
        return signal
    if len(signal) > target_length:
        return signal[:target_length]
    out = np.zeros(target_length, dtype=np.float32)
    out[: len(signal)] = signal
    return out


def _l2_normalize_signals(signals: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(signals, axis=1, keepdims=True) ** 2
    norms[norms == 0] = 1
    return signals / norms


def _merge_window(window: np.ndarray, merge_mode: str = "flatten") -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    if merge_mode == "flatten":
        return flatten_multivariate_window(window)
    if window.ndim == 1:
        return window
    return flatten_multivariate_window(window)


def _preprocess_window(window: np.ndarray, ds_cfg: dict[str, Any]) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    prenormalize = ds_cfg.get("prenormalize", "per_window")
    if prenormalize == "per_window":
        if window.ndim == 1:
            window = window[np.newaxis, :]
        return zscore_window_channels(window)
    if prenormalize in (None, "none"):
        if window.ndim == 1:
            return window[np.newaxis, :]
        return window
    raise ValueError(f"Unknown prenormalize mode: {prenormalize}. Use none or per_window.")


def _windows_to_meta_signals(windows: list[np.ndarray], ds_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> np.ndarray:
    signal_length = int(model_cfg.get("signal_length", ds_cfg.get("pad_to_length", 2048)))
    merge_mode = str(ds_cfg.get("merge_mode", "flatten"))
    signals = []
    for window in windows:
        processed = _preprocess_window(window, ds_cfg)
        merged = _merge_window(processed, merge_mode)
        signals.append(_pad_or_truncate_signal(merged, signal_length))
    return np.asarray(signals, dtype=np.float32)


def _limit_samples_per_class(
    windows: list[np.ndarray],
    labels: list[int],
    samples_per_class: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected_windows: list[np.ndarray] = []
    selected_labels: list[int] = []

    for class_id in sorted(np.unique(labels_arr)):
        class_indices = np.where(labels_arr == class_id)[0]
        if len(class_indices) > samples_per_class:
            class_indices = rng.choice(class_indices, size=samples_per_class, replace=False)
        for idx in class_indices:
            selected_windows.append(windows[int(idx)])
            selected_labels.append(int(class_id))

    return selected_windows, np.asarray(selected_labels, dtype=np.int64)


def _collect_cooler_windows(ds_cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[int], list[str]]:
    root = resolve_path(ds_cfg.get("root", "data/cooler"))
    head_csv = root / ds_cfg.get("head_csv", "cooler_simulation_results/all_head_30.csv")
    tail_csv = root / ds_cfg.get("tail_csv", "cooler_simulation_results/all_tail_30.csv")
    value_columns = resolve_value_columns(ds_cfg)
    group_col = ds_cfg.get("group_column", "group_id")
    cycle_col = ds_cfg.get("cycle_column", "work_cycle")
    sample_length = resolve_sample_length(ds_cfg, default=30)
    label_names = list(ds_cfg.get("label_names", ["normal", "degraded"]))

    windows: list[np.ndarray] = []
    labels: list[int] = []

    for label_id, csv_path in enumerate([head_csv, tail_csv]):
        df = pd.read_csv(csv_path)
        for column in value_columns:
            if column not in df.columns:
                raise KeyError(f"Column '{column}' not found in {csv_path}")
        for _, group in df.groupby(group_col):
            group = group.sort_values(cycle_col)
            if len(group) != sample_length:
                continue
            window = np.stack([group[column].values.astype(np.float32) for column in value_columns], axis=0)
            windows.append(window)
            labels.append(label_id)

    if not windows:
        raise ValueError(f"No cooler windows created (sample_length={sample_length}).")
    return windows, labels, label_names


def _collect_multivariate_file_windows(ds_cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[int], list[str]]:
    root = resolve_path(ds_cfg.get("root"))
    value_columns = resolve_value_columns(ds_cfg)
    sample_length = resolve_sample_length(ds_cfg)
    stride = resolve_stride(ds_cfg)
    class_files = load_class_files(ds_cfg, root)
    region_fraction = float(ds_cfg.get("region_fraction", 1.0))
    use_regions = bool(ds_cfg.get("class_window_regions"))

    windows, labels, _, label_names = collect_multivariate_windows(
        root,
        class_files,
        value_columns,
        sample_length,
        stride,
        unit_col=ds_cfg.get("unit_column", "unit"),
        cycle_col=ds_cfg.get("cycle_column", "cycle"),
        region_resolver=(lambda name: resolve_class_window_region(name, ds_cfg)) if use_regions else None,
        region_fraction=region_fraction,
        whole_file_as_series=bool(ds_cfg.get("whole_file_as_series", False)),
    )
    if not windows:
        raise ValueError(
            f"No windows created for {ds_cfg.get('name')} "
            f"(sample_length={sample_length}, stride={stride})."
        )
    return windows, labels, label_names


def _collect_sifuqi_windows(ds_cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[int], list[str]]:
    ds_cfg = {
        **ds_cfg,
        "class_files": ds_cfg.get(
            "levels",
            {
                "normal": "servo_normal.csv",
                "mild": "servo_mild.csv",
                "moderate": "servo_moderate.csv",
                "severe": "servo_severe.csv",
            },
        ),
        "whole_file_as_series": bool(ds_cfg.get("whole_file_as_series", True)),
    }
    return _collect_multivariate_file_windows(ds_cfg)


def _collect_raw_meta_windows(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[int], list[str]]:
    ds_cfg = cfg["dataset"]
    name = ds_cfg.get("name", "cwru_csv")

    if name == "cooler":
        return _collect_cooler_windows(ds_cfg)
    if name == "sensitivity":
        return _collect_multivariate_file_windows(ds_cfg)
    if name in ("image_quality", "degradation"):
        return _collect_multivariate_file_windows(ds_cfg)
    if name == "sifuqi":
        return _collect_sifuqi_windows(ds_cfg)
    raise ValueError(
        f"Unsupported tl_meta dataset: {name}. "
        "Use cwru_csv, cooler, sensitivity, image_quality, or sifuqi."
    )


def _load_meta_signal_dataset(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    seed = int(cfg.get("experiment", {}).get("seed", 42))

    windows, labels, label_names = _collect_raw_meta_windows(cfg)
    samples_per_class = int(model_cfg.get("samples_per_class", 200))
    windows, labels = _limit_samples_per_class(windows, labels, samples_per_class, seed=seed)
    signals = _windows_to_meta_signals(windows, ds_cfg, model_cfg)
    if bool(model_cfg.get("normalize", True)):
        signals = _l2_normalize_signals(signals)
    return signals, labels, label_names


def _split_meta_indices(labels: np.ndarray, cfg: dict[str, Any], model_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    indices = np.arange(len(labels))
    train_ratio = float(model_cfg.get("train_ratio", 0.7))
    val_ratio = float(model_cfg.get("val_ratio", 0.15))
    stratify = labels if len(np.unique(labels)) > 1 else None

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=stratify,
    )
    temp_labels = labels[temp_idx]
    temp_stratify = temp_labels if len(np.unique(temp_labels)) > 1 else None
    val_size = val_ratio / (1.0 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=1.0 - val_size,
        random_state=seed,
        stratify=temp_stratify,
    )
    return train_idx, val_idx, test_idx


def _validate_episode_budget(dataset: TLMetaSignalDataset, model_cfg: dict[str, Any]) -> None:
    n_support = int(model_cfg.get("n_support", 5))
    n_unlabeled = int(model_cfg.get("n_unlabeled", 5))
    n_query = int(model_cfg.get("n_query", 5))
    per_class_need = n_support + n_unlabeled + n_query
    allow_replace = bool(model_cfg.get("allow_episode_replace", False))
    counts = np.bincount(dataset.labels, minlength=len(dataset.class_names))
    too_small = [
        (dataset.class_names[i], int(counts[i]))
        for i in range(len(dataset.class_names))
        if counts[i] < per_class_need and not allow_replace
    ]
    if too_small:
        details = ", ".join(f"{name}={count}<{per_class_need}" for name, count in too_small)
        raise ValueError(
            f"Not enough samples per class for episode sampling ({details}). "
            "Reduce n_support/n_unlabeled/n_query, increase samples_per_class/window settings, "
            "or set allow_episode_replace: true for small datasets."
        )


def _episode_sampler_kwargs(model_cfg: dict[str, Any]) -> dict[str, bool]:
    return {"allow_episode_replace": bool(model_cfg.get("allow_episode_replace", False))}


def _build_cwru_datasets(cfg: dict[str, Any]):
    ds_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    csv_path = str(resolve_path(ds_cfg.get("csv", "CWRU_12k_1797_10c.csv")))
    seed = int(cfg.get("experiment", {}).get("seed", 42))

    full_dataset = CWRUDataset(
        csv_path,
        samples_per_class=int(model_cfg.get("samples_per_class", 200)),
        signal_length=int(model_cfg.get("signal_length", 2048)),
        overlap=float(model_cfg.get("overlap", 0.5)),
        normalize=bool(model_cfg.get("normalize", True)),
    )

    indices = np.arange(len(full_dataset))
    labels = full_dataset.labels
    train_ratio = float(model_cfg.get("train_ratio", 0.7))
    val_ratio = float(model_cfg.get("val_ratio", 0.15))

    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, random_state=seed, stratify=labels
    )
    temp_labels = labels[temp_idx]
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=1 - val_size, random_state=seed, stratify=temp_labels
    )

    dataset_kwargs = {
        "csv_path": csv_path,
        "samples_per_class": int(model_cfg.get("samples_per_class", 200)),
        "signal_length": int(model_cfg.get("signal_length", 2048)),
        "overlap": float(model_cfg.get("overlap", 0.5)),
        "normalize": bool(model_cfg.get("normalize", True)),
    }
    train_dataset = CWRUDataset(**dataset_kwargs, indices=train_idx)
    val_dataset = CWRUDataset(**dataset_kwargs, indices=val_idx)
    test_dataset = CWRUDataset(**dataset_kwargs, indices=test_idx)
    return train_dataset, val_dataset, test_dataset


def _build_multivariate_meta_datasets(cfg: dict[str, Any]):
    model_cfg = cfg["model"]
    signals, labels, label_names = _load_meta_signal_dataset(cfg)
    train_idx, val_idx, test_idx = _split_meta_indices(labels, cfg, model_cfg)

    train_dataset = TLMetaSignalDataset(signals, labels, label_names, indices=train_idx)
    val_dataset = TLMetaSignalDataset(signals, labels, label_names, indices=val_idx)
    test_dataset = TLMetaSignalDataset(signals, labels, label_names, indices=test_idx)
    return train_dataset, val_dataset, test_dataset


def _build_datasets(cfg: dict[str, Any]):
    ds_cfg = cfg["dataset"]
    name = ds_cfg.get("name", "cwru_csv")
    if name in ("cwru_csv", "cwru"):
        return _build_cwru_datasets(cfg)
    return _build_multivariate_meta_datasets(cfg)


def train(bundle: DiagnosisDataBundle | None, cfg: dict[str, Any], out_dir: Path):
    device = get_device(cfg)
    model_cfg = cfg["model"]

    train_dataset, val_dataset, test_dataset = _build_datasets(cfg)

    n_way = int(model_cfg.get("n_way", 10))
    num_classes = len(test_dataset.class_names)
    if n_way != num_classes:
        raise ValueError(f"model.n_way={n_way} but dataset has {num_classes} classes: {test_dataset.class_names}")

    for split_name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        try:
            _validate_episode_budget(dataset, model_cfg)
        except ValueError as exc:
            raise ValueError(f"{split_name} split: {exc}") from exc

    sampler_kwargs = _episode_sampler_kwargs(model_cfg)
    embedding_dim = int(model_cfg.get("embedding_dim", 128))
    refine_iterations = int(model_cfg.get("refine_iterations", 2))

    encoder = Encoder(embedding_dim=embedding_dim).to(device)
    model = SSMN(
        encoder,
        n_way=n_way,
        embedding_dim=embedding_dim,
        refine_iterations=refine_iterations,
    ).to(device)

    history, best_state, best_episode, best_val_acc = train_ssmn(
        model,
        train_dataset,
        val_dataset,
        device,
        n_way=n_way,
        n_support=int(model_cfg.get("n_support", 5)),
        n_unlabeled=int(model_cfg.get("n_unlabeled", 5)),
        n_query=int(model_cfg.get("n_query", 5)),
        n_episodes=int(model_cfg.get("n_episodes", 2000)),
        val_interval=int(model_cfg.get("val_interval", 200)),
        l_skip=float(model_cfg.get("l_skip", 0.2)),
        sgd_lr=float(model_cfg.get("sgd_lr", 0.1)),
        adam_lr=float(model_cfg.get("adam_lr", 0.001)),
        **sampler_kwargs,
    )

    if best_state is not None:
        save_checkpoint_best(
            out_dir,
            {
                "model_state_dict": best_state,
                "label_names": test_dataset.class_names[:n_way],
                "n_way": n_way,
            },
            epoch=best_episode,
            best_metric=float(best_val_acc),
            metric_name="val_accuracy_percent",
        )
        model.load_state_dict(best_state)

    val_interval = int(model_cfg.get("val_interval", 200))
    plot_training_curves(history["train_loss"], history["val_accuracy"], val_interval, out_dir / "loss_curves.png")
    save_loss_history(history, out_dir)

    meta = {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "label_names": test_dataset.class_names,
        "n_way": n_way,
        "best_episode": best_episode,
        "best_val_accuracy": float(best_val_acc),
    }
    return model, meta


def load_checkpoint(path: Path, cfg: dict[str, Any]):
    device = get_device(cfg)
    model_cfg = cfg["model"]
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    n_way = int(checkpoint.get("n_way", model_cfg.get("n_way", 10)))
    embedding_dim = int(model_cfg.get("embedding_dim", 128))
    refine_iterations = int(model_cfg.get("refine_iterations", 2))

    encoder = Encoder(embedding_dim=embedding_dim).to(device)
    model = SSMN(encoder, n_way=n_way, embedding_dim=embedding_dim, refine_iterations=refine_iterations).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate(model, bundle: DiagnosisDataBundle | None, cfg: dict[str, Any], out_dir: Path, split: str = "test", meta: dict | None = None) -> dict:
    device = get_device(cfg)
    model = model.to(device)
    model_cfg = cfg["model"]

    if meta is None:
        _, _, test_dataset = _build_datasets(cfg)
        n_way = int(model_cfg.get("n_way", 10))
        label_names = test_dataset.class_names[:n_way]
    else:
        test_dataset = meta["test_dataset"]
        n_way = meta.get("n_way", int(model_cfg.get("n_way", 10)))
        label_names = meta.get("label_names", test_dataset.class_names)[:n_way]

    test_acc, predictions, true_labels = evaluate_ssmn(
        model,
        test_dataset,
        device,
        n_way=n_way,
        n_support=int(model_cfg.get("n_support", 5)),
        n_unlabeled=int(model_cfg.get("n_unlabeled", 5)),
        n_query=int(model_cfg.get("n_query", 5)),
        n_episodes=int(model_cfg.get("test_episodes", 300)),
        **_episode_sampler_kwargs(model_cfg),
    )

    metrics = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "dataset": cfg["dataset"]["name"],
        "model": "tl_meta",
        "split": split,
        "accuracy_percent": float(test_acc),
        "accuracy": float(test_acc / 100.0),
        "precision_macro": float(precision_score(true_labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true_labels, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(true_labels, predictions, output_dict=True),
        "label_names": label_names,
    }

    cm = plot_confusion_matrix(
        true_labels,
        predictions,
        label_names,
        out_dir / "confusion_matrix.png",
        title=f"SSMN Confusion Matrix (Accuracy: {test_acc:.2f}%)",
    )
    metrics["confusion_matrix"] = cm.tolist()
    save_json(metrics, out_dir / "test_metrics.json")
    return metrics
