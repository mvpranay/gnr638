"""
Layer-Wise Feature Probing with EfficientNet-B0

What this script does:
1) Extracts intermediate features from selected depths (early/middle/final).
2) Trains separate linear classifiers (logistic regression) on each depth.
3) Reports/plots:
   - Validation accuracy vs depth
   - Feature norm statistics across depths
   - PCA-2D visualization across depths on a fixed subset

Layer selection (clearly documented):
- early_features_2  -> output of `model.features[2]` (early MBConv features)
- middle_features_4 -> output of `model.features[4]` (mid-level abstraction)
- final_features_8  -> output of `model.features[8]` (deep semantic features)

For all selected layers, feature maps are global-average-pooled to a vector.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from tqdm import tqdm


# ==============================
# Global configuration (edit here)
# ==============================
DATA_ROOT = "/kaggle/input/datasets/avinashc5/train-data/train_data"

VAL_SPLIT = 0.2
BATCH_SIZE = 64
NUM_WORKERS = 4
SEED = 42

# Fixed subset for PCA (same samples across layers)
SUBSET_CLASSES = 30
SAMPLES_PER_CLASS = 30


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(data_root: str, seed: int, val_split: float):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    full_dataset = datasets.ImageFolder(root=data_root, transform=transform)
    indices = np.arange(len(full_dataset))
    targets = np.array(full_dataset.targets)

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=targets,
    )

    train_dataset = Subset(full_dataset, train_idx.tolist())
    val_dataset = Subset(full_dataset, val_idx.tolist())
    return full_dataset, train_dataset, val_dataset


def build_dataloaders(train_dataset, val_dataset, batch_size: int, num_workers: int):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def create_feature_extractor(device: torch.device):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.eval().to(device)

    # Selected depths for probing (documented requirement)
    selected_layers = {
        "early_features_2": model.features[2],
        "middle_features_4": model.features[4],
        "final_features_8": model.features[8],
    }

    activations: Dict[str, torch.Tensor] = {}
    hooks = []

    for layer_name, layer_module in selected_layers.items():
        def _make_hook(name):
            def _hook(_module, _inputs, output):
                activations[name] = output.detach()

            return _hook

        hooks.append(layer_module.register_forward_hook(_make_hook(layer_name)))

    return model, selected_layers, activations, hooks


@torch.no_grad()
def extract_features(
    model: nn.Module,
    data_loader: DataLoader,
    activations: Dict[str, torch.Tensor],
    layer_names: List[str],
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    features = {name: [] for name in layer_names}
    labels = []

    for images, y in tqdm(data_loader, desc="Extracting features", leave=False):
        images = images.to(device, non_blocking=True)

        _ = model(images)

        for name in layer_names:
            fmap = activations[name]
            vec = F.adaptive_avg_pool2d(fmap, output_size=1).flatten(1)
            features[name].append(vec.cpu().numpy())

        labels.append(y.numpy())

    features_np = {k: np.concatenate(v, axis=0) for k, v in features.items()}
    labels_np = np.concatenate(labels, axis=0)
    return features_np, labels_np


def train_linear_probes(
    train_features: Dict[str, np.ndarray],
    train_labels: np.ndarray,
    val_features: Dict[str, np.ndarray],
    val_labels: np.ndarray,
) -> Dict[str, float]:
    val_acc = {}

    for layer_name in train_features.keys():
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                n_jobs=-1,
            ),
        )
        clf.fit(train_features[layer_name], train_labels)
        preds = clf.predict(val_features[layer_name])
        acc = accuracy_score(val_labels, preds)
        val_acc[layer_name] = float(acc)
        print(f"[Linear Probe] {layer_name}: val_acc={acc:.4f}")

    return val_acc


def compute_feature_norm_stats(features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for layer_name, x in features.items():
        norms = np.linalg.norm(x, ord=2, axis=1)
        stats[layer_name] = {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "min": float(norms.min()),
            "max": float(norms.max()),
        }
    return stats


def select_fixed_subset_indices(
    full_dataset: datasets.ImageFolder,
    num_classes: int,
    samples_per_class: int,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(seed)
    targets = np.array(full_dataset.targets)
    unique_classes = np.unique(targets)

    if len(unique_classes) < num_classes:
        raise ValueError(
            f"Requested {num_classes} classes, but dataset has only {len(unique_classes)} classes."
        )

    chosen_classes = unique_classes[:num_classes]
    subset_indices = []

    for cls in chosen_classes:
        cls_indices = np.where(targets == cls)[0]
        if len(cls_indices) < samples_per_class:
            raise ValueError(
                f"Class id {cls} has {len(cls_indices)} samples, but {samples_per_class} required."
            )
        picked = rng.choice(cls_indices, size=samples_per_class, replace=False)
        subset_indices.extend(picked.tolist())

    return subset_indices


def plot_accuracy_vs_depth(val_acc: Dict[str, float]) -> None:
    layer_order = ["early_features_2", "middle_features_4", "final_features_8"]
    x = np.arange(len(layer_order))
    y = [val_acc[k] for k in layer_order]

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xticks(x, layer_order, rotation=15)
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Network Depth")
    plt.title("Validation Accuracy vs Depth (EfficientNet-B0 Linear Probes)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("val_accuracy_vs_depth.png", dpi=200)
    plt.close()


def plot_feature_norm_stats(norm_stats: Dict[str, Dict[str, float]]) -> None:
    layer_order = ["early_features_2", "middle_features_4", "final_features_8"]
    means = [norm_stats[k]["mean"] for k in layer_order]
    stds = [norm_stats[k]["std"] for k in layer_order]

    plt.figure(figsize=(7, 4))
    plt.bar(layer_order, means, yerr=stds, capsize=5)
    plt.ylabel("L2 Norm (mean ± std)")
    plt.xlabel("Layer")
    plt.title("Feature Norm Statistics Across Layers")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_norm_stats.png", dpi=200)
    plt.close()


def plot_pca_across_layers(
    subset_features: Dict[str, np.ndarray],
    subset_labels: np.ndarray,
) -> None:
    layer_order = ["early_features_2", "middle_features_4", "final_features_8"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)

    for ax, layer_name in zip(axes, layer_order):
        x = subset_features[layer_name]
        x = StandardScaler().fit_transform(x)
        x2 = PCA(n_components=2, random_state=42).fit_transform(x)

        scatter = ax.scatter(
            x2[:, 0],
            x2[:, 1],
            c=subset_labels,
            cmap="tab20",
            s=8,
            alpha=0.75,
        )
        ax.set_title(f"PCA 2D: {layer_name}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)

    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Class Index")
    fig.suptitle("PCA 2D on Fixed Subset (same samples across layers)", y=1.02)
    fig.tight_layout()
    fig.savefig("pca_2d_across_layers.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

full_dataset, train_dataset, val_dataset = build_datasets(
    data_root=DATA_ROOT,
    seed=SEED,
    val_split=VAL_SPLIT,
)

train_loader, val_loader = build_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

model, selected_layers, activations, hooks = create_feature_extractor(device)
layer_names = list(selected_layers.keys())

print("Selected probing layers:")
for k, v in selected_layers.items():
    print(f"  - {k}: {v.__class__.__name__}")

train_features, train_labels = extract_features(
    model, train_loader, activations, layer_names, device
)
val_features, val_labels = extract_features(
    model, val_loader, activations, layer_names, device
)

val_acc = train_linear_probes(
    train_features=train_features,
    train_labels=train_labels,
    val_features=val_features,
    val_labels=val_labels,
)

norm_stats = compute_feature_norm_stats(val_features)

subset_indices = select_fixed_subset_indices(
    full_dataset=full_dataset,
    num_classes=SUBSET_CLASSES,
    samples_per_class=SAMPLES_PER_CLASS,
    seed=SEED,
)
subset_dataset = Subset(full_dataset, subset_indices)
subset_loader = DataLoader(
    subset_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)
subset_features, subset_labels = extract_features(
    model, subset_loader, activations, layer_names, device
)

with open("val_accuracy.json", "w", encoding="utf-8") as f:
    json.dump(val_acc, f, indent=2)

with open("feature_norm_stats.json", "w", encoding="utf-8") as f:
    json.dump(norm_stats, f, indent=2)

plot_accuracy_vs_depth(val_acc)
plot_feature_norm_stats(norm_stats)
plot_pca_across_layers(subset_features, subset_labels)

for h in hooks:
    h.remove()

print("\nDone.")