import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


class ConvNeXtTinyFewShotLearningAnalysis:
    def __init__(self, seed=42, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.seed = seed
        self.device = device
        self.dataset_path = self._resolve_dataset_path()
        self.set_seed()

    def _resolve_dataset_path(self):
        """Resolve dataset path (Kaggle path first, local fallback)."""
        candidate_paths = [
            Path('/kaggle/input/datasets/avinashc5/train-data/train_data/'),
        ]

        for path in candidate_paths:
            if path.exists() and path.is_dir():
                return path

        raise FileNotFoundError(
            "Could not find dataset folder. Expected one of: "
            f"{', '.join(str(p) for p in candidate_paths)}"
        )

    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def get_data_loaders(self, data_percentage=100, batch_size=32, val_split=0.2):
        """Create data loaders with specified percentage of training data."""
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        full_dataset = datasets.ImageFolder(root=str(self.dataset_path), transform=transform)
        self.num_classes = len(full_dataset.classes)

        indices = np.random.permutation(len(full_dataset))
        val_size = int(len(full_dataset) * val_split)
        val_indices = indices[:val_size]
        train_indices_full = indices[val_size:]

        train_size = max(1, int(len(train_indices_full) * (data_percentage / 100)))
        train_indices = train_indices_full[:train_size]

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="Train", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(train_loader), correct / total

    def validate(self, model, val_loader, criterion):
        """Validate the model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Val", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(val_loader), correct / total

    def train_convnext_tiny(self, data_percentage=100, epochs=20, lr=0.001):
        """Train ConvNeXt-Tiny with specified data percentage."""
        train_loader, val_loader = self.get_data_loaders(data_percentage)

        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        results = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }

        for epoch in tqdm(range(epochs), desc=f"Epochs ({data_percentage}%)"):
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(model, val_loader, criterion)

            results['train_acc'].append(train_acc)
            results['val_acc'].append(val_acc)
            results['train_loss'].append(train_loss)
            results['val_loss'].append(val_loss)

            tqdm.write(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        return model, results

    def analyze_few_shot_learning(self):
        """Run analysis across different data regimes."""
        percentages = [100, 20, 5]
        analysis_results = {}

        for pct in tqdm(percentages, desc="Few-shot regimes"):
            print(f"\nTraining with {pct}% data...")
            model, results = self.train_convnext_tiny(data_percentage=pct)

            final_train_acc = results['train_acc'][-1]
            final_val_acc = results['val_acc'][-1]
            train_val_gap = final_train_acc - final_val_acc

            analysis_results[pct] = {
                'val_accuracy': final_val_acc,
                'train_accuracy': final_train_acc,
                'train_val_gap': train_val_gap,
                'history': results
            }

            print(f"Validation Accuracy: {final_val_acc:.4f}")
            print(f"Train-Val Gap: {train_val_gap:.4f}")

        acc_100 = analysis_results[100]['val_accuracy']
        acc_5 = analysis_results[5]['val_accuracy']
        delta = (acc_100 - acc_5) / acc_100

        print("\n" + "=" * 50)
        print("FEW-SHOT LEARNING ANALYSIS SUMMARY")
        print("=" * 50)
        for pct in percentages:
            print(f"\n{pct}% Data Regime:")
            print(f"  Validation Accuracy: {analysis_results[pct]['val_accuracy']:.4f}")
            print(f"  Training Accuracy: {analysis_results[pct]['train_accuracy']:.4f}")
            print(f"  Train-Val Gap: {analysis_results[pct]['train_val_gap']:.4f}")

        print(f"\nRelative Performance Drop (Δ): {delta:.4f}")
        print(f"Performance degradation from 100% to 5%: {delta * 100:.2f}%")

        return analysis_results, delta


analyzer = ConvNeXtTinyFewShotLearningAnalysis(seed=42)
results, delta = analyzer.analyze_few_shot_learning()
