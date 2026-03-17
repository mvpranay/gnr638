import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
CHECKPOINT_PATH = "efficientnet_b0_aid.pth"
DATA_DIR = "/kaggle/input/datasets/avinashc5/train-data/train_data"
BATCH_SIZE = 32
NUM_WORKERS = 2
VAL_SPLIT = 0.30
SEED = 42

SAVE_CM = "confusion_matrix.png"
SAVE_PCA = "embeddings_pca.png"
SAVE_TSNE = "embeddings_tsne.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# DATA
# ----------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
num_classes = len(full_dataset.classes)

val_size = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size

generator = torch.Generator().manual_seed(SEED)
_, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(NUM_WORKERS > 0),
)

# ----------------------------
# MODEL LOADING
# ----------------------------
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ----------------------------
# UTILS
# ----------------------------
def get_predictions(model, loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predicting", leave=False):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix_percent(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_percent = np.nan_to_num(cm_percent) * 100
    cm_percent_int = np.rint(cm_percent).astype(int)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_percent_int,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "percentage of samples"},
    )
    plt.title("Confusion Matrix - Validation Set (%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._features = None
        # EfficientNet's classifier is a Sequential with Dropout and Linear
        # Hook the pre-call to classifier to capture the 1280-d input
        self._hook_handle = self.model.classifier[0].register_forward_pre_hook(
            self._save_features
        )

    def _save_features(self, module, input):
        # input is a tuple; input[0] is the flattened feature vector (N, 1280)
        self._features = input[0]

    def forward(self, x):
        self.model(x)
        return self._features

    def remove_hook(self):
        self._hook_handle.remove()


def get_embeddings(model, loader):
    extractor = FeatureExtractor(model).to(device)
    extractor.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="Extracting embeddings", leave=False):
            images = images.to(device)
            feats = extractor(images)
            embeddings.extend(feats.cpu().numpy())
            labels.extend(batch_labels.numpy())

    extractor.remove_hook()
    return np.array(embeddings), np.array(labels)


def plot_embeddings_pca(embeddings, labels, class_names, save_path):
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))

    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            label=class_name,
            alpha=0.6,
            s=20,
            color=colors[i],
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.title("Feature Embeddings - PCA")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_embeddings_tsne(embeddings, labels, class_names, save_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))

    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            label=class_name,
            alpha=0.6,
            s=20,
            color=colors[i],
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Feature Embeddings - t-SNE")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


print(f"Using device: {device}")
print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
print(f"Validation samples: {len(val_dataset)}")

preds, labels = get_predictions(model, val_loader)
plot_confusion_matrix_percent(labels, preds, full_dataset.classes, SAVE_CM)

embeddings, emb_labels = get_embeddings(model, val_loader)
print(f"Embeddings shape: {embeddings.shape}")

plot_embeddings_pca(embeddings, emb_labels, full_dataset.classes, SAVE_PCA)
plot_embeddings_tsne(embeddings, emb_labels, full_dataset.classes, SAVE_TSNE)

print("Done.")
