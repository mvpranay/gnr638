import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################
# DATA TRANSFORMS
#########################################

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

#########################################
# DATASET
#########################################

train_dir = "train_data"

# Two dataset objects with different transforms, same underlying data
train_full = datasets.ImageFolder(train_dir, transform=train_transform)
val_full   = datasets.ImageFolder(train_dir, transform=val_transform)

# Split sizes
train_size = int(0.7 * len(train_full))
val_size   = len(train_full) - train_size

# Same seed = same indices for both splits
torch.manual_seed(42)
train_dataset, _ = torch.utils.data.random_split(train_full, [train_size, val_size])

torch.manual_seed(42)
_, val_dataset   = torch.utils.data.random_split(val_full, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=2, persistent_workers=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,
                          num_workers=2, persistent_workers=True)

num_classes = len(train_full.classes)

#########################################
# MODEL: RESNET50
#########################################

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)

#########################################
# FREEZE BACKBONE (Linear Probe Scenario)
#########################################

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

# Uncomment for linear probe
# freeze_backbone(model)

model = model.to(device)

#########################################
# LOSS + OPTIMIZER
#########################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

#########################################
# TRAINING LOOP
#########################################

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar with current metrics
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{correct/total:.4f}'})

    acc = correct / total
    return total_loss/len(loader), acc


#########################################
# VALIDATION LOOP
#########################################

def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs,1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc

#########################################
# MAIN TRAINING
#########################################

epochs = 30

print(f"Starting training on {device}")
print(f"Dataset: {len(full_dataset)} total images, {num_classes} classes")
print(f"Train set: {len(train_dataset)} images")
print(f"Validation set: {len(val_dataset)} images")
print(f"Batches per epoch (train): {len(train_loader)}")
print(f"Batches per epoch (val): {len(val_loader)}")
print("="*60)

for epoch in range(epochs):

    train_loss, train_acc = train_epoch(model, train_loader)
    val_acc = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc : {train_acc:.4f}")
    print(f"Val Acc   : {val_acc:.4f}")
    print("-"*40)

#########################################
# SAVE MODEL
#########################################

torch.save(model.state_dict(), "resnet50_aid.pth")
