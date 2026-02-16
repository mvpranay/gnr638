import os
import time
import gc
import APDNN
from data_loader2 import ImageFolderDataset

# Set random seed for reproducibility
APDNN.set_random_seed(42)

print("=" * 60)
print("CNN Training with Convolutional Neural Network")
print("=" * 60)

def conv2d_macs(h_out, w_out, c_out, k_h, k_w, c_in, include_bias=False):
    macs = h_out * w_out * c_out * (k_h * k_w * c_in)
    if include_bias:
        macs += h_out * w_out * c_out
    return macs

def linear_macs(n_in, n_out, include_bias=False):
    macs = n_in * n_out
    if include_bias:
        macs += n_out
    return macs

def macs_to_flops(macs):
    return macs * 2

# Load datasets and build label mapping
print("\nScanning data_2 for class labels...")
class_names = sorted([d for d in os.listdir("data_2") if os.path.isdir(os.path.join("data_2", d))])
num_classes = len(class_names)
label_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_label = {idx: name for name, idx in label_to_idx.items()}

print(f"Found {num_classes} classes")
print(f"Sample classes: {class_names[:5]}...")

# Load datasets (100 classes with string labels)
print("\nLoading training data...")
train_dataset = ImageFolderDataset(
    data_dir="data_2",
    label_to_idx=label_to_idx,
    batch_size=32,
    shuffle=True,
    augment=True  # Enable augmentation
)

print("\nBuilding CNN model...")

# Model architecture for grayscale images (1 channel input)
# Increased capacity for 100 classes
conv1 = APDNN.Conv2D(1, 16, 3, stride=1, padding=1)  # 1x32x32 -> 16x32x32
conv2 = APDNN.Conv2D(16, 32, 3, stride=1, padding=1)  # 16x16x16 -> 32x16x16

# After two maxpool2d: 32x32 -> 16x16 -> 8x8
# 32 channels * 8 * 8 = 2048 features
fc1 = APDNN.Linear(32 * 8 * 8, 100)  # 100 classes

# MACs/FLOPs estimate per image (forward only)
conv1_macs = conv2d_macs(32, 32, 16, 3, 3, 1, include_bias=False)
conv2_macs = conv2d_macs(16, 16, 32, 3, 3, 16, include_bias=False)
fc1_macs = linear_macs(32 * 8 * 8, 100, include_bias=False)
total_macs = conv1_macs + conv2_macs + fc1_macs
total_flops = macs_to_flops(total_macs)

print("Model compute (per image, forward only):")
print(f"  Conv1 MACs: {conv1_macs:,}")
print(f"  Conv2 MACs: {conv2_macs:,}")
print(f"  FC1 MACs:   {fc1_macs:,}")
print(f"  Total MACs: {total_macs:,}")
print(f"  Total FLOPs (2 FLOPs = 1 MAC): {total_flops:,}\n")

params = conv1.parameters() + conv2.parameters() + fc1.parameters()
num_params = conv1.param_count() + conv2.param_count() + fc1.param_count()
optimizer = APDNN.SGD(params, 0.01)

print(f"Model built with {num_params} parameters\n")

# Training loop
num_epochs = 10

print("=" * 60)
print("Training")
print("=" * 60)

train_start = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_loss = 0.0
    batch_count = 0
    
    for batch_tensor, batch_labels in train_dataset:
        # batch_tensor is [batch_size, 1, 32, 32] (grayscale)
        actual_batch_size = len(batch_labels)
        
        optimizer.zero_grad()
        
        # Forward pass through CNN
        c1 = conv1.forward(batch_tensor)
        a1 = APDNN.relu(c1)
        p1 = APDNN.maxpool2d(a1, pool_size=2, stride=2)  # 32x32 -> 16x16
        
        c2 = conv2.forward(p1)
        a2 = APDNN.relu(c2)
        p2 = APDNN.maxpool2d(a2, pool_size=2, stride=2)  # 16x16 -> 8x8
        
        # Flatten: [batch_size, 32, 8, 8] -> [batch_size, 2048]
        flat = p2.view([actual_batch_size, 32 * 8 * 8])
        
        # Output logits
        logits = fc1.forward(flat)
        
        # Compute loss using cross-entropy
        loss = APDNN.cross_entropy_loss(logits, batch_labels)
        epoch_loss += loss.data[0]
        
        loss.backward()
        optimizer.step()
        
        # Clean up intermediate tensors
        del c1, a1, p1, c2, a2, p2, flat, logits, loss, batch_tensor
        
        batch_count += 1
        if batch_count % 50 == 0:
            print(f"  Batches done: {batch_count}")
    
    avg_loss = epoch_loss / batch_count
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # Force garbage collection after each epoch
    gc.collect()

train_time = time.time() - train_start
print(f"\nTraining completed in {train_time:.2f} seconds")

# Evaluation
print("\n" + "=" * 60)
print("Evaluation on Test Set")
print("=" * 60)

eval_start = time.time()

print("\nLoading test data...")
test_dataset = ImageFolderDataset(
    data_dir="data_2",
    label_to_idx=label_to_idx,
    batch_size=64,
    shuffle=False,
    augment=False
)

correct = 0
total = 0

for batch_tensor, batch_labels in test_dataset:
    actual_batch_size = len(batch_labels)
    
    # Forward pass through CNN
    c1 = conv1.forward(batch_tensor)
    a1 = APDNN.relu(c1)
    p1 = APDNN.maxpool2d(a1, pool_size=2, stride=2)  # 32x32 -> 16x16
    
    c2 = conv2.forward(p1)
    a2 = APDNN.relu(c2)
    p2 = APDNN.maxpool2d(a2, pool_size=2, stride=2)  # 16x16 -> 8x8
    
    # Flatten and output
    flat = p2.view([actual_batch_size, 32 * 8 * 8])
    logits = fc1.forward(flat)
    
    # Predict for each sample in batch (efficient argmax)
    for i in range(actual_batch_size):
        max_idx = 0
        max_val = logits.data[i * 100]
        for c in range(1, 100):
            if logits.data[i * 100 + c] > max_val:
                max_val = logits.data[i * 100 + c]
                max_idx = c
        if max_idx == batch_labels[i]:
            correct += 1
        total += 1
    
    # Clean up
    del c1, a1, p1, c2, a2, p2, flat, logits, batch_tensor

eval_time = time.time() - eval_start
accuracy = correct / total * 100 if total > 0 else 0

print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"Evaluation Time: {eval_time:.2f} seconds")

# Force garbage collection
gc.collect()

print("=" * 60)
print(f"Total Training Time: {train_time:.2f}s")
print(f"Total Evaluation Time: {eval_time:.2f}s")
print("=" * 60)
