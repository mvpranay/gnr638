import os
import time
import gc
import APDNN
from data_loader import ImageFolderDataset
import sys

# take data_dir as command line argument
if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    print("Usage: python cnn_model2.py <data_dir>")
    sys.exit(1)

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
print(f"\nScanning {data_dir} for class labels...")
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
num_classes = len(class_names)
label_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_label = {idx: name for name, idx in label_to_idx.items()}

print(f"Found {num_classes} classes")
print(f"Sample classes: {class_names[:5]}...")

# Load datasets (100 classes with string labels)
print("\nLoading training data...")
train_dataset = ImageFolderDataset(
    data_dir=data_dir,
    label_to_idx=label_to_idx,
    batch_size=32,
    shuffle=True,
    augment=True  # Enable augmentation
)

print("\nBuilding Efficient 4-Block CNN model (8-16-32-64)...")

# Input: 1x32x32
# Block 1: 32x32 -> 16x16 (8 channels)
conv1 = APDNN.Conv2D(1, 8, 3, stride=1, padding=1)  
# Block 2: 16x16 -> 8x8 (16 channels)
conv2 = APDNN.Conv2D(8, 16, 3, stride=1, padding=1) 
# Block 3: 8x8 -> 4x4 (32 channels)
conv3 = APDNN.Conv2D(16, 32, 3, stride=1, padding=1)
# Block 4: 4x4 -> 2x2 (64 channels)
conv4 = APDNN.Conv2D(32, 64, 3, stride=1, padding=1)

# Final Linear Layer: 64 channels * 2 * 2 = 256 features
fc1 = APDNN.Linear(64 * 2 * 2, 100)

params = conv1.parameters() + conv2.parameters() + conv3.parameters() + conv4.parameters() + fc1.parameters()
num_params = conv1.param_count() + conv2.param_count() + conv3.param_count() + conv4.param_count() + fc1.param_count()
optimizer = APDNN.SGD(params, 0.05)

# Efficiency Metrics Update (Per Image)
c1_m = conv2d_macs(32, 32, 8, 3, 3, 1)
c2_m = conv2d_macs(16, 16, 16, 3, 3, 8)
c3_m = conv2d_macs(8, 8, 32, 3, 3, 16)
c4_m = conv2d_macs(4, 4, 64, 3, 3, 32)
fc_m = linear_macs(256, 100)
total_macs_new = c1_m + c2_m + c3_m + c4_m + fc_m

print(f"Total MACs per image: {total_macs_new:,}")
print(f"Total FLOPs per image: {macs_to_flops(total_macs_new):,}")

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
        
        # --- BLOCK 1: 32x32 -> 16x16 ---
        c1 = conv1.forward(batch_tensor)
        a1 = APDNN.relu(c1)
        p1 = APDNN.maxpool2d(a1, pool_size=2, stride=2)
        
        # --- BLOCK 2: 16x16 -> 8x8 ---
        c2 = conv2.forward(p1)
        a2 = APDNN.relu(c2)
        p2 = APDNN.maxpool2d(a2, pool_size=2, stride=2)
        
        # --- BLOCK 3: 8x8 -> 4x4 ---
        c3 = conv3.forward(p2)
        a3 = APDNN.relu(c3)
        p3 = APDNN.maxpool2d(a3, pool_size=2, stride=2)
        
        # --- BLOCK 4: 4x4 -> 2x2 ---
        c4 = conv4.forward(p3)
        a4 = APDNN.relu(c4)
        p4 = APDNN.maxpool2d(a4, pool_size=2, stride=2)
        
        # --- CLASSIFIER ---
        # Flatten: [batch_size, 256, 2, 2] -> [batch_size, 1024]
        flat = p4.view([actual_batch_size, 64 * 2 * 2])
        
        # Output logits (100 classes)
        logits = fc1.forward(flat)
        
        # Compute loss using cross-entropy
        loss = APDNN.cross_entropy_loss(logits, batch_labels)
        epoch_loss += loss.data[0]
        
        loss.backward()
        optimizer.step()
        
        batch_count += 1
        if batch_count % 50 == 0:
            print(f"  Batches done: {batch_count} | Current Loss: {loss.data[0]:.4f}")

        # --- CLEAN UP ---
        # Explicitly delete all intermediate tensors to trigger C++ destructors
        del c1, a1, p1
        del c2, a2, p2
        del c3, a3, p3
        del c4, a4, p4
        del flat, logits, loss, batch_tensor
        
    
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
    data_dir=data_dir,
    label_to_idx=label_to_idx,
    batch_size=64,
    shuffle=False,
    augment=False
)

correct = 0
total = 0

for batch_tensor, batch_labels in test_dataset:
    actual_batch_size = len(batch_labels)
    
    # Block 1: 32x32 -> 16x16
    c1 = conv1.forward(batch_tensor)
    a1 = APDNN.relu(c1)
    p1 = APDNN.maxpool2d(a1, pool_size=2, stride=2)
    
    # Block 2: 16x16 -> 8x8
    c2 = conv2.forward(p1)
    a2 = APDNN.relu(c2)
    p2 = APDNN.maxpool2d(a2, pool_size=2, stride=2)
    
    # Block 3: 8x8 -> 4x4
    c3 = conv3.forward(p2)
    a3 = APDNN.relu(c3)
    p3 = APDNN.maxpool2d(a3, pool_size=2, stride=2)
    
    # Block 4: 4x4 -> 2x2
    c4 = conv4.forward(p3)
    a4 = APDNN.relu(c4)
    p4 = APDNN.maxpool2d(a4, pool_size=2, stride=2)
    
    # Flatten: [batch_size, 256, 2, 2] -> [batch_size, 1024]
    flat = p4.view([actual_batch_size, 64 * 2 * 2])
    logits = fc1.forward(flat)
    
    # --- ARGMAX & ACCURACY ---
    # We iterate through the batch to find the highest logit for each sample
    for i in range(actual_batch_size):
        max_idx = 0
        # logits.data is a flat list, so we offset by (sample_index * num_classes)
        max_val = logits.data[i * 100]
        for c in range(1, 100):
            if logits.data[i * 100 + c] > max_val:
                max_val = logits.data[i * 100 + c]
                max_idx = c
        
        if max_idx == batch_labels[i]:
            correct += 1
        total += 1
    
    # --- CLEAN UP ---
    # Vital for long evaluation runs on large test sets
    del c1, a1, p1
    del c2, a2, p2
    del c3, a3, p3
    del c4, a4, p4
    del flat, logits, batch_tensor

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
