import time
import gc
import APDNN
from data_loader import ImageFolderDataset

# Set random seed for reproducibility
APDNN.set_random_seed(42)

print("=" * 60)
print("CNN Training with Convolutional Neural Network")
print("=" * 60)

# Load datasets (grayscale, 0-9 folders)
print("\nLoading training data...")
train_dataset = ImageFolderDataset(
    data_dir="data_1",
    batch_size=32,
    shuffle=True,
    augment=True  # Enable augmentation
)

print("\nBuilding CNN model...")

# Model architecture for grayscale images (1 channel input)
conv1 = APDNN.Conv2D(1, 8, 3, stride=1, padding=1)  # 1x32x32 -> 8x32x32

# After maxpool2d: 32x32 -> 16x16
# 8 channels * 16 * 16 = 2048 features
fc1 = APDNN.Linear(8 * 16 * 16, 10)  # 10 classes (0-9)

params = conv1.parameters() + fc1.parameters()
num_params = conv1.param_count() + fc1.param_count()
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
        p1 = APDNN.maxpool2d(a1, pool_size=2, stride=2)
        
        # Flatten: [batch_size, 8, 16, 16] -> [batch_size, 2048]
        flat = p1.view([actual_batch_size, 8 * 16 * 16])
        
        # Output logits
        logits = fc1.forward(flat)
        
        # Compute loss using cross-entropy
        loss = APDNN.cross_entropy_loss(logits, batch_labels)
        epoch_loss += loss.data[0]
        
        loss.backward()
        optimizer.step()
        
        # Clean up intermediate tensors
        del c1, a1, p1, flat, logits, loss, batch_tensor
        
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
    data_dir="data_1",
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
    p1 = APDNN.maxpool2d(a1, pool_size=2, stride=2)
    
    # Flatten and output
    flat = p1.view([actual_batch_size, 8 * 16 * 16])
    logits = fc1.forward(flat)
    
    # Predict for each sample in batch (efficient argmax)
    for i in range(actual_batch_size):
        max_idx = 0
        max_val = logits.data[i * 10]
        for c in range(1, 10):
            if logits.data[i * 10 + c] > max_val:
                max_val = logits.data[i * 10 + c]
                max_idx = c
        if max_idx == batch_labels[i]:
            correct += 1
        total += 1
    
    # Clean up
    del c1, a1, p1, flat, logits, batch_tensor

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
