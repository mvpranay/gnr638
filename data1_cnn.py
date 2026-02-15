import time
import gc
import numpy as np
import APDNN
from data_loader import get_data_loader

# Data augmentation function (optional)
def augment_image(img: np.ndarray) -> np.ndarray:
    """Simple data augmentation: random flip and brightness."""
    if np.random.rand() > 0.5:
        img = np.fliplr(img)  # Horizontal flip
    
    brightness_factor = np.random.uniform(0.8, 1.2)
    img = np.clip(img * brightness_factor, 0, 1)
    
    return img

# Set random seed for reproducibility
APDNN.set_random_seed(42)

print("=" * 60)
print("CNN Training with Custom C++ Backend")
print("=" * 60)

# Load dataset with train/test split
train_dataset, test_dataset = get_data_loader(
    data_dir="data_1",
    batch_size=32,
    img_size=32,
    transform=augment_image,
    test_split=0.2,
    print_time=True
)

# Get class information
class_names = train_dataset.get_class_names()
num_classes = len(class_names)
input_size = 32 * 32 * 3  # Flattened image size

print(f"Classes: {class_names}")
print(f"Number of classes: {num_classes}")
print(f"Input size: {input_size}")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}\n")

# Build model
print("Building model...")
layer1 = APDNN.Linear(input_size, 128)
layer2 = APDNN.Linear(128, 64)
layer3 = APDNN.Linear(64, num_classes)

params = layer1.parameters() + layer2.parameters() + layer3.parameters()
optimizer = APDNN.SGD(params, 0.001)

print(f"Model built with {len(params)} parameter tensors\n")

# Training loop
num_epochs = 10
batch_size = 32

print("=" * 60)
print("Training")
print("=" * 60)

train_start = time.time()

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    # Shuffle training data
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    
    for batch_start in range(0, len(train_dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(train_dataset))
        batch_indices = indices[batch_start:batch_end]
        
        batch_images, batch_labels = train_dataset.get_batch(batch_indices)
        
        for img_tensor, label in zip(batch_images, batch_labels):
            optimizer.zero_grad()
            
            # Forward pass
            h1 = layer1.forward(img_tensor)
            a1 = APDNN.relu(h1)
            h2 = layer2.forward(a1)
            a2 = APDNN.relu(h2)
            logits = layer3.forward(a2)
            
            # Simple MSE loss (one-hot encoding)
            target_vec = [0.0] * num_classes
            target_vec[label] = 1.0
            target = APDNN.Tensor(target_vec, [1, num_classes])
            
            loss = APDNN.mse_loss(logits, target)
            epoch_loss += loss.data[0]
            
            loss.backward()
            optimizer.step()
            
            # Explicitly clean up intermediate tensors to prevent memory leak
            del h1, a1, h2, a2, logits, target, loss
        
        # Clean up batch data
        del batch_images, batch_labels
        
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
    
    # Force garbage collection after each epoch
    gc.collect()

train_time = time.time() - train_start
print(f"\nTraining completed in {train_time:.2f} seconds")

# Evaluation
print("\n" + "=" * 60)
print("Evaluation on Test Set")
print("=" * 60)

eval_start = time.time()

correct = 0
total = 0

for idx in range(len(test_dataset)):
    img_tensor, label = test_dataset[idx]
    img_tensor = APDNN.Tensor(img_tensor.flatten().tolist(), [1, input_size])
    
    # Forward pass
    h1 = layer1.forward(img_tensor)
    a1 = APDNN.relu(h1)
    h2 = layer2.forward(a1)
    a2 = APDNN.relu(h2)
    logits = layer3.forward(a2)
    
    # Predict
    pred = np.argmax(logits.data)
    if pred == label:
        correct += 1
    total += 1
    
    # Clean up intermediate tensors
    del h1, a1, h2, a2, logits, img_tensor

eval_time = time.time() - eval_start
accuracy = correct / total * 100

print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"Evaluation Time: {eval_time:.2f} seconds")

# Force garbage collection
gc.collect()

print("=" * 60)
print(f"Total Training Time: {train_time:.2f}s")
print(f"Total Evaluation Time: {eval_time:.2f}s")
print("=" * 60)

