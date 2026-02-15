import APDNN

# Use the same seed as main.cpp
APDNN.set_random_seed(42)

# Create tensors
x = APDNN.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
print(f"Tensor data: {x.data}")
print(f"Tensor shape: {x.shape}")

# Create linear layer
layer = APDNN.Linear(2, 3)
print(f"Weights shape: {layer.weights.shape}")

# Forward pass
output = layer.forward(x)
print(f"Output: {output.data}")

# XOR Example
dataset = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0])
]

layer1 = APDNN.Linear(2, 5)
layer2 = APDNN.Linear(5, 1)

params = layer1.parameters() + layer2.parameters()
optimizer = APDNN.SGD(params, 0.1)

print("Training XOR Network...")

for epoch in range(2000):
    total_loss = 0
    for x_data, y_data in dataset:
        optimizer.zero_grad()
        
        input_tensor = APDNN.Tensor(x_data, [1, 2])
        target = APDNN.Tensor(y_data, [1, 1])
        
        h1 = layer1.forward(input_tensor)
        a1 = APDNN.relu(h1)
        out = layer2.forward(a1)
        
        loss = APDNN.mse_loss(out, target)
        total_loss += loss.data[0]
        
        loss.backward()
        optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss / 4:.4f}")

print("\nResults:")
for x_data, _ in dataset:
    input_tensor = APDNN.Tensor(x_data, [1, 2])
    out = layer2.forward(APDNN.relu(layer1.forward(input_tensor)))
    print(f"{x_data[0]} XOR {x_data[1]} = {out.data[0]:.4f}")