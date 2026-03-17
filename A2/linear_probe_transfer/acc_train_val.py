import matplotlib.pyplot as plt

# Your data as lists
train_accuracies = [0.1220, 0.3183, 0.4970, 0.5853, 0.6466, 0.6691, 0.6985, 0.7240, 0.7363, 0.7430, 0.7459, 0.7624, 0.7722, 0.7730, 0.7724, 0.7845, 0.7871, 0.7935, 0.7967, 0.7990, 0.8090, 0.8010, 0.8049, 0.8078, 0.8141, 0.8204, 0.8231, 0.8190, 0.8217, 0.8239]  # Fill with your training accuracy values
val_accuracies = [0.3456, 0.5691, 0.6730, 0.7173, 0.7421, 0.7679, 0.7827, 0.7960, 0.8041, 0.8146, 0.8184, 0.8194, 0.8217, 0.8327, 0.8289, 0.8375, 0.8341, 0.8432, 0.8456, 0.8494, 0.8470, 0.8532, 0.8594, 0.8584, 0.8613, 0.8618, 0.8690, 0.8689, 0.8684, 0.8632]    # Fill with your validation accuracy values

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Training Accuracy', marker='o')
plt.plot(val_accuracies, label='Validation Accuracy', marker='s')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_acc.png')
plt.show()