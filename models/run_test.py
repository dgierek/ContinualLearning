from data.dataset import CVDataset
from torch.utils.data import DataLoader
import torch
from models.resnet_test import ResNet18_1D

# Load test dataset
dataset = CVDataset("data/pandas_converted_data/test_data.pt")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Inspect one batch
x, y = next(iter(loader))
print(x.shape)   # Expect something like (batch, channels, length)
print(y)         # Integer class labels

# Determine number of classes from dataset metadata
num_classes = len(dataset.label_map)

# Initialize the 1D ResNet model
model = ResNet18_1D(num_classes)

# Forward pass through the model
out = model(x)

# Convert logits to probabilities
probs = torch.softmax(out, dim=1)

# Predicted class indices
preds = torch.argmax(out, dim=1)

print("Logits:\n", out)
print("Probabilities:\n", probs)
print("Predicted classes:", preds)
print("True labels:", y)
