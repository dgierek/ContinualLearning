from data.dataset import CVDataset
from torch.utils.data import DataLoader
import torch
from models.resnet_test import ResNet18_1D

dataset = CVDataset("data/pandas_converted_data/test_data.pt")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

x, y = next(iter(loader))
print(x.shape)
print(y)

num_classes = len(dataset.label_map)
model = ResNet18_1D(num_classes)

# Forward pass
out = model(x)

probs = torch.softmax(out, dim=1)
preds = torch.argmax(out, dim=1)

print("Logits:\n", out)
print("Probabilities:\n", probs)
print("Predicted classes:", preds)
print("True labels:", y)

