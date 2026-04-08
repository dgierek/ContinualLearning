import torch
from torch.utils.data import DataLoader
from models.resnet_cv import ResNet18CV
from training.trainer import train_one_epoch

def train_joint(train_dataset, num_classes, in_channels=1, epochs=20, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNet18CV(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        loss, acc = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}")

    return model
