import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / len(dataloader), correct / total
