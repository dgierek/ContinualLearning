import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, optimizer, device):
    """
    Train the model for a single epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network being trained.

    dataloader : torch.utils.data.DataLoader
        Provides batches of (inputs, labels). Each batch is a tuple (x, y):
        - x : tensor of shape (batch_size, input_dim or channels, ...)
        - y : tensor of integer class labels

    optimizer : torch.optim.Optimizer
        Optimization algorithm (Adam, SGD, etc.) controlling parameter updates.

    device : torch.device
        CPU or GPU where tensors and model should be placed.

    Returns
    -------
    avg_loss : float
        Mean cross‑entropy loss over all batches.

    accuracy : float
        Fraction of correctly predicted samples across the epoch.

    Notes
    -----
    - The function performs one full pass over the dataset.
    - CrossEntropyLoss expects raw logits (no softmax).
    - `model.train()` enables dropout, batchnorm updates, etc.
    """

    # Put model into training mode (important for dropout/batchnorm)
    model.train()

    # Standard multi-class classification loss
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0      # Accumulate batch losses
    correct = 0           # Count correct predictions
    total = 0             # Count total samples

    # Iterate over batches
    for x, y in dataloader:
        # Move data to GPU/CPU
        x, y = x.to(device), y.to(device)

        # Reset gradients from previous step
        optimizer.zero_grad()

        # Forward pass: model outputs raw logits
        logits = model(x)

        # Compute loss for this batch
        loss = criterion(logits, y)

        # Backpropagate gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Compute predictions: argmax over class dimension
        preds = logits.argmax(dim=1)

        # Count correct predictions
        correct += (preds == y).sum().item()
        total += y.size(0)

    # Average loss over batches, accuracy over all samples
    return total_loss / len(dataloader), correct / total

def train_one_epoch_ewc(model, dataloader, optimizer, device, ewc_list, lambda_ewc):
    model.train()

    weights = torch.tensor([3.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    total_loss = 0.0  # Accumulate batch losses
    correct = 0  # Count correct predictions
    total = 0  # Count total samples
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        if ewc_list:
            reg = sum(ewc.penalty(model) for ewc in ewc_list)
            loss = loss + lambda_ewc * reg

        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Compute predictions: argmax over class dimension
        preds = logits.argmax(dim=1)

        # Count correct predictions
        correct += (preds == y).sum().item()
        total += y.size(0)

    # Average loss over batches, accuracy over all samples
    return total_loss / len(dataloader), correct / total

def train_one_epoch_mas(model, dataloader, optimizer, device, mas, lambda_mas):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x)
        ce_loss = ce_loss_fn(logits, y)

        # MAS penalty
        mas_loss = mas.penalty(model)
        loss = ce_loss + lambda_mas * mas_loss

        # Backward
        loss.backward()

        # Update MAS importance (gradient of output norm)
        mas.update_importance(model)

        optimizer.step()

        # Accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        total_loss += loss.item()

    return total_loss / len(dataloader), correct / total
