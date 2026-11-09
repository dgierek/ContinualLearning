import torch
import torch.nn as nn
import numpy as np

criterion = torch.nn.CrossEntropyLoss()

def evaluate(model, test_dataset):
    # calculating test loss/accuracy
    model.eval()

    test_total_loss = 0.0  # Accumulate batch losses
    test_correct = 0  # Count correct predictions
    test_total = 0  # Count total samples

    for x_test, y_test in test_dataset:
        # Forward pass: model outputs raw logits
        test_logits = model(x_test.unsqueeze(0))

        y_test_tensor = torch.tensor([y_test], dtype=torch.long)

        # Compute loss for this batch
        test_loss = criterion(test_logits, y_test_tensor)

        # Track loss
        test_total_loss += test_loss.item()

        # Compute predictions: argmax over class dimension
        test_preds = test_logits.argmax(dim=1)

        # Count correct predictions
        # print('test prediction:', np.array(test_preds), 'correct answer', y_test)
        test_correct += (test_preds == y_test).sum().item()
        test_total += y_test_tensor.size(0)

    return test_total_loss / len(test_dataset), test_correct / test_total

def train_one_epoch(model, dataloader, optimizer, device):
    # Put model into training mode (important for dropout/batchnorm)
    model.train()

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
 # CHECK THAT !!!!!!!!!!!!!!!
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
