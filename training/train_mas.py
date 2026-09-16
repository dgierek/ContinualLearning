import torch
from torch import nn
from torch.utils.data import DataLoader

from models.small_net import CVMidNet
from training.run_joint_training import evaluate, debug_predictions


def evaluate_per_substance(model, datasets, device):
  """
  datasets: dict {substance_id: torch Dataset}
  returns: dict {substance_id: accuracy}
  """
  model.eval()
  results = {}

  with torch.no_grad():
    for sid, dataset in datasets.items():
      loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
      correct = 0
      total = 0

      for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

      results[sid] = correct / total

  return results


class MAS:
    def __init__(self, model, device):
        self.device = device

        # Store reference parameters θ*
        self.prev_params = {n: p.clone().detach() for n, p in model.named_parameters()}

        # Importance Ω_i
        self.omega = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()}

    def update_importance(self, model):
        """
        MAS importance update:
        Ω_i += | ∂||f(x)||_2 / ∂θ_i |
        """
        for n, p in model.named_parameters():
            if "bn" in n:      # skip BatchNorm
                continue
            if p.grad is not None:
                self.omega[n] += p.grad.detach().abs()

    def update_omega_after_task(self, model):
        """
        Normalize Ω_i after finishing a task.
        """
        for n in self.omega:
            self.omega[n] = torch.clamp(self.omega[n], min=0.0)

        # Update reference parameters θ*
        self.prev_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    def penalty(self, model):
        """
        MAS penalty: Σ Ω_i (θ_i - θ*_i)^2
        """
        loss = 0
        for n, p in model.named_parameters():
            if "bn" in n:
                continue
            loss += torch.sum(self.omega[n] * (p - self.prev_params[n])**2)
        return loss


def train_one_epoch_mas(model, dataloader, optimizer, device, mas, lambda_mas):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # Forward
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


def train_mas(context_datasets, test_datasets, substance_datasets,
              num_classes, criterion, lambda_mas=1e-3, epochs=20,
              in_channels=1, lr=1e-3):

    results = []
    results_over_time = {
        "after_task_1": [],
        "after_task_2": [],
        "after_task_3": []
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    model = CVMidNet(in_channels=in_channels, num_classes=num_classes).to(device)
    print(f"Using model {model.__class__.__name__} with {sum(p.numel() for p in model.parameters())} parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize MAS object
    mas = MAS(model, device)

    for t, train_dataset in enumerate(context_datasets):
        loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        print(f"\n=== Training on context {t + 1} ===")

        for epoch in range(epochs):
            loss, acc = train_one_epoch_mas(model, loader, optimizer, device, mas, lambda_mas)
            print(f"  Epoch {epoch + 1}: loss={loss:.4f}, acc={acc:.4f}")

            # Early stopping
            if acc >= 0.95:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Update MAS importance after finishing the task
        mas.update_omega_after_task(model)

        # Evaluate on all tasks seen so far
        print(f"\n--- Evaluation after finishing context {t + 1} ---")
        accs = []
        for eval_id in range(3):
            acc = evaluate(model, test_datasets[eval_id], device)
            accs.append(acc)
            print(f"Accuracy on Task {eval_id + 1}: {acc:.3f}")
            print(f"\n=== Debug Task {eval_id + 1} ===")
            debug_predictions(model, test_datasets[eval_id], device)

        # Per-substance accuracy
        sub_accs = evaluate_per_substance(model, substance_datasets, device)
        results_over_time[f"after_task_{t+1}"] = sub_accs

        # Store task-level accuracies
        accs = []
        for k in range(len(test_datasets)):
            accs.append(evaluate(model, test_datasets[k], device))
        results.append(accs)

    return model, results, results_over_time
