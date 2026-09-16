import torch
from torch.utils.data import DataLoader

from models.resnet_cv import ResNet18CV
from models.small_net import CVSmallCNN, CVMidNet
from training.run_joint_training import debug_predictions
from training.train_none import evaluate
from training.trainer import train_one_epoch_ewc


class EWC:
  def __init__(self, model, dataset, device, criterion):
    self.model = model
    self.device = device
    self.criterion = criterion

    self.means = {n: p.clone().detach() for n, p in model.named_parameters()}
    self.fisher = self._compute_fisher(dataset)

  def _compute_fisher(self, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    fisher = {n: torch.zeros_like(p, device=self.device)
              for n, p in self.model.named_parameters()}

    self.model.eval()
    for x, y in loader:
      x, y = x.to(self.device), y.to(self.device)
      self.model.zero_grad()
      logits = self.model(x)
      loss = self.criterion(logits, y)
      loss.backward()

      for (n, p) in self.model.named_parameters():
        if p.grad is not None:
          fisher[n] += p.grad.detach() ** 2

    for n in fisher:
      fisher[n] /= len(loader)
    return fisher

  def penalty(self, model):
    loss = 0.0
    for (n, p) in model.named_parameters():
      loss += (self.fisher[n] * (p - self.means[n]) ** 2).sum()
    return loss

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def train_ewc(context_datasets, test_datasets, substance_datasets, num_classes, criterion, lambda_ewc=1e10, epochs=20,
              in_channels=1, lr=1e-3):
  ewc_list = []
  results = []
  results_over_time = {
    "after_task_1": [],
    "after_task_2": [],
    "after_task_3": []
  }

  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Initialize model
  model = CVMidNet(in_channels=in_channels, num_classes=num_classes).to(device)

  model_name = model.__class__.__name__
  print(f"Using model {model_name} with {count_parameters(model)} trainable parameters.")

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  for t, train_dataset in enumerate(context_datasets):
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"\n=== Training on context {t + 1} ===")

    for epoch in range(epochs):
      loss, acc = train_one_epoch_ewc(model, loader, optimizer, device, ewc_list, lambda_ewc)
      current_lr = optimizer.param_groups[0]['lr']
      print(f"  Epoch {epoch + 1}: loss={loss:.4f}, acc={acc:.4f}, lr={current_lr:.4f}")

      # ---- EARLY STOPPING WHEN ACC > 0.9 ----
      if acc >= 0.95:
        print(f"Early stopping triggered at epoch {epoch + 1} (acc={acc:.3f})")
        break
      # ----------------------------------------

    # build EWC object for this task
    ewc_list.append(EWC(model, train_dataset, device, criterion))

    # Evaluate on all tasks seen so far
    print(f"\n--- Evaluation after finishing context {t + 1} ---")
    accs = []
    for eval_id in range(3):
      acc = evaluate(model, test_datasets[eval_id], device)
      accs.append(acc)
      print(f"Accuracy on Task {eval_id + 1}: {acc:.3f}")
      print(f"\n=== Debug Task {eval_id + 1}===")
      debug_predictions(model, test_datasets[eval_id], device)

    sub_accs = evaluate_per_substance(model, substance_datasets, device)
    results_over_time[f"after_task_{t+1}"] = sub_accs


    # evaluate on all tasks
    accs = []
    for k in range(len(test_datasets)):
      accs.append(evaluate(model, test_datasets[k], device))

    results.append(accs)

  return model, results, results_over_time
