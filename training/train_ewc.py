import torch
from torch.utils.data import DataLoader
from models.small_net import CVMidNet, print_model_info
from training.run_joint_training import debug_predictions
from training.train_none import evaluate
from training.trainer import train_one_epoch_ewc


class EWC:
  def __init__(self, model, loader, device, criterion):
    self.model = model
    self.device = device
    self.criterion = criterion

    self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    self.fisher = self._compute_fisher(loader)

    # for n, f in self.fisher.items():
    #   print(
    #     n,
    #     "mean =", f.mean().item(),
    #     "max =", f.max().item(),
    #     "min =", f.min().item()
    #   )

  def _compute_fisher(self, loader):
    fisher = {n: torch.zeros_like(p, device=self.device)
              for n, p in self.model.named_parameters()
              if p.requires_grad}

    self.model.eval()

    for x, y in loader:
      x, y = x.to(self.device), y.to(self.device)
      self.model.zero_grad()
      logits = self.model(x)
      loss = self.criterion(logits, y)
      loss.backward()

      for (n, p) in self.model.named_parameters():
        if p.grad is not None:
          fisher[n] += p.grad.detach() ** 2 / len(loader)

    return fisher

  def penalty(self, model):
    loss = 0.0
    for (n, p) in model.named_parameters():
      if n in self.fisher:
        loss += (self.fisher[n] * (p-self.params[n])**2).sum()

    return loss


def train_ewc(context_datasets, test_datasets, num_classes, criterion, lambda_ewc=1e10, epochs=20,
              in_channels=1, lr=1e-3):
  ewc_list = []

  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Initialize model
  model = CVMidNet(in_channels=in_channels, num_classes=num_classes).to(device)

  print_model_info(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  history = {
    'test_accuracy_task_1': [],
    'test_accuracy_task_2': [],
    'test_accuracy_task_3': [],
    'test_accuracy_task_4': [],
  }

  for t, train_dataset in enumerate(context_datasets):
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"\n=== Training on context {t + 1} ===")

    # Scheduler for changng learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode="min",
      factor=0.8,  # halve LR
      patience=5,  # wait 5 epochs with no improvement
    )

    for epoch in range(epochs):
      loss, acc = train_one_epoch_ewc(model, loader, optimizer, device, ewc_list, lambda_ewc, criterion)
      current_lr = optimizer.param_groups[0]['lr']
      scheduler.step(loss)
      print(f"  Epoch {epoch + 1}: loss={loss:.4f}, acc={acc:.4f}, lr={current_lr:.4f}")

      # ---- EARLY STOPPING WHEN ACC > 0.9 ----
      if acc >= 0.95:
        print(f"Early stopping triggered at epoch {epoch + 1} (acc={acc:.3f})")
        break
      # ----------------------------------------

    # build EWC object for this task
    ewc_list.append(EWC(model, loader, device, criterion))

    # Evaluate on all tasks seen so far
    print(f"\n--- Evaluation after finishing context {t + 1} ---")
    for eval_id in range(4):
      acc = evaluate(model, test_datasets[eval_id], device)
      history[f'test_accuracy_task_{eval_id + 1}'].append(acc)
      print(f"Accuracy on Task {eval_id + 1}: {acc:.3f}")
      print(f"\n=== Debug Task {eval_id + 1}===")
      debug_predictions(model, test_datasets[eval_id], device)

  return model, history
