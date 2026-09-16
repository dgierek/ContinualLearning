from torch.utils.data import DataLoader
from models.small_net import CVNanoNet, CVMidNet
from training.run_joint_training import debug_predictions
from training.trainer import train_one_epoch

import torch

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

def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_none(context_datasets, test_datasets, substance_datasets, num_classes, in_channels=1, epochs=3, lr=1e-3):
    """
    Naive sequential training across contexts (NONE baseline).
    Catastrophic forgetting is expected.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    model = CVMidNet(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model_name = model.__class__.__name__
    print(f"Using model {model_name} with {count_parameters(model)} trainable parameters.")

    # Store forgetting results
    all_results = []

    results_over_time = {
        "after_task_1": [],
        "after_task_2": [],
        "after_task_3": []
    }

    # Sequentially train on each context
    for t, dataset in enumerate(context_datasets):
        print(f"\n=== Training on context {t+1} ===")

        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Scheduler for changng learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.8,  # halve LR
            patience=5,  # wait 5 epochs with no improvement
        )

        # Train on current task
        for epoch in range(epochs):
            loss, acc = train_one_epoch(model, loader, optimizer, device)
            scheduler.step(loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, lr={current_lr:.4f}")



        # Evaluate on all tasks seen so far
        print(f"\n--- Evaluation after finishing context {t+1} ---")
        accs = []
        for eval_id in range(3):
            acc = evaluate(model, test_datasets[eval_id], device)
            accs.append(acc)
            print(f"Accuracy on Task {eval_id+1}: {acc:.3f}")
            print(f"\n=== Debug Task {eval_id+1}===")
            debug_predictions(model, test_datasets[eval_id], device)

        sub_accs = evaluate_per_substance(model, substance_datasets, device)
        results_over_time[f"after_task_{t + 1}"] = sub_accs



        all_results.append(accs)

    return model, all_results, results_over_time
