from torch.utils.data import DataLoader
from models.small_net import CVMidNet, print_model_info
from training.run_joint_training import debug_predictions
from training.trainer import train_one_epoch

import torch

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


def train_none(context_datasets, test_datasets, num_classes, in_channels=1, epochs=3, lr=1e-3):
    """
    Naive sequential training across contexts (NONE baseline).
    Catastrophic forgetting is expected.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    model = CVMidNet(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print_model_info(model)

    history = {
      'test_accuracy_task_1': [],
      'test_accuracy_task_2': [],
      'test_accuracy_task_3': [],
      'test_accuracy_task_4': [],
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
        for eval_id in range(4):
            acc = evaluate(model, test_datasets[eval_id], device)
            history[f'test_accuracy_task_{eval_id+1}'].append(acc)
            print(f"Accuracy on Task {eval_id+1}: {acc:.3f}")
            print(f"\n=== Debug Task {eval_id+1}===")
            debug_predictions(model, test_datasets[eval_id], device)



    return model, history
