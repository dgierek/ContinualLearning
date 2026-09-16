import torch
from torch.utils.data import DataLoader
from models.small_net import CVNanoNet
from training.trainer import train_one_epoch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_joint(train_dataset, num_classes, in_channels=1, epochs=20, lr=1e-3):
    """
    Joint training across all contexts (upper-bound baseline).

    This corresponds to the 'JOINT' setting in continual learning literature
    and in the Nature Machine Intelligence 2022 paper (https://doi.org/10.1038/s42256-022-00568-3):
    - All contexts/tasks are merged into a single dataset.
    - The model trains on the combined data as if everything were available
      from the start.
    - This eliminates catastrophic forgetting entirely.
    - Serves as an upper-bound reference for continual learning methods.

    Parameters
    ----------
    train_dataset : torch.utils.data.Dataset
        A single dataset containing samples from all contexts combined.
        Typically created by concatenating context-specific datasets.

    num_classes : int
        Number of output classes for classification.

    in_channels : int, default=1
        Number of input channels (CV curves are usually 1-channel).

    epochs : int, default=20
        Number of epochs to train on the joint dataset.

    lr : float, default=1e-3
        Learning rate for Adam optimizer.

    Returns
    -------
    model : torch.nn.Module
        The trained model after joint exposure to all contexts.

    Notes
    -----
    - JOINT training is the "oracle" baseline: it assumes full access to all
      data at once, which is unrealistic in continual learning scenarios.
    - Used to measure the maximum achievable performance.
    - NONE vs JOINT reveals the magnitude of catastrophic forgetting.
    """

    # Select GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save the history of loss:
    epoch_losses = []

    # Save the history of learning rate:
    epoch_lr = []

    # Initialize model for CV classification
    model = CVNanoNet(in_channels=in_channels, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model_name = model.__class__.__name__
    print(f"Using model {model_name} with {count_parameters(model)} trainable parameters.")

    # Standard Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler for changng learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.8,  # halve LR
        patience=5,  # wait 5 epochs with no improvement
    )

    # Build dataloader for the joint dataset
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train for several epochs on the combined dataset
    for epoch in range(epochs):
        loss, acc = train_one_epoch(model, loader, optimizer, device)
        epoch_losses.append(loss)
        scheduler.step(loss)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_lr.append(current_lr)

        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, lr={current_lr:.4f}")

    return model, epoch_losses, epoch_lr
