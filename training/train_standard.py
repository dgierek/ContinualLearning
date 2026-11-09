import torch
from torch.utils.data import DataLoader
from models.small_net import CVMidNet
from training.trainer import train_one_epoch, evaluate
from models.small_net import print_model_info



def train_standard(train_dataset, test_dataset, num_classes, in_channels=1, epochs=20, lr=1e-3):

    # Select GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save the training history:
    history = {
        'training_loss': [],
        'testing_loss': [],
        'training_accuracy': [],
        'testing_accuracy': [],
        'learning_rate': []
    }

    # Initialize model for CV classification
    model = CVMidNet(in_channels=in_channels, num_classes=num_classes).to(device)

    print_model_info(model)

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
        history['training_loss'].append(loss)
        history['training_accuracy'].append(acc)
        scheduler.step(loss)

        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)

        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, lr={current_lr:.4f}")

        testing_loss, testing_accuracy = evaluate(model, test_dataset)
        history['testing_loss'].append(testing_loss)
        history['testing_accuracy'].append(testing_accuracy)

    return model, history