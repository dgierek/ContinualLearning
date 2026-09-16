import torch
import matplotlib.pyplot as plt
import os

def plot_loss_curve(losses, title="Training Loss Curve", out_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")

    plt.show()

def plot_lr_curve(losses, title="Training learning rate", out_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title(title)
    plt.grid(True)

    if out_path is not None:
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot: {out_path}")

    plt.show()


if __name__ == "__main__":
    # Path to your saved loss file
    loss_path = r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\results\joint_losses.pt"

    if not os.path.exists(loss_path):
        raise FileNotFoundError(f"Loss file not found: {loss_path}")

    losses = torch.load(loss_path)
    print("Loaded losses:", losses)

    plot_loss_curve(
        losses,
        title="JOINT Training Loss",
        out_path="joint_loss_curve.png"
    )
