import torch
from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from analysis.plot_loss_curve import plot_loss_curve, plot_lr_curve
from data.dataset import CVDataset
from training.train_joint import train_joint


def load_task(path):
    print(f"Loading: {path}")
    return CVDataset(path)

def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
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


def debug_predictions(model, dataset, device):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()

            print("True labels:", y.cpu().tolist())
            print("Pred labels:", preds.tolist())
            print("Unique predicted:", set(preds.tolist()))

def confusion_matrix(model, dataset, device, num_classes):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)

            for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1

    return cm

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load susbstance data
    # -----------------------------
    substance0_train = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance0_joint_train.pt")
    substance1_train = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance1_joint_train.pt")
    substance2_train = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance2_joint_train.pt")
    substance3_train = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance3_joint_train.pt")


    substance0_test = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance0_joint_test.pt")
    substance1_test = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance1_joint_test.pt")
    substance2_test = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance2_joint_test.pt")
    substance3_test = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance3_joint_test.pt")

    # Joint dataset = all tasks combined
    joint_dataset = ConcatDataset([substance0_train, substance1_train, substance2_train, substance3_train])

    # Number of classes = 4 substances
    num_classes = 4

    # Determine number of input channels from first sample
    sample_x, _ = substance0_train[0]
    in_channels = sample_x.shape[0]

    # -----------------------------
    # Run JOINT training
    # -----------------------------
    print("\n=== Running JOINT training ===")

    epochs_num = 20

    model_save_path = fr"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\trained_models\joint_baseline_epochs{epochs_num}"

    model, epoch_loss, epoch_lr = train_joint(
        train_dataset=joint_dataset,
        num_classes=num_classes,
        in_channels=in_channels,
        epochs=epochs_num,
        lr=1e-3
    )

    # -----------------------------
    # Evaluate on each task
    # -----------------------------
    print("\n=== Evaluation ===")

    Task1_test = ConcatDataset([substance0_test, substance1_test])

    joint_confusions = []
    joint_accuracies = []

    for k, dataset in enumerate([Task1_test, substance2_test, substance3_test]):
        acc = evaluate(model, dataset, device)
        cm = confusion_matrix(model, dataset, device, num_classes)

        joint_confusions.append(cm)
        joint_accuracies.append(acc)

        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix Task {k + 1}, accuracy {acc:.4f}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.savefig(model_save_path+f"confusion_matrix_task{k+1}")
        plt.show()

    print("\n=== Debug Task 1 ===")
    debug_predictions(model, Task1_test, device)

    print("\n=== Debug Task 2 ===")
    debug_predictions(model, substance2_test, device)

    print("\n=== Debug Task 3 ===")
    debug_predictions(model, substance3_test, device)

    plot_loss_curve(epoch_loss, out_path=model_save_path+"loss_curve")
    plot_lr_curve(epoch_lr, out_path=model_save_path+"lr_history")

    torch.save(model.state_dict(), model_save_path)