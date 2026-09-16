import torch
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import CVDataset
from training.train_none import train_none


def load_task(path):
    print(f"Loading: {path}")
    return CVDataset(path)


def plot_mean_accuracies(mean_results_over_time, N_RUNS, save_path="NONE_baseline_performance.png"):
    substances = [0, 1, 2, 3]
    timepoints = ["after_task_1", "after_task_2", "after_task_3"]

    data = np.array([[mean_results_over_time[t][sid] for sid in substances]
                     for t in timepoints])

    x = np.arange(len(timepoints))
    width = 0.2

    plt.figure(figsize=(10, 6))

    for i, sid in enumerate(substances):
        plt.bar(x + i * width, data[:, i], width, label=f"Substance {sid}")

    plt.xticks(x + width * 1.5, timepoints)
    plt.ylim(0, 1)
    plt.ylabel("Average accuracy")
    plt.title(f"Accuracy per Substance Over Time for None Baseline for {N_RUNS} runs")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved NONE baseline performance plot to {save_path}")


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Load substance data
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

    context_datasets = [
        ConcatDataset([substance0_train, substance1_train]),
        substance2_train,
        substance3_train
    ]

    test_datasets = [
        ConcatDataset([substance0_test, substance1_test]),
        substance2_test,
        substance3_test
    ]

    substance_datasets = {
        0: substance0_test,
        1: substance1_test,
        2: substance2_test,
        3: substance3_test
    }

    num_classes = 4
    sample_x, _ = substance0_train[0]
    in_channels = sample_x.shape[0]

    # -----------------------------
    # Run NONE baseline N times
    # -----------------------------
    N_RUNS = 100

    all_runs_results = {
        "after_task_1": [],
        "after_task_2": [],
        "after_task_3": []
    }

    for run in range(N_RUNS):
        print(f"\n=== NONE baseline run {run+1}/{N_RUNS} ===")

        model, results, results_over_time = train_none(
            context_datasets=context_datasets,
            test_datasets=test_datasets,
            substance_datasets=substance_datasets,
            num_classes=num_classes,
            in_channels=in_channels,
            epochs=3,
            lr=5e-4
        )

        for tp in ["after_task_1", "after_task_2", "after_task_3"]:
            acc_list = [results_over_time[tp][sid] for sid in range(4)]
            all_runs_results[tp].append(acc_list)

    # -----------------------------
    # Compute mean accuracies
    # -----------------------------
    mean_results_over_time = {}

    for tp in ["after_task_1", "after_task_2", "after_task_3"]:
        arr = np.array(all_runs_results[tp])  # shape (N_RUNS, 4)
        mean_arr = arr.mean(axis=0)
        mean_results_over_time[tp] = {
            0: mean_arr[0],
            1: mean_arr[1],
            2: mean_arr[2],
            3: mean_arr[3]
        }

    # -----------------------------
    # Plot mean accuracies
    # -----------------------------
    plot_mean_accuracies(mean_results_over_time, N_RUNS)
