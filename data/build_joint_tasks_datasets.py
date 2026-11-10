import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data import load_substance, synthetic_k0_E0
from analysis.plotting import set_standard_plot_params

set_standard_plot_params()

test_size = 0.3

def save_dataset(curves, labels, label_map, out_path):
    data = {
        "curves": curves,
        "labels": labels,
        "label_map": label_map
    }
    torch.save(data, out_path)
    print(f"Saved: {out_path}")
    print("  Unique labels:", set(labels))
    print("  Num curves:", len(curves))


def normalize_curve(curve):
    #normalising only current:
    current = curve[1]
    v = curve[0]
    current_norm = (current - global_min) / (
            global_max - global_min + 1e-8
    )
    return torch.stack([v, current_norm], dim=0)

def augment_curve(curve, i):
    # curve: (2, N)
    v = curve[0]
    current = curve[1]
    # 1. Gaussian noise
    current_noise = current + 0.005 * current.max().item() * torch.randn_like(current)

    # 2. Horizontal shift
    current_rolled = current_noise.roll(shifts=np.random.randint(-10, 10), dims=0)

    # 3. Vertical scaling
    current_vertical_scale = current_rolled * (1 + 0.01 * torch.randn(1))

    # 4. Baseline drift
    current_vertical_drift = current_vertical_scale + 0.002 * current.max().item() * torch.randn(1)

    if i==0:
        curv_evolution.append([v, current_noise])
        curv_evolution.append([v, current_rolled])
        curv_evolution.append([v, current_vertical_scale])
        curv_evolution.append([v, current_vertical_drift])


    return torch.stack([v, current_vertical_drift], dim=0)


if __name__ == "__main__":

    base_dir = synthetic_k0_E0
    out_dir  = r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\Synthetic_data_k0_E0_joint_datasets"

    os.makedirs(out_dir, exist_ok=True)

    global_min = float("inf")
    global_max = float("-inf")

    for substance_id in [0, 1, 2, 3]:
        in_path = os.path.join(base_dir, f"substance_{substance_id}.pt")
        curves, labels, label_map = load_substance(in_path)
        for curve in curves:
            current = curve[1]
            global_min = min(global_min, current.min().item())
            global_max = max(global_max, current.max().item())
            plt.plot(curve[0], curve[1])

    plt.show(block=True)

    for substance_id in [0, 1, 2, 3]:
        in_path = os.path.join(base_dir, f"substance_{substance_id}.pt")

        print(f"\n=== Processing {in_path} ===")

        curves, labels, label_map = load_substance(in_path)

        train_curves, test_curves, train_labels, test_labels = train_test_split(
            curves,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=42
        )

        normalised_test_curves = []
        augmented_train_curves = []
        augmented_train_labels = []
        # for visualisation:
        i = 0
        curv_evolution = []

        for test_curve in test_curves:
            normalised_test_curves.append(normalize_curve(test_curve))

        for c, lbl in zip(train_curves, train_labels):
            c_norm = normalize_curve(c)

            if i==0:
                curv_evolution.append(c)
                curv_evolution.append(c_norm)

            augmented_train_curves.append(c_norm)
            augmented_train_labels.append(lbl)

            for _ in range(3):
                augmented_train_curves.append(
                    augment_curve(c_norm.clone(), i)
                )
                i=1
                augmented_train_labels.append(lbl)

        plt.plot(curv_evolution[0][0], curv_evolution[0][1] * 1.e6, label='Original curve')
        plt.xlabel("Voltage [V]")
        plt.ylabel("Current [$\mu$A]")
        plt.show(block=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        labels = [
            'Normalisation',
            'Gaussian noise',
            'Horizontal shift',
            'Vertical scaling',
            'Vertical shift'
        ]

        # Full plot
        for i, label in zip(range(1, 6), labels):
            ax1.plot(curv_evolution[i][0], curv_evolution[i][1], label=label)

        ax1.set_xlabel("Voltage [V]")
        ax1.set_ylabel("Normalised current")
        ax1.set_title('A', fontweight='bold', loc='left')
        ax1.legend()

        # Zoomed plot
        for i, label in zip(range(1, 6), labels):
            ax2.plot(curv_evolution[i][0], curv_evolution[i][1], label=label, lw=2.5)

        # === Choose your zoom region here ===
        ax2.set_xlim(0.33, 0.42)
        ax2.set_ylim(0.7, 0.83)
        # ====================================

        ax2.set_xlabel("Voltage [V]")
        # ax2.set_ylabel("Normalised current")
        ax2.set_title('B', fontweight='bold', loc='left')

        plt.tight_layout()
        plt.show(block=True)

        # Save train file
        out_train = os.path.join(out_dir, f"substance{substance_id}_joint_train.pt")
        save_dataset(augmented_train_curves, augmented_train_labels, label_map, out_train)

        # Save test file
        out_test = os.path.join(out_dir, f"substance{substance_id}_joint_test.pt")
        save_dataset(normalised_test_curves, test_labels, label_map, out_test)
