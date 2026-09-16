import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt


def load_substance(path):
    data = torch.load(path)
    curves = data["curves"]
    labels = data["labels"]
    label_map = data.get("label_map", {})
    return curves, labels, label_map


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


def split_substance(curves, labels, train_ratio=0.7):
    """
    Shuffle and split curves into train/test using a percentage.
    train_ratio: float between 0 and 1 (e.g., 0.7 means 70% train)
    """

    # Shuffle indices
    idx = list(range(len(curves)))
    random.shuffle(idx)

    curves = [curves[i] for i in idx]
    labels = [labels[i] for i in idx]

    # Compute split point
    train_size = int(len(curves) * train_ratio)

    # Split
    train_curves = curves[:train_size]
    train_labels = labels[:train_size]

    test_curves = curves[train_size:]
    test_labels = labels[train_size:]

    return train_curves, train_labels, test_curves, test_labels


def normalize_curve(curve, idx=None):
    # curve: tensor of shape (channels, N)
    # Before normalization
    if idx is not None and idx < 3:  # print only for first 3 curves
        print(f"\nCurve {idx} BEFORE normalization:")
        print("  Mean per channel:", curve.mean(dim=1).tolist())
        print("  Std per channel:", curve.std(dim=1).tolist())

    mean = curve.mean(dim=1, keepdim=True)
    std = curve.std(dim=1, keepdim=True) + 1e-8
    norm_curve = (curve - mean) / std

    # After normalization
    if idx is not None and idx < 3:
        print(f"Curve {idx} AFTER normalization:")
        print("  Mean per channel:", norm_curve.mean(dim=1).tolist())
        print("  Std per channel:", norm_curve.std(dim=1).tolist())

    return norm_curve

def augment_curve(curve):
    # 1. Gaussian noise
    curve = curve + 0.05 * torch.randn_like(curve)

    # 2. Horizontal shift
    curve = curve.roll(shifts=np.random.randint(-10, 10), dims=1)

    # 3. Vertical scaling
    curve = curve * (1 + 0.1 * torch.randn(1))

    # 4. Baseline drift
    curve = curve + 0.02 * torch.randn(1)

    return curve

def plot_curve_versions(norm, noise1, noise2, idx):
    plt.figure(figsize=(10, 6))

    # normalized curve
    plt.plot(norm[1].numpy(), label="Normalized (channel 0)", alpha=0.7)

    # noise versions
    plt.plot(noise1[1].numpy(), label="Noise v1 (channel 0)", alpha=0.7)
    plt.plot(noise2[1].numpy(), label="Noise v2 (channel 0)", alpha=0.7)

    plt.title(f"Curve {idx} — Raw vs Normalized vs Noise")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":

    base_dir = r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\data_45_CVs_scan_rates"
    out_dir  = r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets"

    os.makedirs(out_dir, exist_ok=True)

    for substance_id in [0, 1, 2, 3]:
        in_path = os.path.join(base_dir, f"substance{substance_id}.pt")

        print(f"\n=== Processing {in_path} ===")

        curves, labels, label_map = load_substance(in_path)

        augmented_curves = []
        augmented_labels = []

        for i, (c, lbl) in enumerate(zip(curves, labels)):
            c_norm = normalize_curve(c, idx=i)

            # two noisy copies
            noise1 = augment_curve(c_norm)
            noise2 = augment_curve(c_norm)
            noise3 = augment_curve(c_norm)
            noise4 = augment_curve(c_norm)

            # Save augmented versions
            augmented_curves.append(c_norm)
            augmented_curves.append(noise1)
            augmented_curves.append(noise2)
            augmented_curves.append(noise3)
            augmented_curves.append(noise3)

            # Append labels (same label for all three)
            augmented_labels.append(lbl)
            augmented_labels.append(lbl)
            augmented_labels.append(lbl)
            augmented_labels.append(lbl)
            augmented_labels.append(lbl)

            # Plot only first few curves
            if i < 3:
                plot_curve_versions(c_norm, noise1, noise2, idx=i)

        train_curves, train_labels, test_curves, test_labels = split_substance(augmented_curves, augmented_labels,
                                                                               train_ratio=0.8)

        # Save train file
        out_train = os.path.join(out_dir, f"substance{substance_id}_joint_train.pt")
        save_dataset(train_curves, train_labels, label_map, out_train)

        # Save test file
        out_test = os.path.join(out_dir, f"substance{substance_id}_joint_test.pt")
        save_dataset(test_curves, test_labels, label_map, out_test)
