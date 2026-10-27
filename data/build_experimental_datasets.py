import os
import glob
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
import random
import matplotlib.pyplot as plt

TARGET_LEN = 1001

###############################################
# 1. Load experimental CSV
###############################################

def load_cv_file(path):
    # Skip metadata lines
    df = pd.read_csv(path, skiprows=5, encoding="utf-16")
    df = df.dropna()
    df.columns = ["V", "I"]

    # Convert to float (this is the critical step)
    df["V"] = pd.to_numeric(df["V"], errors="coerce")
    df["I"] = pd.to_numeric(df["I"], errors="coerce")

    voltage = df["V"].values
    current = df["I"].values
    return voltage, current

###############################################
# 2. Resample to fixed length
###############################################

def resample_cv(voltage, current, target_len=TARGET_LEN):
    # Find turning point (max voltage)
    turn_idx = np.argmax(voltage)

    # Forward scan (increasing voltage)
    v_f = voltage[:turn_idx + 1]
    i_f = current[:turn_idx + 1]

    # Reverse scan (decreasing voltage)
    v_r = voltage[turn_idx:]
    i_r = current[turn_idx:]

    # Reverse the reverse scan so voltage becomes increasing
    v_r_rev = v_r[::-1]
    i_r_rev = i_r[::-1]

    # Resample each half
    half = target_len // 2

    v_f_uni = np.linspace(v_f.min(), v_f.max(), half)
    i_f_uni = np.interp(v_f_uni, v_f, i_f)

    v_r_uni = np.linspace(v_r_rev.min(), v_r_rev.max(), target_len - half)
    i_r_uni = np.interp(v_r_uni, v_r_rev, i_r_rev)

    # Combine back into forward + reverse order
    v_res = np.concatenate([v_f_uni, v_r_uni[::-1]])
    i_res = np.concatenate([i_f_uni, i_r_uni[::-1]])

    return v_res, i_res


###############################################
# 3. Your normalization
###############################################

def normalize_curve(curve, idx=None):
    mean = curve.mean(dim=1, keepdim=True)
    std = curve.std(dim=1, keepdim=True) + 1e-8
    norm_curve = (curve - mean) / std
    return norm_curve

###############################################
# 4. Your augmentation
###############################################

def augment_curve(curve):
    # curve: (2, N)
    v = curve[0]
    i = curve[1]

    plt.plot(v, i)
    plt.title('Original CV')
    plt.show(block=True)

    # 1. Gaussian noise (current only)
    i = i + 0.05 * torch.randn_like(i)

    plt.plot(v, i)
    plt.title('CV after gaussian noise')
    plt.show(block=True)


    # 2. Horizontal shift (current only)
    i = i.roll(shifts=np.random.randint(-10, 10), dims=0)

    plt.plot(v, i)
    plt.title('CV after horizontal shift')
    plt.show(block=True)

    # 3. Vertical scaling (current only)
    i = i * (1 + 0.1 * torch.randn(1))

    plt.plot(v, i)
    plt.title('CV after vertical scaling')
    plt.show(block=True)

    # 4. Baseline drift (current only)
    i = i + 0.02 * torch.randn(1)

    plt.plot(v, i)
    plt.title('CV after baseline drift')
    plt.show(block=True)

    return torch.stack([v, i], dim=0)

###############################################
# 5. Percentage split
###############################################

def split_substance(curves, labels, train_ratio=0.7):
    idx = list(range(len(curves)))
    random.shuffle(idx)

    curves = [curves[i] for i in idx]
    labels = [labels[i] for i in idx]

    train_size = int(len(curves) * train_ratio)

    train_curves = curves[:train_size]
    train_labels = labels[:train_size]

    test_curves = curves[train_size:]
    test_labels = labels[train_size:]

    return train_curves, train_labels, test_curves, test_labels

###############################################
# 6. Process one substance folder
###############################################

def process_substance(folder_path, label):
    curves = []
    labels = []

    for file in glob.glob(os.path.join(folder_path, "*.csv")):
        voltage, current = load_cv_file(file)
        v_res, i_res = resample_cv(voltage, current)

        curve = torch.tensor(np.stack([v_res, i_res], axis=0), dtype=torch.float32)

        curve = normalize_curve(curve)

        noise1 = augment_curve(curve)
        noise2 = augment_curve(curve)
        noise3 = augment_curve(curve)
        noise4 = augment_curve(curve)

        # import matplotlib.pyplot as plt
        # plt.plot(noise1[0], noise1[1])
        # plt.show(block=True)

        curves.append(curve)
        curves.append(noise1)
        curves.append(noise2)
        curves.append(noise3)
        curves.append(noise4)
        labels.append(label)
        labels.append(label)
        labels.append(label)
        labels.append(label)
        labels.append(label)

    return curves, labels

###############################################
# 7. Save dataset
###############################################

def save_dataset(curves, labels, label_map, out_path):
    data = {
        "curves": curves,
        "labels": labels,
        "label_map": label_map
    }
    torch.save(data, out_path)
    print("Saved:", out_path, "Num curves:", len(curves))

###############################################
# 8. Main
###############################################

base_dir = r"experimental_data"

substances = {
    0: "Co(phen)3",
    1: "FcMeOH",
    2: "FeCl2",
    3: "Syringaldazine"
}

out_dir = r"experimental_joint_datasets"
os.makedirs(out_dir, exist_ok=True)

for label, name in substances.items():
    folder = os.path.join(base_dir, name)

    print(f"\n=== Processing {name} ===")

    curves, labels = process_substance(folder, label)

    train_curves, train_labels, test_curves, test_labels = split_substance(curves, labels)

    save_dataset(train_curves, train_labels, substances, os.path.join(out_dir, f"substance{label}_exp_train.pt"))
    save_dataset(test_curves, test_labels, substances, os.path.join(out_dir, f"substance{label}_exp_test.pt"))
