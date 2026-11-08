from pathlib import Path
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

TARGET_LEN = 1001

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

    return torch.tensor(v_res, dtype=torch.float64), torch.tensor(i_res, dtype=torch.float64)

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

root = Path(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experimental_data")

substances = {
    0: "Co(phen)3",
    1: "FcMeOH",
    2: "FeCl2",
    3: "Syringaldazine"
}

for label, substance in substances.items():

    curves = []
    labels = []
    length = []

    substance_folder = root / substance

    for file in substance_folder.rglob("*.csv"):
        voltage, current = load_cv_file(file)

        # plt.plot(voltage, current, label='CV before resampling')
        length.append(voltage.shape[0])
        voltage, current = resample_cv(
            voltage,
            current
        )


        # plt.plot(voltage, current, label='CV after resampling')
        # plt.legend()
        # plt.show(block=True)

        curve = torch.stack([voltage, current])

        curves.append(curve)
        labels.append(label)

    torch.save(
        {
            "curves"   : curves,
            "labels"   : labels,
            "label_map": substances
        },
        rf"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experiental_data_processed\{substance}.pt"
    )
    print('Curves in ',substance,':', len(curves))
    print('Orginal average curve lenght:', np.mean(length))



