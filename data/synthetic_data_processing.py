import pandas as pd
from data import synthetic_k0_E0_raw
import torch
import numpy as np

root = synthetic_k0_E0_raw
out_dir = r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\Synthetic_data_k0_E0_processed"

labels = [0, 1, 2, 3]

print(list(root.rglob("*.csv")))

for file, label in zip(root.rglob("*.csv"), labels):
    df = pd.read_csv(file)

    cols = list(df.columns)

    curves = []
    curves_labels = []

    # process pairs of columns
    for i in range(0, len(cols), 2):
        voltage_col = cols[i]
        current_col = cols[i + 1]

        voltage = pd.to_numeric(df[voltage_col], errors="coerce").to_numpy()
        current = pd.to_numeric(df[current_col], errors="coerce").to_numpy()

        curve = torch.tensor(
            np.array([voltage, current]),
            dtype=torch.float32
        )

        curves.append(curve)
        curves_labels.append(label)

    torch.save(
        {
            "curves"   : curves,
            "labels"   : curves_labels,
            "label_map": {}
        },
        out_dir + f"\substance_{label}.pt"
    )
