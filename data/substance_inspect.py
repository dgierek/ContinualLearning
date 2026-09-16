import torch
import matplotlib.pyplot as plt
import numpy as np

def load_substance(path):
    print(f"Loading: {path}")
    data = torch.load(path)
    curves = data["curves"]
    labels = data["labels"]
    label_map = data.get("label_map", {})
    meta = data.get("meta", None)  # optional metadata

    print("\n=== Dataset Info ===")
    print(f"Number of curves: {len(curves)}")
    print(f"Unique labels: {set(labels)}")
    print(f"Label map: {label_map}")

    return curves, labels, meta


if __name__ == "__main__":
    # Path to your substance file
    path = r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\data_45_CVs_scan_rates\substance3.pt"

    curves, labels, meta = load_substance(path)
