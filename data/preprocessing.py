import numpy as np
import torch
from tqdm import tqdm

def resample_curve(y, target_len=256):
    """Resample a 1D array to a fixed length using linear interpolation."""
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, y)


def preprocess_row(row, target_len=256):
    # Convert lists to numpy arrays
    current = np.array(row["current"], dtype=np.float32)
    potential = np.array(row["potential"], dtype=np.float32)

    # Resample both
    current = resample_curve(current, target_len)
    potential = resample_curve(potential, target_len)

    # Normalize current
    current_norm = current / np.max(np.abs(current))

    # Broadcast context channels
    scan_channel = np.full(target_len, row["scan_rate"], dtype=np.float32)
    flow_channel = np.full(target_len, row["flow_rate"], dtype=np.float32)

    # Stack into (C, H, W) = (3, 1, target_len)
    tensor = [
      current_norm.tolist(),
      scan_channel.tolist(),
      flow_channel.tolist()
    ]

    return tensor


def preprocess_dataframe_and_save(rows, pt_file_name, target_len=256):
    """

    :param rows: pandas DataFrame returned by data.cvs_loader.load_cv_dataset
    :param target_len:
    :return:
    """
    tensors = []
    labels = []

    # Build label mapping
    substances = rows["substance"].unique()
    label_map = {name: i for i, name in enumerate(substances)}

    # Process each row
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Preprocessing"):
      tensor = preprocess_row(row, target_len=target_len)  # pure Python lists
      tensors.append(tensor)
      labels.append(label_map[row["substance"]])

    # Save everything in a single .pt file
    torch.save({
      "tensors"  : tensors,  # list of lists
      "labels"   : labels,  # list of ints
      "label_map": label_map
    }, pt_file_name)

    print(f"Saved processed dataset to {pt_file_name}")