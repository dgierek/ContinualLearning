from torch.utils.data import Dataset
import torch, numpy as np


def make_cv_tensor(row):
    current = np.array(row["current"], dtype=np.float32)
    potential = np.array(row["potential"], dtype=np.float32)

    # Normalize current
    current_norm = current / np.max(np.abs(current))

    # Broadcast context
    scan_channel = np.full_like(current_norm, row["scan_rate"], dtype=np.float32)
    flow_channel = np.full_like(current_norm, row["flow_rate"], dtype=np.float32)

    # Stack into channels: (C, H, W)
    # Here H = 1 (single scan), W = number of points
    return np.stack([current_norm, scan_channel, flow_channel], axis=0)


class CVDataset(Dataset):
    def __init__(self, pt_file_path):
        data = torch.load(pt_file_path)

        self.tensors = data["tensors"]   # list of lists
        self.labels = data["labels"]     # list of ints
        self.label_map = data["label_map"]

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        x = torch.tensor(self.tensors[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

