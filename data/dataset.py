import torch

class CVDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.curves = data["curves"]
        self.labels = data["labels"]
        self.label_map = data["label_map"]

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        x = self.curves[idx]      # shape (channels, length)
        y = self.labels[idx]      # integer label
        return x, y
