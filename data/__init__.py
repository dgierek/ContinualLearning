import torch
from pathlib import Path

synthetic_k0_E0_raw = Path(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\Synthetic_data_k0_E0")

test_synthetic = Path(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\data_45_CVs_scan_rates")
experimental = Path(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experimental_data_processed")
synthetic_k0_E0 = Path(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\Synthetic_data_k0_E0_processed")

synthetic_k0_E0_joint = r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\Synthetic_data_k0_E0_joint_datasets"


def load_substance(path):
  data = torch.load(path)
  curves = data["curves"]
  labels = data["labels"]
  label_map = data.get("label_map", {})
  return curves, labels, label_map


