import torch
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import numpy as np

from analysis.plotting import set_standard_plot_params
from data.dataset import load_synthetic_test_data, load_synthetic_k0_E0_test_data
from models.small_net import CVMidNet, print_model_info


def print_label_stats(dataset, name):
  labels = [y for _, y in dataset]
  unique, counts = np.unique(labels, return_counts=True)
  print(f"{name}: {dict(zip(unique, counts))}")

test_data = load_synthetic_k0_E0_test_data()
substance0_test, substance1_test, substance2_test, substance3_tes = test_data

print_label_stats(substance0_test, "substance0")
print_label_stats(substance1_test, "substance1")
print_label_stats(substance2_test, "substance2")
print_label_stats(substance3_tes, "substance3")

model = CVMidNet()

file_name = 'Standard_baseline_150_epochs_dataset_Synthetic_k0_E0'

checkpoint = torch.load(rf"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results\model_{file_name}.pth")


# load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print_model_info(model)


for x, y in ConcatDataset(test_data):
  x = x.unsqueeze(0)
  y = torch.tensor([y], dtype=torch.long)

  output = model(x)  # shape [1, 4]
  pred_class = output.argmax(dim=1)
  print('True class: ', y, 'Predicted class:', pred_class)



titles = ['A', 'B', 'C', 'D']

set_standard_plot_params()
fig, axes = plt.subplots(2, 2, figsize=(15, 7.5), sharey=True, sharex=True)

ax1, ax2, ax3, ax4 = axes.flat

ax1.set_ylabel('Normalised\ncurrent')
ax3.set_ylabel('Normalised\ncurrent')
ax3.set_xlabel('Voltage [V]')
ax4.set_xlabel('Voltage [V]')

for substance, ax, title in zip(test_data, axes.flat, titles):
    for x, y in substance:
      ax.plot(x[0],x[1])
      ax.set_title(title, fontweight='bold', loc='left')
      ax.text(0, 0.4, f'True class: {y}')

plt.show(block=True)