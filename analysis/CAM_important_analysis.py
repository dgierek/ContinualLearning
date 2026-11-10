import torch
from torch.utils.data import ConcatDataset
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

from data.dataset import load_synthetic_test_data, load_synthetic_k0_E0_test_data
from models.small_net import CVMidNet, print_model_info
from plotting import set_standard_plot_params

def forward_hook(module, input, output):
  global activations
  activations = output.detach()

def backward_hook(module, grad_in, grad_out):
  global gradients
  gradients = grad_out[0].detach()

activations = None
gradients = None

# create model architecture
model = CVMidNet()

file_name = 'Standard_baseline_100_epochs_dataset_Synthetic_k0_E0'

checkpoint = torch.load(rf"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results\model_{file_name}.pth")

# load weights
model.load_state_dict(checkpoint['model_state_dict'])
print_model_info(model)

# inference mode
model.eval()

model.conv3.register_forward_hook(forward_hook)
model.conv3.register_full_backward_hook(backward_hook)

test_data = load_synthetic_k0_E0_test_data()
test_data_concat = ConcatDataset(test_data)

print('Loaded test dataset of length ', test_data_concat.__len__())

indices = 0, 14, 28, 42
titles = ['A', 'B', 'C', 'D']

set_standard_plot_params()
fig, axes = plt.subplots(2, 2, figsize=(15, 7.5), sharey=True, sharex=True)

ax1, ax2, ax3, ax4 = axes.flat

ax1.set_ylabel('Normalised\ncurrent')
ax3.set_ylabel('Normalised\ncurrent')
ax3.set_xlabel('Voltage [V]')
ax4.set_xlabel('Voltage [V]')


for i, ax, title in zip(indices, [ax1, ax2, ax3, ax4], titles):
    x_test, y_test = test_data_concat[i]

    x_test = x_test.unsqueeze(0)
    y_test = torch.tensor([y_test], dtype=torch.long)

    output = model(x_test)           # shape [1, 4]
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    weights = gradients.mean(dim=2)  # [1, channels]

    cam = (weights[:, :, None] * activations).sum(dim=1)
    cam = torch.relu(cam)

    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    cam = zoom(cam, x_test.shape[-1] / len(cam))

    sc = ax.scatter(
        x_test[0][0],
        x_test[0][1],
        c=cam,
        cmap="viridis",
        s=10,
        vmin=0,
        vmax = 1
    )
    ax.grid(True)
    ax.set_title(title, fontweight='bold', loc='left')
    # ax.text(0,0.4,f'True class: {y_test.item()}\n'
    #                           f'Predicted class: {pred_class.item()}',
    #          fontsize=15)

cbar = fig.colorbar(
    sc,
    ax=axes,
    location='right',
    fraction=0.03
)

cbar.set_label("Importance")

plt.savefig(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results"
            fr"\importance_{file_name}")
# plt.show(block=True)