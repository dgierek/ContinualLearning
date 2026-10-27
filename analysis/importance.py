import torch
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import CVDataset


def integrated_gradients(model, x, target_class=None, steps=100, device='cpu'):
    model.eval()

    # x shape: (1, 2, 1001)
    x = x.to(device)

    # Baseline: zero CV
    baseline = torch.zeros_like(x).to(device)

    # If target_class not provided, use predicted class
    out = model(x)
    if target_class is None:
      target_class = out.argmax(dim=1).item()

    # Generate scaled inputs
    scaled_inputs = [
      baseline + (float(i) / steps) * (x - baseline)
      for i in range(steps + 1)
    ]

    grads = []

    for s_inp in scaled_inputs:
      s_inp.requires_grad_(True)
      out = model(s_inp)
      model.zero_grad()
      out[0, target_class].backward()
      grads.append(s_inp.grad.detach().cpu().numpy())

    grads = np.array(grads)  # shape: (steps+1, 1, 2, 1001)

    # Average gradients along the path
    avg_grads = grads.mean(axis=0)[0]  # shape: (2, 1001)

    # Integrated gradients
    ig = (x.detach().cpu().numpy()[0] - baseline.cpu().numpy()[0]) * avg_grads

    # We take importance from the current channel (index 1)
    importance = np.abs(ig[1])

    return importance


def load_task(path):
    print(f"Loading: {path}")
    return CVDataset(path)


if __name__ == "__main__":
    importance_maps = np.load(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\analysis\importance_maps_100runs.npy")

    # Compute mean and std
    mean_importance = importance_maps.mean(axis=0)
    std_importance = importance_maps.std(axis=0)

    Num = 1

    substance_train = load_task(
      rf"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance{Num}_joint_train.pt")

    sample_x, _ = substance_train[0]

    voltage = sample_x[0].numpy()
    current = sample_x[1].numpy()

    # Normalize importance for colormap
    imp_norm = (mean_importance - mean_importance.min()) / (mean_importance.max() - mean_importance.min())

    # # Plot
    # plt.figure(figsize=(12, 5))
    # plt.plot(mean_importance, label="Mean importance")
    # plt.fill_between(
    #   np.arange(len(mean_importance)),
    #   mean_importance - std_importance,
    #   mean_importance + std_importance,
    #   alpha=0.3,
    #   label="±1 std"
    # )
    # plt.legend()
    # plt.title("Averaged Importance Map over 100 runs")
    # plt.show()

    voltage_idx = np.arange(voltage.shape[0])


    plt.figure(figsize=(10, 6))
    cmap = plt.cm.inferno

    # Draw colored line segments using REAL voltage
    for i in range(len(voltage) - 1):
      plt.scatter(
        voltage[i:i + 2],  # <-- voltage instead of index
        current[i:i + 2],
        color=cmap(imp_norm[i]),
        s=10
      )

    # Colorbar
    ax = plt.gca()
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(imp_norm)
    plt.colorbar(sm, ax=ax, label='Normalised importance')

    plt.xlabel("Normalised Voltage")
    plt.ylabel("Normalised Current")
    plt.title(f"CV with Integrated Gradients Importance Overlay for Substance {Num}")
    plt.show()