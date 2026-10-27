import torch
import matplotlib.pyplot as plt

def plot_sample_cvs(pt_paths, labels=None, title="Sample CVs from Experimental Substances"):
    """
    pt_paths: list of paths to substance*_exp_train.pt files
    labels: optional list of names to display in legend
    """

    plt.figure(figsize=(10, 6))

    for i, path in enumerate(pt_paths):
        data = torch.load(path)
        curves = data["curves"]

        # Pick the first curve (or any index you want)
        curve = curves[5].cpu().numpy()

        voltage = curve[0]
        current = curve[1]

        label = labels[i] if labels else f"Substance {i}"
        plt.plot(voltage, current, label=label)

    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

pt_paths = [
    "experimental_joint_datasets/substance0_exp_train.pt",
    "experimental_joint_datasets/substance1_exp_train.pt",
    "experimental_joint_datasets/substance2_exp_train.pt",
    "experimental_joint_datasets/substance3_exp_train.pt"
]

labels = ["Co(phen)3", "FcMeOH", "FeCl2", "Syringaldazine"]

plot_sample_cvs(pt_paths, labels)
