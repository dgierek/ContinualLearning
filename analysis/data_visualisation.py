import torch
import matplotlib.pyplot as plt
import os

plt.rcParams['axes.labelsize'] = 18      # X/Y axis label size
plt.rcParams['axes.titlesize'] = 20      # Title size
plt.rcParams['xtick.labelsize'] = 16     # X tick labels
plt.rcParams['ytick.labelsize'] = 16     # Y tick labels
plt.rcParams['legend.fontsize'] = 16     # Legend text
plt.rcParams['figure.titlesize'] = 18    # Title text

plt.rcParams.update({'font.size': 20})

DATA_DIR = r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\data_45_CVs_scan_rates"

def load_substance(path):
    data = torch.load(path)
    curves = data["curves"]          # list of (2, 1001) tensors
    return curves

def pick_one_example(curves):
    """Pick the first example (or random if you prefer)."""
    return curves[0].cpu().numpy()   # shape (2, 1001)

def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pt")])

    plt.figure(figsize=(10, 6))

    for f in files:
        # Extract substance number from filename, e.g. "substance3.pt"
        substance_id = ''.join(filter(str.isdigit, f)) or f

        curves = load_substance(os.path.join(DATA_DIR, f))
        cv = pick_one_example(curves)

        voltage = cv[0]
        current = cv[1] * 1e6

        label = f"Substance {substance_id}"

        plt.plot(voltage, current, label=label)

    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [$\mu$A]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
