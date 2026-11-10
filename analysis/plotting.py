import matplotlib.pyplot as plt
import torch

def set_standard_plot_params():
    plt.rcParams['axes.labelsize'] = 18  # X/Y axis label size
    plt.rcParams['axes.titlesize'] = 20  # Title size
    plt.rcParams['xtick.labelsize'] = 16  # X tick labels
    plt.rcParams['ytick.labelsize'] = 16  # Y tick labels
    plt.rcParams['legend.fontsize'] = 16  # Legend text
    plt.rcParams['figure.titlesize'] = 18  # Title text

    plt.rcParams.update({'font.size': 20})

def plot_training_history(history, save_path = '', logscale=False):
    set_standard_plot_params()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.5))

    ax1.plot(history['training_loss'], label='Training')
    ax1.plot(history['testing_loss'], label='Testing')
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss function')
    ax1.set_title('A', fontweight='bold', loc='left')
    ax1.grid(True)

    ax2.plot(history['training_accuracy'], label='Training')
    ax2.plot(history['testing_accuracy'], label='Testing')
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Discriminative accuracy')
    ax2.set_title('B', fontweight='bold', loc='left')
    ax2.grid(True)

    if logscale:
        ax1.set_yscale('log')
        ax2.set_yscale('log')

    plt.subplots_adjust(wspace=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.show(block=True)

if __name__ == "__main__":
    file_name = 'Standard_baseline_100_epochs_dataset_Synthetic_k0_E0'

    checkpoint = torch.load(
        rf"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results\model_{file_name}.pth")

    history = checkpoint['history']

    plot_training_history(history)