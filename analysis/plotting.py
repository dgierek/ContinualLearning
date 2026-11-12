import matplotlib.pyplot as plt
import torch
import numpy as np

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

def plot_model_testing_accuracies_CL_scenario(runs_history, save_path=''):

    mean_results = {
        key: np.mean([d[key] for d in runs_history], axis=0)
        for key in runs_history[0]
    }

    accs = np.array(list(mean_results.values()))

    x = np.arange(accs.shape[-1])
    width = 0.25

    set_standard_plot_params()
    fig, ax = plt.subplots(figsize=(8, 5))

    for task in range(accs.shape[0]):
        ax.bar(
            x + task * width,
            accs[task, :],
            width,
            label=f"Substance {task}"
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(["After Task 1\nSubstance 0 + 1",
                        "After Task 2\nSubstance 2",
                        "After Task 3\nSubstance 3"])

    ax.set_ylabel("Testing accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(draggable=True, fontsize=12)
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show(block=True)

if __name__ == "__main__":
    # file_name = 'Standard_baseline_100_epochs_dataset_Synthetic_k0_E0'
    #
    # checkpoint = torch.load(
    #     rf"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results\model_{file_name}.pth")
    #
    # history = checkpoint['history']
    #
    # plot_training_history(history)

    runs_history = torch.load(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results\Standard_baseline_None_baseline_runs_history.pt")
    plot_model_testing_accuracies_CL_scenario(runs_history)