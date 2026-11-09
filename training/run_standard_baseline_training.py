import torch
from torch.utils.data import ConcatDataset

from analysis.plotting import plot_training_history
from train_standard import train_standard
from data.dataset import load_training_data

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    experimental_data = False
    epochs=150

    (substance0_train, substance1_train, substance2_train, substance3_train,
    substance0_test, substance1_test, substance2_test, substance3_test) = load_training_data(experimental_data)

    training_dataset = ConcatDataset([substance0_train, substance1_train, substance2_train, substance3_train])
    test_dataset = ConcatDataset([substance0_test, substance1_test, substance2_test, substance3_test])

    num_classes = 4
    sample_x, _ = substance0_train[0]
    in_channels = sample_x.shape[0]

    model, history = train_standard(
        train_dataset=training_dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
        in_channels=in_channels,
        epochs=epochs,
        lr=5e-4
    )

    plot_training_history(history, save_path=r'C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results'
                                             fr'\loss_accuracy_plots_Standard_baseline_{epochs}_epochs')

    checkpoint = {
        "model_state_dict": model.state_dict(),
        'history': history
    }

    torch.save(checkpoint, r'C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results'
                                             fr'\model_Standard_baseline_{epochs}_epochs.pth')

