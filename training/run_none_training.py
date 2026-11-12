import torch
from torch.utils.data import ConcatDataset

from data.dataset import CVDataset, load_synthetic_k0_E0_data
from training.train_none import train_none
from analysis.plotting import plot_model_testing_accuracies_CL_scenario


if __name__ == "__main__":

    (substance0_train, substance1_train, substance2_train, substance3_train,
    substance0_test, substance1_test, substance2_test, substance3_test) = load_synthetic_k0_E0_data()

    training_dataset = ConcatDataset([substance0_train, substance1_train, substance2_train, substance3_train])
    test_dataset = ConcatDataset([substance0_test, substance1_test, substance2_test, substance3_test])

    context_datasets = [
        ConcatDataset([substance0_train, substance1_train]),
        substance2_train,
        substance3_train
    ]

    test_datasets = [
        substance0_test,
        substance1_test,
        substance2_test,
        substance3_test
    ]

    num_classes = 4
    sample_x, _ = substance0_train[0]
    in_channels = sample_x.shape[0]

    # -----------------------------
    # Run NONE baseline N times
    # -----------------------------
    N_RUNS = 100

    runs_history = []

    for run in range(N_RUNS):
        print(f"\n=== NONE baseline run {run+1}/{N_RUNS} ===")

        model, history = train_none(
            context_datasets=context_datasets,
            test_datasets=test_datasets,
            num_classes=num_classes,
            in_channels=in_channels,
            epochs=10,
            lr=5e-4
        )

        runs_history.append(history)

    torch.save(runs_history,
               r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results\None_baseline_runs_history_E0_k0.pt")
    plot_model_testing_accuracies_CL_scenario(runs_history,
                                              save_path=r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results\None_baseline_E0_k0")