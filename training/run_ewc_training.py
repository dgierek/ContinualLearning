import torch
from torch import nn
from torch.utils.data import ConcatDataset

from analysis.plotting import plot_model_testing_accuracies_CL_scenario
from data.dataset import load_synthetic_k0_E0_data
from training.train_ewc import train_ewc
from tqdm import tqdm


if __name__ == "__main__":

    lambda_ewc = 1e8
    N_RUNS = 1
    # -----------------------------
    # Load susbstance data
    # -----------------------------


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

    # Number of classes = 4 substances
    num_classes = 4

    # Determine number of input channels
    sample_x, _ = substance0_train[0]
    in_channels = sample_x.shape[0]   # should be 2

    runs_history = []

    for i in tqdm(range(N_RUNS)):

        model, history = train_ewc(
            context_datasets=context_datasets,
            test_datasets=test_datasets,
            num_classes=num_classes,
            in_channels=in_channels,
            epochs=40,
            lr=5e-4,
            criterion=nn.CrossEntropyLoss(),
            lambda_ewc=lambda_ewc
        )

        runs_history.append(history)

    mean_results_over_time = {}

    exponent = len(str(abs(int(lambda_ewc)))) - 1

    torch.save(runs_history,
               rf"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results\EWC_baseline_runs_history_E0_k0_lambda_e{exponent}_test.pt")
    plot_model_testing_accuracies_CL_scenario(runs_history,
                                              save_path=fr"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\training\results\EWC_baseline_E0_k0_lambda_e{exponent}_test")