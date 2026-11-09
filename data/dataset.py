import torch


class CVDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.curves = data["curves"]
        self.labels = data["labels"]
        self.label_map = data["label_map"]

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        x = self.curves[idx]      # shape (channels, length)
        y = self.labels[idx]      # integer label
        return x, y


def load_task(path):
    print(f"Loading: {path}")
    return CVDataset(path)

def load_training_data(experimental_data):
    if experimental_data:
      substance0_train = load_task(
        fr"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experimental_joint_datasets\Co(phen)3_joint_train.pt")
      substance1_train = load_task(
        fr"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experimental_joint_datasets\FcMeOH_joint_train.pt")
      substance2_train = load_task(
        fr"CC:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experimental_joint_datasets\FeCl2_joint_train.pt")
      substance3_train = load_task(
        fr"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experimental_joint_datasets\Syringaldazine_joint_train.pt")

      substance0_test, substance1_test, substance2_test, substance3_test = load_experimental_test_data()

    else:
      substance0_train = load_task(
          r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance0_joint_train.pt")
      substance1_train = load_task(
          r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance1_joint_train.pt")
      substance2_train = load_task(
          r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance2_joint_train.pt")
      substance3_train = load_task(
          r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance3_joint_train.pt")

      substance0_test, substance1_test, substance2_test, substance3_test = load_synthetic_test_data()

    return (substance0_train, substance1_train, substance2_train, substance3_train,
            substance0_test, substance1_test, substance2_test, substance3_test)

def load_synthetic_test_data():
    substance0_test = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance0_joint_test.pt")
    substance1_test = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance1_joint_test.pt")
    substance2_test = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance2_joint_test.pt")
    substance3_test = load_task(
        r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\joint_datasets\substance3_joint_test.pt")

    return substance0_test, substance1_test, substance2_test, substance3_test

def load_experimental_test_data():
    substance0_test = load_task(
      fr"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experimental_joint_datasets\Co(phen)3_joint_test.pt")
    substance1_test = load_task(
      fr"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experimental_joint_datasets\FcMeOH_joint_test.pt")
    substance2_test = load_task(
      fr"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experimental_joint_datasets\FeCl2_joint_test.pt")
    substance3_test = load_task(
      fr"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\experimental_joint_datasets\Syringaldazine_joint_test.pt")

    return substance0_test, substance1_test, substance2_test, substance3_test