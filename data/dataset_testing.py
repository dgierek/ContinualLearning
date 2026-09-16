import numpy as np

from data.dataset import CVDataset


def print_label_stats(dataset, name):
  labels = [y for _, y in dataset]
  unique, counts = np.unique(labels, return_counts=True)
  print(f"{name}: {dict(zip(unique, counts))}")

substance0 =\
  CVDataset(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\data_45_CVs_scan_rates\substance0.pt")
substance1 =\
  CVDataset(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\data_45_CVs_scan_rates\substance1.pt")
substance2 =\
  CVDataset(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\data_45_CVs_scan_rates\substance2.pt")
substance3 =\
  CVDataset(r"C:\Users\Public\Desktop\Praktyki IChF\ContinualLearning\data\data_45_CVs_scan_rates\substance3.pt")



print_label_stats(substance0, "substance0")
print_label_stats(substance1, "substance1")
print_label_stats(substance2, "substance2")
print_label_stats(substance3, "substance3")

for i in range(10):
    print("sub0:", substance0[i][1])
    print("sub1:", substance1[i][1])


