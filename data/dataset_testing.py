from data.dataset import CVDataset
from torch.utils.data import DataLoader
from data.cvs_loader import load_cv_dataset
from data.preprocessing import preprocess_dataframe_and_save

df_rows = load_cv_dataset('data/test_data')

preprocess_dataframe_and_save(df_rows, 'data/pandas_converted_data/test_data.pt')



dataset = CVDataset("data/pandas_converted_data/test_data.pt")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

x, y = next(iter(loader))
print(x.shape)
print(y)


