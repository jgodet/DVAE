import pandas as pd
import torch
from torch.utils.data import Dataset

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None, do_fillna=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if do_fillna :
            self.data.fillna(0, inplace=True) 

        print("Name of the last column:", self.data.columns[-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract features (data) and labels from the DataFrame
        features = self.data.iloc[idx, :-1].values  # Use .values to get numpy array
        label = self.data.iloc[idx, -1]  # Assuming labels are in the last column
        # Convert features and labels to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        # Apply any transformations to features
        if self.transform:
            features = self.transform(features)
        return features, label





