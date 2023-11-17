import numpy as np
import torch
from torch.utils.data import Dataset


# Create pytorch datasets
class ReceiptCountDataset(Dataset):
    def __init__(self, df, window_size, prediction_size):
        self.window_size = window_size
        self.prediction_size = prediction_size
        self.df = df

    def __len__(self):
        return len(self.df) - self.window_size - self.prediction_size + 1

    def __getitem__(self, idx):
        # print(f"index: {idx}")
        X = self.df.iloc[idx:idx + self.window_size, :].values
        y = self.df.iloc[idx + self.window_size:idx + self.window_size + self.prediction_size, -1].values

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Make features such as month_of_year, quarter, etc. cyclic
        # The features are day_of_week', 'day_of_month', 'day_of_year', 'week_of_year','month_of_year', 'quarter', 'days_in_month'
        # # Convert to sine and cosine components
        # X = np.concatenate([
        #     np.sin(X[:, 0:1] * (2 * np.pi / 7)),
        #     np.cos(X[:, 0:1] * (2 * np.pi / 7)),
        #     np.sin(X[:, 1:2] * (2 * np.pi / 31)),
        #     np.cos(X[:, 1:2] * (2 * np.pi / 31)),
        #     np.sin(X[:, 2:3] * (2 * np.pi / 366)),
        #     np.cos(X[:, 2:3] * (2 * np.pi / 366)),
        #     np.sin(X[:, 3:4] * (2 * np.pi / 52)),
        #     np.cos(X[:, 3:4] * (2 * np.pi / 52)),
        #     np.sin(X[:, 4:5] * (2 * np.pi / 12)),
        #     np.cos(X[:, 4:5] * (2 * np.pi / 12)),
        #     np.sin(X[:, 5:6] * (2 * np.pi / 4)),
        #     np.cos(X[:, 5:6] * (2 * np.pi / 4)),
        #     np.sin(X[:, 6:7] * (2 * np.pi / 31)),
        #     np.cos(X[:, 6:7] * (2 * np.pi / 31)),
        #     X[:, 7:8]
        # ], axis=1)

        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        # Print shapes
        # print(f"X shape: {X.shape}")
        # print(f"y shape: {y.shape}")

        return X, y
