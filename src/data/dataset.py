from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Create pytorch datasets
class ReceiptCountDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int, prediction_size: int):
        """
        Initialize the ReceiptCountDataset class.

        Args:
            df (pd.DataFrame): The input dataframe.
            window_size (int): The size of the input window.
            prediction_size (int): The size of the prediction window.
        """
        self.window_size = window_size
        self.prediction_size = prediction_size
        self.df = df

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.df) - self.window_size - self.prediction_size + 1

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: The input and output tensors.
        """
        # Get the input window
        X = self.df.iloc[idx:idx + self.window_size, :].values

        # Get the output window
        y = self.df.iloc[idx + self.window_size:idx +
                         self.window_size + self.prediction_size, -1].values

        # Convert the input and output windows to numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Convert the input and output windows to PyTorch tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        return X, y
