from typing import Any, Union

import torch
import numpy as np


class Ensemble:
    def __init__(self, lr_model: Any, nn_model: Any, window_size: int):
        """
        Initialize the Ensemble model.

        Args:
            lr_model (Any): The linear regression model.
            nn_model (Any): The neural network model.
            window_size (int): The window size for prediction.
        """
        self.window_size = window_size
        self.lr_model = lr_model
        self.nn_model = nn_model
        self.nn_model.eval()

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions using the ensemble model.

        Args:
            X (Union[np.ndarray, torch.Tensor]): The input data.

        Returns:
            np.ndarray: The predictions.
        """
        time = X[:, 0] + self.window_size
        lr_pred = self.lr_model.predict(time)
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X).float()
            nn_pred = self.nn_model(X.unsqueeze(0))
            nn_pred = nn_pred.detach().cpu().numpy()[0]

        pred = lr_pred + nn_pred
        return pred

    def __repr__(self) -> str:
        """
        Return a string representation of the ensemble model.

        Returns:
            str: The string representation.
        """
        return f'Ensemble(models={self.lr_model}, {self.nn_model})'
