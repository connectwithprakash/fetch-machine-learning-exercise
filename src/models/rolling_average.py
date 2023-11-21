from typing import List
import numpy as np


class RollingAverage:
    def __init__(self, window_size: int, prediction_size: int):
        """
        Initialize the RollingAverage object.

        Args:
            window_size (int): The size of the rolling window.
            prediction_size (int): The number of predictions to make.
        """
        self.window_size = window_size
        self.prediction_size = prediction_size

    def predict(self, X: List[float]) -> np.ndarray:
        """
        Predict the rolling average of the given input.

        Args:
            X (List[float]): The input data.

        Returns:
            np.ndarray: The array of predicted rolling averages.
        """
        predictions = []
        # Get the last 'window_size' elements from the input data
        window_X = X[-self.window_size:]
        for _ in range(self.prediction_size):  # Repeat 'prediction_size' times
            # Calculate the mean of the current window and add it to the predictions list
            predictions.append(window_X.mean())
            # Shift the window to the left by one position
            window_X = np.roll(window_X, -1)
            # Replace the last element of the window with the latest prediction
            window_X[-1] = predictions[-1]

        return np.asarray(predictions)
