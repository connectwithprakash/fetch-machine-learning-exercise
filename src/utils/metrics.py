import numpy as np


class RegressionMetrics:
    def __init__(self, target: np.ndarray, prediction: np.ndarray):
        """
        Initialize RegressionMetrics class.

        Args:
            target (np.ndarray): The target values. Shape: (batch_size, length)
            prediction (np.ndarray): The predicted values. Shape: (batch_size, length)
        """
        self.target = target
        self.prediction = prediction

        if not isinstance(target, np.ndarray):
            self.target = np.array(self.target)
        if not isinstance(prediction, np.ndarray):
            self.prediction = np.array(self.prediction)

    def mean_absolute_error(self) -> float:
        """
        Calculate the mean absolute error (MAE).

        Returns:
            float: The mean absolute error.
        """
        batch_mae = np.mean(np.abs(self.target - self.prediction), axis=1)
        mae = np.mean(batch_mae).round(4)
        return mae

    def mean_squared_error(self) -> float:
        """
        Calculate the mean squared error (MSE).

        Returns:
            float: The mean squared error.
        """
        batch_mse = np.mean(np.square(self.target - self.prediction), axis=1)
        mse = np.mean(batch_mse).round(4)
        return mse

    def root_mean_squared_error(self) -> float:
        """
        Calculate the root mean squared error (RMSE).

        Returns:
            float: The root mean squared error.
        """
        batch_rmse = np.sqrt(
            np.mean(np.square(self.target - self.prediction), axis=1))
        rmse = np.mean(batch_rmse).round(4)
        return rmse

    def all(self) -> dict:
        """
        Calculate all regression metrics.

        Returns:
            dict: A dictionary containing all regression metrics.
        """
        return {
            "mae": self.mean_absolute_error(),
            "mse": self.mean_squared_error(),
            "rmse": self.root_mean_squared_error(),
        }
