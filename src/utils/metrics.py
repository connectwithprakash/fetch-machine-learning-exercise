import numpy as np


# Computes the regression metrics for the given target and prediction and averages them over the batch
class RegressionMetrics:
    def __init__(self, target, prediction):
        self.target = target  # Shape: (batch_size, length)
        self.prediction = prediction  # Shape: (batch_size, length)
        if not isinstance(target, np.ndarray):
            self.target = np.array(self.target)
        if not isinstance(prediction, np.ndarray):
            self.prediction = np.array(self.prediction)

    def mean_absolute_error(self):
        batch_mae = np.mean(np.abs(self.target - self.prediction), axis=1)
        mae = np.mean(batch_mae).round(4)
        return mae

    def mean_squared_error(self):
        batch_mse = np.mean(np.square(self.target - self.prediction), axis=1)
        mse = np.mean(batch_mse).round(4)
        return mse

    def root_mean_squared_error(self):
        batch_rmse = np.sqrt(
            np.mean(np.square(self.target - self.prediction), axis=1))
        rmse = np.mean(batch_rmse).round(4)
        return rmse

    def all(self):
        return {
            "mae": self.mean_absolute_error(),
            "mse": self.mean_squared_error(),
            "rmse": self.root_mean_squared_error(),
        }
