from typing import Any
import numpy as np

class RollingAverage:
    def __init__(self, window_size, prediction_size):
        self.window_size = window_size
        self.prediction_size = prediction_size

    def predict(self, X):
        predictions = []
        window_X = X[-self.window_size:]
        for i in range(self.prediction_size):
            predictions.append(window_X.mean())
            window_X = np.roll(window_X, -1)
            window_X[-1] = predictions[-1]

        return np.asarray(predictions)
