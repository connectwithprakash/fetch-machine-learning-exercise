import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        # Check if the data is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self.coef = np.cov(X, y)[0, 1] / np.var(X)
        self.intercept = np.mean(y) - self.coef * np.mean(X)

    def predict(self, X):
        # Check if the data is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return self.coef * X + self.intercept
