import numpy as np


class LinearRegression:
    def __init__(self):
        self.coef = None  # Coefficients of the linear regression model
        self.intercept = None  # Intercept of the linear regression model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear regression model to the given training data.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.

        Returns:
            None
        """
        # Check if the data is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self.coef = np.cov(X, y)[0, 1] / np.var(X)
        self.intercept = np.mean(y) - self.coef * np.mean(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given input features.

        Args:
            X (np.ndarray): The input features.

        Returns:
            np.ndarray: The predicted target values.
        """
        # Check if the data is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return self.coef * X + self.intercept
