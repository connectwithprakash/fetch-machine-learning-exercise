import numpy as np


class MinMaxScaler:
    """
    Min-max scaling of the receipt counts.

    Attributes:
        min (float): The minimum value of the data.
        max (float): The maximum value of the data.
    """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the scaler to the data.

        Args:
            X (np.ndarray): The input data.
        """
        self.min = X.min()
        self.max = X.max()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using min-max scaling.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The scaled data.
        """
        return (X - self.min) / (self.max - self.min)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform the scaled data to the original scale.

        Args:
            X (np.ndarray): The scaled data.

        Returns:
            np.ndarray: The data in the original scale.
        """
        # Check if the data is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        return X * (self.max - self.min) + self.min

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the scaler to the data and transform it.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The scaled data.
        """
        self.fit(X)
        return self.transform(X)
