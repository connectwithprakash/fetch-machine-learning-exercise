import numpy as np

# Min max scaling of the receipt counts
class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = X.min()
        self.max = X.max()

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def inverse_transform(self, X):
        # Check if the data is a numpy array
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        return X * (self.max - self.min) + self.min
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
