import numpy as np

import torch
from pibrary.logger import logger
from torch.utils.data import DataLoader

from src.utils.callbacks import EarlyStopping


class Trainer:
    def __init__(self, model: torch.nn.Module, criterion=None, optimizer=None, learning_rate: float = 0.001, batch_size: int = 32, num_epochs: int = 10, patience: int = 10, device: str = "cpu", verbose: int = 1, **kwargs):
        """
        Initialize the Trainer for training a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            criterion: The loss function (criterion) for training (default: None MSE loss).
            optimizer: The optimizer for training (default: None for Adam optimizer).
            learning_rate (float): The learning rate for the optimizer (default: 0.001).
            batch_size (int): The batch size for training (default: 32).
            num_epochs (int): The number of training epochs (default: 10).
            patience (int): The patience for early stopping (default: 10).
            device (str): The device for training (default: "cpu").
            verbose (int): Verbosity level (default: 1).
            **kwargs: Additional keyword arguments.
        """
        self.device = torch.device(
            device) if torch.cuda.is_available() else torch.device("cpu")
        # Print the device information
        logger.info(f"Using device: {self.device}")
        self.model = model.to(self.device)
        assert isinstance(
            self.model, torch.nn.Module), "Model must be an instance of torch.nn.Module"
        if criterion is None:
            self.criterion = torch.nn.MSELoss(reduction="mean")
            logger.info(
                "No criterion provided. Using MSE loss as default criterion.")
        else:
            self.criterion = criterion
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate)
            logger.info(
                "No optimizer provided. Using Adam optimizer as default optimizer.")
        else:
            self.optimizer = optimizer

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.verbose = verbose
        self.best_model = None

    def optimizer_step(self):
        """
        Perform a single optimization step.

        Returns:
            None.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def training_step(self, batch, batch_idx: int) -> float:
        """
        Perform a training step.

        Args:
            batch: The training batch.
            batch_idx (int): Batch index.

        Returns:
            float: Loss value for the batch.
        """
        X, y = batch
        # Move to the device
        X = X.to(self.device)
        y = y.to(self.device)
        y_pred = self.model.forward(X)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer_step()
        return loss.item()

    def validation_step(self, batch, batch_idx: int) -> float:
        """
        Perform a validation step.

        Args:
            batch: The validation batch.
            batch_idx (int): Batch index.

        Returns:
            float: Loss value for the batch.
        """
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            y_pred = self.model.forward(X)
            loss = self.criterion(y_pred, y)
        return loss.item()

    def predict_step(self, X):
        """
        Perform a prediction step.

        Args:
            X: The input data.

        Returns:
            np.ndarray: The predicted results.
        """
        X = X.to(self.device)
        with torch.no_grad():
            y_pred = self.model.forward(X)
            y_pred = y_pred.exp().detach().cpu().numpy()
        return y_pred

    def train(self, dataloader: DataLoader) -> float:
        """
        Train the model.

        Args:
            dataloader: The training data loader.

        Returns:
            float: The epoch loss.
        """
        self.model.train()
        n_iters = len(dataloader)
        epoch_loss = 0
        self.optimizer.zero_grad()
        for idx, batch in enumerate(dataloader):
            loss = self.training_step(batch, idx)
            epoch_loss += (loss/n_iters)
        return epoch_loss

    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            dataloader: The validation data loader.

        Returns:
            float: The epoch loss.
        """
        self.model.eval()
        n_iters = len(dataloader)
        epoch_loss = 0
        for idx, batch in enumerate(dataloader):
            loss = self.validation_step(batch, idx)
            epoch_loss += (loss/n_iters)
        return epoch_loss

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        """
        Create a data loader for a dataset.

        Args:
            dataset: The dataset.
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: The data loader.
        """
        # Implement data loader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def fit(self, train_dataset, val_dataset=None):
        """
        Fit the model to the training data.

        Args:
            train_dataset: The training dataset.
            val_dataset: The validation dataset (default: None).

        Returns:
            torch.nn.Module: The trained model.
        """
        # Implement training loop
        train_dataloader = self._create_dataloader(
            train_dataset, self.batch_size, True)

        if val_dataset is not None:
            val_dataloader = self._create_dataloader(
                val_dataset, self.batch_size, False)
        else:
            val_dataloader = None

        early_stopping = EarlyStopping(
            patience=self.patience, verbose=self.verbose)

        train_losses = []
        val_losses = []
        for epoch in range(self.num_epochs):
            # Train the model
            train_loss = self.train(train_dataloader)
            train_losses.append(train_loss)
            # Validate the model
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                val_losses.append(val_loss)
            else:
                val_loss = np.nan

            if self.verbose:
                epoch_log = f"Epoch: [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss}"
                if val_dataloader is not None:
                    epoch_log += f", Val Loss: {val_loss}"
                logger.info(epoch_log)

            # Early stopping
            early_stop = early_stopping(val_loss, self.model)
            if early_stop:
                logger.warning(
                    "Stopping training early. Training is complete.")
                self.model = self.best_model
                break
            else:
                self.best_model = self.model

        return self.model, train_losses, val_losses

    def predict(self, test_dataset):
        """
        Make predictions on a test dataset.

        Args:
            test_dataset: The test dataset.

        Returns:
            np.ndarray: Predicted results.
        """
        # Implement prediction loop
        dataloader = self._create_dataloader(
            test_dataset, self.batch_size, False)

        self.model.eval()
        results = []
        for batch in dataloader:
            X = batch
            X = X.to(self.device)
            y_pred = self.predict_step(X)
            results.extend(y_pred)

        return np.array(results)
