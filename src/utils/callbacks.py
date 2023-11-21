import numpy as np
import torch

from pibrary.logger import logger


class EarlyStopping:
    """
    EarlyStopping class to stop training when validation loss stops improving.
    """

    def __init__(self, patience: int = 10, verbose: bool = False) -> None:
        """
        Initializes the EarlyStopping class.

        Args:
            patience (int, optional): The number of epochs to wait for improvement. Defaults to 10.
            verbose (bool, optional): Whether to print additional information. Defaults to False.

        Raises:
            ValueError: If patience is not greater than zero.
        """
        assert patience > 0, "Patience must be greater than zero"
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.best_val_loss = np.Inf
        self.verbose = verbose

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Call function of the EarlyStopping class.

        Args:
            val_loss (float): The validation loss.
            model (torch.nn.Module): The model to save.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if val_loss > self.best_val_loss:
            self.counter += 1
            if self.verbose:
                logger.warning(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}.")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.verbose:
                logger.success(
                    f"Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f})."
                )

        return self.early_stop
