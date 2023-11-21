import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): The number of expected features in the input x
            hidden_size (int): The number of features in the hidden state h
            num_layers (int): Number of recurrent layers
            output_size (int): The number of output features
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None, c0: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)
            h0 (torch.Tensor, optional): Initial hidden state tensor of shape (num_layers * num_directions, batch, hidden_size)
            c0 (torch.Tensor, optional): Initial cell state tensor of shape (num_layers * num_directions, batch, hidden_size)

        Returns:
            torch.Tensor: Output tensor of shape (batch, output_size)
        """
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out
