import torch
from torch import nn


# Define the simple LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x, h0=None, c0=None):
        # x shape: (batch, seq_len, input_size)
        # h0 shape: (num_layers * num_directions, batch, hidden_size)
        # c0 shape: (num_layers * num_directions, batch, hidden_size)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        # lstm_out shape: (batch, seq_len, hidden_size * 2)
        
        # Take the last time step output from both directions
        lstm_out = lstm_out[:, -1, :]
        
        # # Split the forward and backward outputs
        # forward_out = lstm_out[:, :self.hidden_size]
        # backward_out = lstm_out[:, self.hidden_size:]
        
        # # Concatenate the forward and backward outputs
        # combined = torch.cat((forward_out, backward_out), dim=1)
        
        out = self.fc(lstm_out)
        return out
