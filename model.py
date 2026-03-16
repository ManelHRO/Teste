# model.py
from __future__ import annotations
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.2, fc_size: int = 64, out_size: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, fc_size)
        self.act = nn.ReLU()
        self.fc_out = nn.Linear(fc_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        z = self.act(self.fc1(last))
        return self.fc_out(z)
