"""
temporal_lstm.py — Temporal LSTM module.

Takes the sequence of per-frame spatial descriptors produced by the
SpatialTransformerEncoder and models temporal dynamics across T frames.

Input  shape: [B, T, d_model]
Output shape: [B, T, lstm_hidden]
"""

import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    """
    Multi-layer LSTM that captures temporal dependencies across frames.

    Args:
        input_size  (int): Feature dimension of each frame token (= d_model).
        hidden_size (int): LSTM hidden state dimension.
        num_layers  (int): Number of stacked LSTM layers.
        dropout     (float): Dropout between LSTM layers (0 for num_layers=1).

    Input  shape: [B, T, input_size]
    Output shape: [B, T, hidden_size]
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # dropout only applied between layers, so set to 0 when num_layers==1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Frame descriptors of shape [B, T, input_size].

        Returns:
            Hidden states for all frames, shape [B, T, hidden_size].
        """
        # hidden state and cell state are initialised to zeros by default
        out, _ = self.lstm(x)      # out: [B, T, hidden_size]
        return out
