"""
spatial_transformer.py — Spatial Transformer Encoder.

Processes all joints within a single frame jointly using multi-head
self-attention.  Each time-step is processed independently so the spatial
module is unaware of temporal order — that is handled by the LSTM.

Architecture per frame:
  [B*T, J, d_model] → TransformerEncoder (2 layers, 4 heads)
                     → mean-pool over joints
                     → [B*T, d_model]
"""

import torch
import torch.nn as nn


class SpatialTransformerEncoder(nn.Module):
    """
    Applies a Transformer encoder over the J joint tokens for every frame.

    The time dimension is folded into the batch dimension so that each
    (batch, frame) pair is processed as an independent set of J tokens.
    After encoding, joint tokens are pooled via mean to produce a single
    frame-level descriptor.

    Args:
        d_model         (int): Token embedding dimension.
        nhead           (int): Number of attention heads.
        num_layers      (int): Number of stacked TransformerEncoderLayer blocks.
        dim_feedforward (int): Hidden size of the FFN inside each layer.
        dropout         (float): Dropout probability.

    Input  shape: [B, T, J, d_model]
    Output shape: [B, T, d_model]
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # input: [batch, seq, d_model]
            norm_first=False,    # post-norm (standard)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Joint embeddings of shape [B, T, J, d_model].

        Returns:
            Frame-level descriptors of shape [B, T, d_model].
        """
        B, T, J, d = x.shape                   # [B, T, J, d_model]

        # Fold time into batch so attention runs over the J joint tokens
        x = x.reshape(B * T, J, d)             # [B*T, J, d_model]

        # Self-attention over joint tokens within each frame
        x = self.transformer(x)                # [B*T, J, d_model]

        # Mean-pool over joint tokens → single frame descriptor
        x = x.mean(dim=1)                      # [B*T, d_model]

        # Restore time dimension
        x = x.reshape(B, T, d)                 # [B, T, d_model]
        return x
