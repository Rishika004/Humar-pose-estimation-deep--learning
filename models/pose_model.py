"""
pose_model.py — Full 3D Human Pose Estimation Model.

Architecture:
    Input [B, T, J, 3]
        ↓  JointEmbedding
    [B, T, J, d_model]
        ↓  SpatialTransformerEncoder  (processes each frame independently)
    [B, T, d_model]
        ↓  TemporalLSTM  (models motion across T frames)
    [B, T, lstm_hidden]
        ↓  Linear head
    [B, T, J, 3]  (or [B, J, 3] when predict_last_frame_only=True)
"""

import torch
import torch.nn as nn

from .embedding import JointEmbedding
from .spatial_transformer import SpatialTransformerEncoder
from .temporal_lstm import TemporalLSTM


class Pose3DModel(nn.Module):
    """
    End-to-end 3D pose estimation model combining spatial attention and
    temporal sequence modelling.

    Args:
        d_model                (int): Embedding and transformer hidden dim.
        nhead                  (int): Attention heads in spatial transformer.
        num_transformer_layers (int): Stacked spatial transformer layers.
        dim_feedforward        (int): FFN width inside each transformer layer.
        lstm_hidden            (int): LSTM hidden state size.
        lstm_layers            (int): Number of LSTM layers.
        dropout                (float): Dropout probability.
        num_joints             (int): Number of skeleton joints (J).
        joint_dim              (int): Input coordinate dimension (3 for xyz).
        predict_last_frame_only (bool): If True, return only the last frame.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        num_joints: int = 17,
        joint_dim: int = 3,
        predict_last_frame_only: bool = False,
    ):
        super().__init__()
        self.predict_last_frame_only = predict_last_frame_only
        self.num_joints = num_joints
        self.joint_dim = joint_dim

        # ── Stage 1: embed raw joint coordinates ──────────────────────────
        self.embedding = JointEmbedding(
            num_joints=num_joints,
            joint_dim=joint_dim,
            d_model=d_model,
        )

        # ── Stage 2: spatial self-attention per frame ──────────────────────
        self.spatial_encoder = SpatialTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # ── Stage 3: temporal LSTM across frames ───────────────────────────
        self.temporal_lstm = TemporalLSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        # ── Stage 4: regress 3D joint positions from LSTM output ───────────
        self.head = nn.Linear(lstm_hidden, num_joints * joint_dim)

        # Print parameter count on init for quick sanity check
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Pose3DModel] Trainable parameters: {total_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input joint coordinates, shape [B, T, J, 3].

        Returns:
            Predicted joint positions:
              - [B, T, J, 3] when predict_last_frame_only=False
              - [B,    J, 3] when predict_last_frame_only=True
        """
        B, T, J, _ = x.shape                   # [B, T, J, 3]

        # ── Stage 1: joint embedding ───────────────────────────────────────
        x = self.embedding(x)                   # [B, T, J, d_model]

        # ── Stage 2: spatial transformer (per-frame joint attention) ───────
        x = self.spatial_encoder(x)             # [B, T, d_model]

        # ── Stage 3: temporal LSTM (motion across T frames) ────────────────
        x = self.temporal_lstm(x)               # [B, T, lstm_hidden]

        # ── Stage 4: linear regression head ────────────────────────────────
        x = self.head(x)                        # [B, T, J*3]
        x = x.reshape(B, T, J, self.joint_dim)  # [B, T, J, 3]

        if self.predict_last_frame_only:
            return x[:, -1]                     # [B, J, 3]
        return x                                # [B, T, J, 3]
