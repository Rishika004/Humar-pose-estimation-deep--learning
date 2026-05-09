"""
embedding.py — Joint embedding layer.

Projects raw 3-D joint coordinates into a higher-dimensional representation
and adds a learnable per-joint-type positional embedding so the model can
distinguish which body part each token belongs to.
"""

import torch
import torch.nn as nn


class JointEmbedding(nn.Module):
    """
    Embed raw joint coordinates into a d_model-dimensional space.

    Two components are summed:
    1. A linear projection of the 3-D coordinate vector.
    2. A learnable joint-type embedding (one vector per joint index).

    Args:
        num_joints (int): Number of skeleton joints J.
        joint_dim  (int): Input coordinate dimension (3 for x, y, z).
        d_model    (int): Output embedding dimension.

    Input  shape: [B, T, J, joint_dim]
    Output shape: [B, T, J, d_model]
    """

    def __init__(self, num_joints: int = 17, joint_dim: int = 3, d_model: int = 128):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model

        # Linear projection: maps 3-D coords → d_model for every joint token
        self.coord_proj = nn.Linear(joint_dim, d_model)

        # Learnable joint-type table: one d_model vector per joint index
        # shape [J, d_model] — broadcast over the batch and time dimensions
        self.joint_type_emb = nn.Embedding(num_joints, d_model)

        # Joint index buffer — registered so it moves to the right device
        joint_ids = torch.arange(num_joints)           # [J]
        self.register_buffer("joint_ids", joint_ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Joint coordinates of shape [B, T, J, joint_dim].

        Returns:
            Embeddings of shape [B, T, J, d_model].
        """
        # ── coordinate projection ──────────────────────────────────────────
        coord_emb = self.coord_proj(x)          # [B, T, J, d_model]

        # ── joint-type embedding ───────────────────────────────────────────
        type_emb = self.joint_type_emb(self.joint_ids)  # [J, d_model]
        # Broadcast: add type embedding to every (batch, time) position
        type_emb = type_emb.unsqueeze(0).unsqueeze(0)   # [1, 1, J, d_model]

        return coord_emb + type_emb             # [B, T, J, d_model]
