"""
metrics.py — Pose estimation evaluation metrics.

Implements:
  - MPJPE  : Mean Per-Joint Position Error (mm)
  - P-MPJPE: Procrustes-aligned MPJPE (mm)
  - Per-joint MPJPE breakdown across all 17 joints

All numpy-based functions accept arrays of shape [N, T, J, 3] or [N, J, 3].
The torch-based `mpjpe` function is used as the training loss.
"""

from __future__ import annotations

import numpy as np
import torch


# ── Training loss (PyTorch) ───────────────────────────────────────────────────

def mpjpe(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Per-Joint Position Error — used as the training loss.

    Computes the mean Euclidean distance across all joints, frames, and
    samples in the batch.

    Args:
        pred   : Predicted joints,      shape [..., J, 3].
        target : Ground-truth joints,   shape [..., J, 3].

    Returns:
        Scalar loss tensor.
    """
    return torch.norm(pred - target, dim=-1).mean()   # scalar


# ── Numpy evaluation metrics ─────────────────────────────────────────────────

def mpjpe_per_sample(
    pred: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Per-sample MPJPE in mm.

    Args:
        pred   : shape [N, T, J, 3]  (or [N, J, 3] for single-frame)
        target : same shape as pred

    Returns:
        Array of shape [N] — one MPJPE value per sequence.
    """
    error = np.linalg.norm(pred - target, axis=-1)   # [..., J]
    # Average over all axes except the first (batch)
    axes = tuple(range(1, error.ndim))
    return error.mean(axis=axes)                       # [N]


def per_joint_mpjpe(
    pred: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    Per-joint MPJPE averaged over all samples and frames.

    Args:
        pred   : shape [N, T, J, 3]  (or [N, J, 3])
        target : same shape as pred

    Returns:
        Array of shape [J] — one error value per joint.
    """
    error = np.linalg.norm(pred - target, axis=-1)   # [N, T, J] or [N, J]
    # Average over all axes except the last (joints)
    axes = tuple(range(error.ndim - 1))
    return error.mean(axis=axes)                       # [J]


# ── Procrustes alignment helpers ─────────────────────────────────────────────

def _procrustes_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Align a single predicted pose to its ground-truth via Procrustes analysis.

    Solves for the optimal rotation R, uniform scale s, and translation t
    that minimise  ||s * pred @ R.T + t  -  target||_F  using SVD.

    Args:
        pred   : shape [J, 3] — predicted skeleton (single frame).
        target : shape [J, 3] — ground-truth skeleton (single frame).

    Returns:
        Aligned prediction of shape [J, 3].
    """
    # 1. Centre both point clouds
    pred_mean   = pred.mean(axis=0)       # [3]
    target_mean = target.mean(axis=0)     # [3]
    pred_c   = pred   - pred_mean         # [J, 3]
    target_c = target - target_mean       # [J, 3]

    # 2. Compute scale as Frobenius norm of centred clouds
    pred_scale   = np.sqrt((pred_c   ** 2).sum())
    target_scale = np.sqrt((target_c ** 2).sum())

    # Guard against degenerate pose (all joints coincide)
    if pred_scale < 1e-8 or target_scale < 1e-8:
        return pred

    pred_c_n   = pred_c   / pred_scale
    target_c_n = target_c / target_scale

    # 3. SVD of cross-covariance matrix to find optimal rotation
    M = target_c_n.T @ pred_c_n           # [3, 3]
    U, _s, Vt = np.linalg.svd(M)

    # 4. Correct for reflections (ensure det(R) = +1)
    det = np.linalg.det(U @ Vt)
    S = np.eye(3)
    S[2, 2] = np.sign(det)               # flip last singular vector if needed

    R = U @ S @ Vt                        # [3, 3] rotation matrix

    # 5. Apply optimal scale × rotation + target translation
    aligned = target_scale * (pred_c_n @ R.T) + target_mean   # [J, 3]
    return aligned


def p_mpjpe(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Procrustes-aligned MPJPE (P-MPJPE) in mm.

    Each predicted pose is rigidly aligned (rotation + scale + translation)
    to its corresponding ground-truth pose before the error is measured.

    Args:
        pred   : shape [N, T, J, 3]  (or [N, J, 3])
        target : same shape as pred

    Returns:
        Scalar mean P-MPJPE over all samples and frames.
    """
    # Handle both [N, T, J, 3] and [N, J, 3] inputs
    if pred.ndim == 3:
        pred   = pred[:, np.newaxis]     # [N, 1, J, 3]
        target = target[:, np.newaxis]

    N, T, J, _ = pred.shape
    errors = np.empty((N, T))

    for n in range(N):
        for t in range(T):
            aligned = _procrustes_align(pred[n, t], target[n, t])   # [J, 3]
            errors[n, t] = np.linalg.norm(aligned - target[n, t], axis=-1).mean()

    return float(errors.mean())
