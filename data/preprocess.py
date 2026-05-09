"""
preprocess.py — Data preprocessing utilities.

──────────────────────────────────────────────────────────────────────────────
HOW TO DOWNLOAD THE HUMAN3.6M DATA (VideoPose3D preprocessed format)
──────────────────────────────────────────────────────────────────────────────

The project uses two .npz files produced by the VideoPose3D preprocessing
pipeline.  The full Human3.6M dataset requires a licence agreement, but
the two preprocessed files together are only ~1–2 GB for the small subset
used here (S1 + S9, Walking + Eating).

Steps:
  1. Request access to Human3.6M at http://vision.imar.ro/human3.6m/
     (academic use is free — approval usually takes 1–2 days).

  2. Clone the VideoPose3D repository:
       git clone https://github.com/facebookresearch/VideoPose3D.git
       cd VideoPose3D

  3. Follow the data preparation instructions in DATASETS.md:
       cd data
       # Download the raw files from the Human3.6M website into ./h36m/
       python prepare_data_h36m.py --from-source

     This generates:
       data_3d_h36m.npz       — 3D ground-truth joint positions (mm)
       data_2d_h36m_gt.npz    — 2D projections from the camera

  4. Copy both files into the pose3d project's data directory:
       cp data_3d_h36m.npz    <project_root>/pose3d/data/h36m/
       cp data_2d_h36m_gt.npz <project_root>/pose3d/data/h36m/

  5. You can now train with:
       python train.py --data_root data/h36m

  No further preprocessing is needed — normalization stats are computed
  automatically and cached to data/norm_stats.npz on the first run.

──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import warnings

import numpy as np


# ── Normalization utilities ───────────────────────────────────────────────────

def compute_stats(
    sequences: list[np.ndarray],
    save_path: str = os.path.join("data", "norm_stats.npz"),
) -> dict[str, np.ndarray]:
    """
    Compute per-axis zero-mean / unit-variance normalization statistics over
    a list of pose sequences and save them to disk.

    Statistics are computed over ALL joints and ALL frames in the training
    split, then saved so validation / test sets can reuse them.

    Args:
        sequences : List of arrays, each shaped [T, J, 3].
        save_path : Path where the .npz file will be written.

    Returns:
        dict with keys 'mean' and 'std', both of shape [3].
    """
    if not sequences:
        raise ValueError("sequences list is empty — nothing to compute stats from.")

    # Stack all frames from all sequences: [N_total_frames, J, 3]
    all_frames = np.concatenate([s.reshape(-1, s.shape[-2], s.shape[-1])
                                  for s in sequences], axis=0)

    # Compute per-coordinate-axis statistics (collapse over frames and joints)
    mean = all_frames.mean(axis=(0, 1))   # [3]
    std  = all_frames.std(axis=(0, 1))    # [3]

    # Avoid division by zero for axes with no variance
    std = np.where(std < 1e-8, 1.0, std)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path, mean=mean, std=std)
    print(f"[preprocess] Normalization stats saved -> {save_path}")
    print(f"             mean={mean.round(3)},  std={std.round(3)}")

    return {"mean": mean, "std": std}


def load_stats(path: str = os.path.join("data", "norm_stats.npz")) -> dict[str, np.ndarray]:
    """
    Load previously computed normalization statistics from disk.

    Args:
        path : Path to the .npz file produced by compute_stats().

    Returns:
        dict with keys 'mean' [3] and 'std' [3].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Normalization stats not found at '{path}'. "
            "Run compute_stats() on the training set first, or use --dummy mode."
        )
    d = np.load(path)
    return {"mean": d["mean"], "std": d["std"]}


def normalize(joints: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """
    Apply zero-mean / unit-variance normalization.

    Args:
        joints : Joint positions, shape [..., 3].
        stats  : Dict with 'mean' [3] and 'std' [3].

    Returns:
        Normalized array, same shape as input.
    """
    return (joints - stats["mean"]) / stats["std"]


def denormalize(joints: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """
    Reverse normalization to recover joint positions in millimetres.

    Use this before computing MPJPE so that errors are reported in mm.

    Args:
        joints : Normalized joint positions, shape [..., 3].
        stats  : Dict with 'mean' [3] and 'std' [3].

    Returns:
        Denormalized array in the original coordinate scale (mm).
    """
    return joints * stats["std"] + stats["mean"]
