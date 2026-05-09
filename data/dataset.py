"""
dataset.py — Human3.6M dataset loader.

Loads the VideoPose3D preprocessed .npz files, filters by subject and
action, applies a sliding window to generate fixed-length clips, and
normalises joint coordinates to zero mean / unit variance.

Classes
-------
PoseDataset  — Real Human3.6M data loader.
DummyDataset — Random synthetic data for pipeline testing (no files needed).
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import compute_stats, load_stats, normalize


# ── Helpers ───────────────────────────────────────────────────────────────────

def _match_actions(available: list[str], requested: list[str]) -> list[str]:
    """
    Return the subset of *available* actions that match any *requested* name.

    Human3.6M stores actions with optional repetition suffixes, e.g.
    'Walking', 'Walking 1'.  A requested name 'Walking' therefore matches
    both variants via a startswith check.

    If requested == ['all'], every available action is returned.
    """
    if requested == ["all"]:
        return list(available)

    matched: list[str] = []
    for req in requested:
        hits = [a for a in available if a.startswith(req)]
        if not hits:
            warnings.warn(
                f"[Dataset] Action '{req}' not found in available actions "
                f"{available} — skipping.",
                stacklevel=3,
            )
        matched.extend(hits)
    return matched


def _sliding_window(
    array: np.ndarray,
    seq_len: int,
    stride: int,
) -> list[np.ndarray]:
    """
    Extract fixed-length clips from a continuous pose array via sliding window.

    Args:
        array   : Pose sequence, shape [N_frames, J, 3].
        seq_len : Clip length T.
        stride  : Step between consecutive clip starts.

    Returns:
        List of arrays, each shaped [T, J, 3].
    """
    n_frames = len(array)
    clips: list[np.ndarray] = []
    for start in range(0, n_frames - seq_len + 1, stride):
        clips.append(array[start : start + seq_len])
    return clips


# ── Real dataset ──────────────────────────────────────────────────────────────

class PoseDataset(Dataset):
    """
    Human3.6M pose dataset using the VideoPose3D preprocessed format.

    The dataset loader:
      1. Reads ``data_3d_h36m.npz`` from *data_root*.
      2. Filters to the requested subjects and actions.
      3. Applies a sliding window (length *seq_len*, step *stride*).
      4. Normalises coordinates using training-set statistics.

    Args:
        data_root  : Directory containing ``data_3d_h36m.npz``.
        subjects   : List of subject IDs, e.g. ``['S1', 'S5']``.
        actions    : List of action names, e.g. ``['Walking', 'Eating']``;
                     pass ``['all']`` to include every action.
        seq_len    : Number of frames per clip (T).
        stride     : Sliding-window step between clips.
        split      : ``'train'`` or ``'val'``.  Training split computes and
                     saves normalisation statistics; validation split reuses
                     them from disk or from *norm_stats*.
        norm_stats : Pre-computed stats dict (``{'mean': ..., 'std': ...}``).
                     Required when split='val' and no saved stats exist yet.
    """

    def __init__(
        self,
        data_root: str,
        subjects: list[str],
        actions: list[str],
        seq_len: int = 16,
        stride: int = 8,
        split: str = "train",
        norm_stats: dict | None = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.split = split

        # ── Load raw sequences ─────────────────────────────────────────────
        self._sequences_raw: list[np.ndarray] = []
        self._load_raw(data_root, subjects, actions, seq_len, stride)

        if len(self._sequences_raw) == 0:
            raise RuntimeError(
                "No clips were loaded.  Check --data_root, --subjects_train / "
                "--subjects_test, and --actions."
            )

        # ── Normalization ──────────────────────────────────────────────────
        stats_path = os.path.join("data", "norm_stats.npz")

        if split == "train":
            if os.path.exists(stats_path):
                print(f"[Dataset] Reloading norm stats from {stats_path}")
                self.norm_stats = load_stats(stats_path)
            else:
                self.norm_stats = compute_stats(self._sequences_raw, stats_path)
        else:
            # Validation / test split — must receive or find existing stats
            if norm_stats is not None:
                self.norm_stats = norm_stats
            elif os.path.exists(stats_path):
                self.norm_stats = load_stats(stats_path)
            else:
                raise FileNotFoundError(
                    "norm_stats.npz not found and none were passed for the "
                    "val/test split.  Train first, or pass norm_stats explicitly."
                )

        # ── Normalise and convert to float32 ──────────────────────────────
        self.sequences: list[np.ndarray] = [
            normalize(seq, self.norm_stats).astype(np.float32)
            for seq in self._sequences_raw
        ]

        print(
            f"[Dataset] split={split:5s}  clips={len(self.sequences):,}  "
            f"shape={self.sequences[0].shape}"
        )

    def _load_raw(
        self,
        data_root: str,
        subjects: list[str],
        actions: list[str],
        seq_len: int,
        stride: int,
    ) -> None:
        """Parse the .npz file and extract sliding-window clips."""
        npz_path = os.path.join(data_root, "data_3d_h36m.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Could not find '{npz_path}'.  "
                "See data/preprocess.py for download instructions, "
                "or use --dummy for a quick pipeline test."
            )

        raw = np.load(npz_path, allow_pickle=True)
        data: dict = raw["positions_3d"].item()   # {subject: {action: array[N,J,3]}}

        for subj in subjects:
            if subj not in data:
                warnings.warn(
                    f"[Dataset] Subject '{subj}' not found in the dataset — skipping.",
                    stacklevel=4,
                )
                continue

            subj_data: dict = data[subj]
            matched = _match_actions(list(subj_data.keys()), actions)

            for action in matched:
                poses = subj_data[action]          # [N_frames, J, 3]
                clips = _sliding_window(poses, seq_len, stride)
                self._sequences_raw.extend(clips)

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """
        Returns:
            Tuple of one tensor of shape [T, J, 3].
        """
        seq = torch.from_numpy(self.sequences[idx])   # [T, J, 3]
        return (seq,)


# ── Dummy dataset (no files required) ────────────────────────────────────────

class DummyDataset(Dataset):
    """
    Synthetic dataset that generates random tensors with the correct shapes.

    Intended for fast pipeline validation without any real data files.
    Prints a clear notice when instantiated.

    Args:
        num_samples : Number of sequences to generate.
        seq_len     : Frames per sequence (T).
        num_joints  : Joints per frame (J).
        joint_dim   : Coordinates per joint (default 3 for xyz).
        seed        : RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 2000,
        seq_len: int = 16,
        num_joints: int = 17,
        joint_dim: int = 3,
        seed: int = 42,
    ):
        super().__init__()
        print("[DUMMY MODE] Using synthetic data -- no real dataset required")

        rng = torch.Generator()
        rng.manual_seed(seed)
        self.data = torch.randn(
            num_samples, seq_len, num_joints, joint_dim,
            generator=rng,
        )

        # Identity norm stats so denormalize is a no-op in dummy mode
        self.norm_stats = {
            "mean": np.zeros(joint_dim, dtype=np.float32),
            "std":  np.ones(joint_dim,  dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """Returns a tuple of one tensor of shape [T, J, 3]."""
        return (self.data[idx],)
