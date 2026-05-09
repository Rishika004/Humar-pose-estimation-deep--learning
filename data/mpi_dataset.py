"""
mpi_dataset.py -- MPI-INF-3DHP dataset loader.

Registration (free, usually same-day email verification):
  https://vcai.mpi-inf.mpg.de/3dhp-dataset/

Expected directory layout after download and extraction:
  <data_root>/
    S1/Seq1/annot.mat
    S1/Seq2/annot.mat
    S2/Seq1/annot.mat
    ...
    S6/Seq2/annot.mat
    mpi_inf_3dhp_test_set/
      TS1/annot_data.mat
      TS2/annot_data.mat
      ...
      TS6/annot_data.mat

Quick setup:
  1. Register at https://vcai.mpi-inf.mpg.de/3dhp-dataset/
  2. Download and extract into pose3d/data/mpi_inf_3dhp/
  3. Train:
       python train.py --dataset mpi --data_root data/mpi_inf_3dhp

MPI-INF-3DHP has 28 joints.  We remap to the same 17-joint H36M skeleton
used throughout this project so the rest of the pipeline (metrics,
visualization, config) requires no changes.

Joint mapping (MPI index -> H36M index):
  H36M  0 Hip        <- MPI  4 pelvis
  H36M  1 RHip       <- MPI 23 right_hip
  H36M  2 RKnee      <- MPI 24 right_knee
  H36M  3 RFoot      <- MPI 25 right_ankle
  H36M  4 LHip       <- MPI 18 left_hip
  H36M  5 LKnee      <- MPI 19 left_knee
  H36M  6 LFoot      <- MPI 20 left_ankle
  H36M  7 Spine      <- MPI  3 spine
  H36M  8 Thorax     <- MPI  2 spine2
  H36M  9 Neck       <- MPI  5 neck
  H36M 10 Head       <- MPI  6 head
  H36M 11 LShoulder  <- MPI  9 left_shoulder
  H36M 12 LElbow     <- MPI 10 left_elbow
  H36M 13 LWrist     <- MPI 11 left_wrist
  H36M 14 RShoulder  <- MPI 14 right_shoulder
  H36M 15 RElbow     <- MPI 15 right_elbow
  H36M 16 RWrist     <- MPI 16 right_wrist
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from .preprocess import compute_stats, load_stats, normalize


# ── Joint remapping ───────────────────────────────────────────────────────────

# MPI-INF-3DHP source indices for each of the 17 H36M target joints (in order)
MPI_TO_H36M_IDX = [4, 23, 24, 25, 18, 19, 20, 3, 2, 5, 6, 9, 10, 11, 14, 15, 16]

# Training subjects/sequences present in the download
TRAIN_SUBJECTS = ["S1", "S2", "S3", "S4", "S5", "S6"]
TRAIN_SEQS     = ["Seq1", "Seq2"]

# Test set folder names inside mpi_inf_3dhp_test_set/
TEST_SUBJECTS  = ["TS1", "TS2", "TS3", "TS4", "TS5", "TS6"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_train_annot(annot_path: str, camera_id: int = 0) -> np.ndarray:
    """
    Load 3D annotations from a training sequence .mat file.

    The training annot.mat stores a MATLAB cell array 'annot3' of shape
    [14 cameras, 1], where each cell contains [N_frames, 84] float32
    (84 = 28 joints * 3 coords).  We read one camera and reshape.

    Args:
        annot_path : Full path to annot.mat.
        camera_id  : Which camera to load (0-13).  Camera 4 is roughly
                     front-facing for most sequences.

    Returns:
        Array of shape [N_frames, 28, 3] in millimetres.
    """
    mat = scipy.io.loadmat(annot_path, squeeze_me=False)

    # Try 'univ_annot3' first (universal 3-D, comparable across subjects)
    # Fall back to 'annot3' if not present
    for key in ("univ_annot3", "annot3"):
        if key in mat:
            raw = mat[key]  # object array [14, 1], each cell [N_frames, 84]
            break
    else:
        raise KeyError(
            f"Neither 'univ_annot3' nor 'annot3' found in {annot_path}.  "
            f"Available keys: {[k for k in mat if not k.startswith('_')]}"
        )

    cell = raw[camera_id, 0]          # [N_frames, 84]
    poses = cell.reshape(-1, 28, 3)   # [N_frames, 28, 3]
    return poses.astype(np.float32)


def _load_test_annot(annot_path: str) -> np.ndarray:
    """
    Load 3D annotations from a test sequence annot_data.mat file.

    Test annotations use 'jointPositions' of shape [N_frames, 28*3].

    Args:
        annot_path : Full path to annot_data.mat.

    Returns:
        Array of shape [N_frames, 28, 3] in millimetres.
    """
    mat = scipy.io.loadmat(annot_path, squeeze_me=True)

    for key in ("jointPositions", "univ_annot3", "annot3"):
        if key in mat:
            raw = mat[key]
            break
    else:
        raise KeyError(
            f"Could not find joint position key in {annot_path}.  "
            f"Available: {[k for k in mat if not k.startswith('_')]}"
        )

    raw = np.array(raw)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)      # single frame edge case
    poses = raw.reshape(-1, 28, 3)    # [N_frames, 28, 3]
    return poses.astype(np.float32)


def _remap_joints(poses: np.ndarray) -> np.ndarray:
    """
    Select and reorder 17 joints from the 28-joint MPI skeleton to match H36M.

    Args:
        poses : shape [N_frames, 28, 3]

    Returns:
        shape [N_frames, 17, 3]
    """
    return poses[:, MPI_TO_H36M_IDX, :]    # [N_frames, 17, 3]


def _sliding_window(
    array: np.ndarray,
    seq_len: int,
    stride: int,
) -> list[np.ndarray]:
    """Extract fixed-length clips via sliding window."""
    n = len(array)
    return [array[s : s + seq_len] for s in range(0, n - seq_len + 1, stride)]


# ── Dataset class ─────────────────────────────────────────────────────────────

class MPIDataset(Dataset):
    """
    MPI-INF-3DHP dataset loader.

    Loads .mat annotation files, remaps to the 17-joint H36M skeleton,
    applies a sliding window, and normalises coordinates.

    Args:
        data_root  : Root directory of the MPI-INF-3DHP download.
        split      : 'train' (S1-S6 / Seq1-Seq2) or 'test' (TS1-TS6).
        subjects   : Override default subject list, e.g. ['S1', 'S2'].
                     Pass None to use the default for the split.
        seq_len    : Clip length T (frames).
        stride     : Sliding-window step.
        camera_id  : Which camera to use for training data (0-13).
        norm_stats : Pre-computed stats dict for test split.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        subjects: list[str] | None = None,
        seq_len: int = 16,
        stride: int = 8,
        camera_id: int = 4,
        norm_stats: dict | None = None,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.split     = split
        self._seqs_raw: list[np.ndarray] = []

        if split == "train":
            self._load_train(data_root, subjects or TRAIN_SUBJECTS,
                             seq_len, stride, camera_id)
        elif split == "test":
            self._load_test(data_root, subjects or TEST_SUBJECTS,
                            seq_len, stride)
        else:
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        if len(self._seqs_raw) == 0:
            raise RuntimeError(
                f"No clips loaded from '{data_root}' (split={split}).  "
                "Check the path and that the .mat files exist."
            )

        # ── Normalisation ──────────────────────────────────────────────────
        stats_path = os.path.join("data", "norm_stats_mpi.npz")

        if split == "train":
            if os.path.exists(stats_path):
                print(f"[MPIDataset] Reloading norm stats from {stats_path}")
                self.norm_stats = load_stats(stats_path)
            else:
                self.norm_stats = compute_stats(self._seqs_raw, stats_path)
        else:
            if norm_stats is not None:
                self.norm_stats = norm_stats
            elif os.path.exists(stats_path):
                self.norm_stats = load_stats(stats_path)
            else:
                raise FileNotFoundError(
                    "norm_stats_mpi.npz not found.  Train split first."
                )

        self.sequences: list[np.ndarray] = [
            normalize(s, self.norm_stats).astype(np.float32)
            for s in self._seqs_raw
        ]

        print(
            f"[MPIDataset] split={split:5s}  clips={len(self.sequences):,}  "
            f"shape={self.sequences[0].shape}"
        )

    # ── Internal loaders ──────────────────────────────────────────────────

    def _load_train(
        self,
        data_root: str,
        subjects: list[str],
        seq_len: int,
        stride: int,
        camera_id: int,
    ) -> None:
        for subj in subjects:
            for seq in TRAIN_SEQS:
                annot_path = os.path.join(data_root, subj, seq, "annot.mat")
                if not os.path.exists(annot_path):
                    warnings.warn(
                        f"[MPIDataset] {annot_path} not found -- skipping.",
                        stacklevel=4,
                    )
                    continue
                try:
                    poses28  = _load_train_annot(annot_path, camera_id)  # [N, 28, 3]
                    poses17  = _remap_joints(poses28)                     # [N, 17, 3]
                    clips    = _sliding_window(poses17, seq_len, stride)
                    self._seqs_raw.extend(clips)
                    print(f"  Loaded {subj}/{seq}: {len(poses17)} frames -> "
                          f"{len(clips)} clips")
                except Exception as e:
                    warnings.warn(
                        f"[MPIDataset] Failed to load {annot_path}: {e}",
                        stacklevel=4,
                    )

    def _load_test(
        self,
        data_root: str,
        subjects: list[str],
        seq_len: int,
        stride: int,
    ) -> None:
        test_dir = os.path.join(data_root, "mpi_inf_3dhp_test_set")
        for subj in subjects:
            annot_path = os.path.join(test_dir, subj, "annot_data.mat")
            if not os.path.exists(annot_path):
                warnings.warn(
                    f"[MPIDataset] {annot_path} not found -- skipping.",
                    stacklevel=4,
                )
                continue
            try:
                poses28  = _load_test_annot(annot_path)          # [N, 28, 3]
                poses17  = _remap_joints(poses28)                 # [N, 17, 3]
                clips    = _sliding_window(poses17, seq_len, stride)
                self._seqs_raw.extend(clips)
                print(f"  Loaded test/{subj}: {len(poses17)} frames -> "
                      f"{len(clips)} clips")
            except Exception as e:
                warnings.warn(
                    f"[MPIDataset] Failed to load {annot_path}: {e}",
                    stacklevel=4,
                )

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        """Returns a tuple of one tensor of shape [T, 17, 3]."""
        seq = torch.from_numpy(self.sequences[idx])   # [T, J, 3]
        return (seq,)
