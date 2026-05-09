"""
visualization.py -- 3D skeleton plotting utilities.

Provides two functions:
  plot_skeleton_3d      : Draw a single skeleton onto a matplotlib 3-D axes.
  visualize_prediction  : Side-by-side plot of predicted vs ground-truth skeleton.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend -- safe for servers and CI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 -- registers 3d projection

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _center_on_root(joints: np.ndarray) -> np.ndarray:
    """
    Subtract the root joint (Hip, index 0) so the skeleton is centred at
    the origin.  This removes the large camera-space offset in MPI data
    (where Z ~ 3500mm from the camera) and makes the pose readable.
    """
    return joints - joints[0:1, :]   # [J, 3] - [1, 3]


def plot_skeleton_3d(
    joints: np.ndarray,
    ax: "Axes3D",
    color: str = "steelblue",
    joint_color: str = "tomato",
    alpha: float = 0.9,
    linewidth: float = 2.5,
    markersize: float = 5.0,
) -> None:
    """
    Draw a single 3-D skeleton onto an existing matplotlib 3-D axes.

    Bones are drawn as line segments; joints are drawn as scatter points.
    The skeleton is centred on the root joint before plotting so that
    camera-space offsets don't distort the view.

    Args:
        joints      : Joint positions, shape [J, 3]  (x, y, z in mm).
        ax          : A mpl_toolkits.mplot3d.Axes3D axes object.
        color       : Colour for bone line segments.
        joint_color : Colour for joint scatter points.
        alpha       : Transparency for bones.
        linewidth   : Width of bone lines.
        markersize  : Size of joint scatter markers.
    """
    joints = _center_on_root(np.asarray(joints))   # remove camera offset

    # MPI / camera space has Y pointing DOWN -- flip so head is at top
    joints = joints * np.array([1, -1, 1])

    for (i, j) in config.SKELETON_BONES:
        xs = [joints[i, 0], joints[j, 0]]
        ys = [joints[i, 1], joints[j, 1]]
        zs = [joints[i, 2], joints[j, 2]]
        ax.plot3D(xs, ys, zs, color=color, alpha=alpha, linewidth=linewidth)

    ax.scatter(
        joints[:, 0], joints[:, 1], joints[:, 2],
        c=joint_color, s=markersize ** 2, zorder=5,
    )


def visualize_prediction(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
    save_path: str,
    frame_idx: int = -1,
    title: str = "3D Pose Estimation",
) -> None:
    """
    Side-by-side 3-D comparison of predicted vs ground-truth skeleton.

    Shows three views per skeleton (front, side, top) so the pose is
    unambiguous regardless of the original coordinate system.

    Args:
        pred_seq  : Predicted pose sequence,    shape [T, J, 3] or [J, 3].
        gt_seq    : Ground-truth pose sequence, same shape as pred_seq.
        save_path : File path for the saved PNG.
        frame_idx : Which frame to visualise (-1 = last frame).
        title     : Figure suptitle.
    """
    pred_seq = np.asarray(pred_seq)
    gt_seq   = np.asarray(gt_seq)

    if pred_seq.ndim == 2:
        pred_seq = pred_seq[np.newaxis]
        gt_seq   = gt_seq[np.newaxis]

    pred_frame = pred_seq[frame_idx]   # [J, 3]
    gt_frame   = gt_seq[frame_idx]     # [J, 3]

    # Three viewpoints: front (elev=0,azim=0), side (elev=0,azim=90), top (elev=90,azim=0)
    views = [
        ("Front",  0,   0),
        ("Side",   0,  90),
        ("Top",   90,   0),
    ]

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for col, (view_name, elev, azim) in enumerate(views):
        # Ground truth
        ax_gt = fig.add_subplot(2, 3, col + 1, projection="3d")
        ax_gt.set_title(f"GT -- {view_name}", fontsize=9)
        plot_skeleton_3d(gt_frame, ax_gt, color="steelblue")
        _format_ax(ax_gt, elev, azim)

        # Prediction
        ax_pr = fig.add_subplot(2, 3, col + 4, projection="3d")
        ax_pr.set_title(f"Pred -- {view_name}", fontsize=9)
        plot_skeleton_3d(pred_frame, ax_pr, color="darkorange")
        _format_ax(ax_pr, elev, azim)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization -> {save_path}")


# ── Internal helper ───────────────────────────────────────────────────────────

def _format_ax(ax: "Axes3D", elev: int = 20, azim: int = -60) -> None:
    """Apply consistent axis formatting and viewing angle."""
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    # Equal aspect ratio so limbs aren't distorted
    _set_equal_axes(ax)


def _set_equal_axes(ax: "Axes3D") -> None:
    """Force equal axis scaling on a 3-D axes so bones look natural."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    spans  = limits[:, 1] - limits[:, 0]
    centres = limits.mean(axis=1)
    radius = spans.max() / 2.0
    ax.set_xlim3d([centres[0] - radius, centres[0] + radius])
    ax.set_ylim3d([centres[1] - radius, centres[1] + radius])
    ax.set_zlim3d([centres[2] - radius, centres[2] + radius])
