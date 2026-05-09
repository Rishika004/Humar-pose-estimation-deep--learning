"""
evaluate.py -- Evaluation script for the trained 3D pose estimation model.

Loads a checkpoint, runs inference on the test split, and reports:
  - Overall MPJPE (mm)
  - P-MPJPE after Procrustes alignment (mm)
  - Per-joint MPJPE for all 17 joints
  - Results saved to results/evaluation_report.txt

Usage
-----
  python evaluate.py --checkpoint checkpoints/best_model.pth --dummy
  python evaluate.py --checkpoint checkpoints/best_model.pth --dataset mpi --data_root data/mpi_inf_3dhp
  python evaluate.py --checkpoint checkpoints/best_model.pth --dataset h36m --data_root data/h36m
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.dataset import DummyDataset, PoseDataset
from data.mpi_dataset import MPIDataset
from data.preprocess import denormalize
from models.pose_model import Pose3DModel
from utils.metrics import mpjpe_per_sample, p_mpjpe, per_joint_mpjpe
from utils.visualization import visualize_prediction


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained 3D pose estimation checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to the .pth checkpoint file")

    # Dataset
    p.add_argument("--dataset",      type=str, default="mpi",
                   choices=["h36m", "mpi", "dummy"],
                   help="Which dataset to evaluate on: h36m | mpi | dummy")
    p.add_argument("--data_root",    type=str, default="data/mpi_inf_3dhp",
                   help="Dataset root directory")

    # H36M-specific
    p.add_argument("--subjects_test", nargs="+", default=config.SUBJECTS_TEST,
                   help="H36M test subject IDs")
    p.add_argument("--actions",       nargs="+", default=config.DEFAULT_ACTIONS,
                   help="H36M action names, or 'all'")

    # MPI-specific
    p.add_argument("--mpi_subjects_test", nargs="+",
                   default=["S5", "S6"],
                   help="MPI-INF-3DHP subjects to evaluate on (S1-S6)")
    p.add_argument("--mpi_camera",    type=int, default=4,
                   help="Camera index for MPI data (0-13)")

    p.add_argument("--dummy",         action="store_true",
                   help="Shortcut for --dataset dummy")
    p.add_argument("--seq_len",       type=int, default=config.SEQUENCE_LENGTH)
    p.add_argument("--batch_size",    type=int, default=config.BATCH_SIZE)
    p.add_argument("--seed",          type=int, default=config.SEED)
    p.add_argument("--results_dir",   type=str, default="results")
    p.add_argument("--save_vis",      action="store_true",
                   help="Save a sample visualisation PNG")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt       = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    d_model    = ckpt.get("d_model", saved_args.get("d_model", config.D_MODEL))

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_mode = "dummy" if args.dummy else args.dataset

    if dataset_mode == "dummy":
        test_ds    = DummyDataset(num_samples=400, seq_len=args.seq_len,
                                  seed=args.seed)
        norm_stats = test_ds.norm_stats
        mode_label = "DUMMY"

    elif dataset_mode == "mpi":
        test_ds = MPIDataset(
            data_root=args.data_root,
            split="train",          # held-out subjects from the training split
            subjects=args.mpi_subjects_test,
            seq_len=args.seq_len,
            stride=config.STRIDE,
            camera_id=args.mpi_camera,
        )
        norm_stats = test_ds.norm_stats
        mode_label = f"MPI  subjects={args.mpi_subjects_test}"

    else:  # h36m
        test_ds = PoseDataset(
            data_root=args.data_root,
            subjects=args.subjects_test,
            actions=args.actions,
            seq_len=args.seq_len,
            stride=config.STRIDE,
            split="val",
        )
        norm_stats = test_ds.norm_stats
        mode_label = f"H36M  subjects={args.subjects_test}"

    num_workers = 0 if os.name == "nt" else 4
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = Pose3DModel(
        d_model=d_model,
        nhead=config.NHEAD,
        num_transformer_layers=config.NUM_TRANSFORMER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        lstm_hidden=config.LSTM_HIDDEN,
        lstm_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT,
        num_joints=config.NUM_JOINTS,
        joint_dim=config.JOINT_DIM,
        predict_last_frame_only=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────
    all_preds:   list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for (poses,) in tqdm(test_loader, desc="Evaluating"):
            poses = poses.to(device)
            pred  = model(poses)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(poses.cpu().numpy())

    preds_norm   = np.concatenate(all_preds,   axis=0)  # [N, T, J, 3]
    targets_norm = np.concatenate(all_targets, axis=0)  # [N, T, J, 3]

    # ── Denormalize -> millimetres ─────────────────────────────────────────
    preds_mm   = denormalize(preds_norm,   norm_stats)
    targets_mm = denormalize(targets_norm, norm_stats)

    # ── Compute metrics ───────────────────────────────────────────────────
    per_sample   = mpjpe_per_sample(preds_mm, targets_mm)   # [N]
    mean_mpjpe   = float(per_sample.mean())

    print("Computing P-MPJPE (Procrustes alignment) -- this may take a moment...")
    mean_p_mpjpe = p_mpjpe(preds_mm, targets_mm)

    per_joint    = per_joint_mpjpe(preds_mm, targets_mm)    # [J]

    # ── Format report ─────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("=" * 58)
    lines.append("  3D Human Pose Estimation -- Evaluation Report")
    lines.append("=" * 58)
    lines.append(f"  Checkpoint  : {args.checkpoint}")
    lines.append(f"  Dataset     : {mode_label}")
    lines.append(f"  Test clips  : {len(test_ds):,}")
    lines.append("")
    lines.append(f"  MPJPE       : {mean_mpjpe:.2f} mm")
    lines.append(f"  P-MPJPE     : {mean_p_mpjpe:.2f} mm")
    lines.append("")
    lines.append("  Per-joint MPJPE (mm):")
    lines.append("  " + "-" * 30)
    for j, (name, err) in enumerate(zip(config.JOINT_NAMES, per_joint)):
        lines.append(f"  {j:2d}  {name:<12s}  {err:6.2f} mm")
    lines.append("=" * 58)

    report = "\n".join(lines)
    print("\n" + report)

    # ── Save to file ──────────────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    report_path = os.path.join(args.results_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"\nReport saved -> {report_path}")

    # ── Optional visualisation ────────────────────────────────────────────
    if args.save_vis:
        vis_path = os.path.join(args.results_dir, "sample_prediction.png")
        visualize_prediction(
            pred_seq=preds_mm[0],
            gt_seq=targets_mm[0],
            save_path=vis_path,
            title=f"Sample prediction (MPJPE={per_sample[0]:.1f} mm)",
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate(parse_args())
