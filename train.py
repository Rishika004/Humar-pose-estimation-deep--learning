"""
train.py — Training script for the 3D Human Pose Estimation model.

Usage examples
--------------
  # Quick pipeline test with synthetic data (no dataset required):
  python train.py --dummy --epochs 2

  # Train on the default small subset (S1 train / S9 test):
  python train.py --data_root data/h36m

  # Larger training run:
  python train.py --data_root data/h36m --subjects_train S1 S5 S6 --actions all

  # Resume from a checkpoint:
  python train.py --dummy --resume checkpoints/best_model.pth
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.amp
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from data.dataset import DummyDataset, PoseDataset
from data.mpi_dataset import MPIDataset
from models.pose_model import Pose3DModel
from utils.metrics import mpjpe


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments; defaults come from config.py."""
    p = argparse.ArgumentParser(
        description="Train 3D Human Pose Estimation (SpatialTransformer + TemporalLSTM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    p.add_argument("--dataset",         type=str, default="h36m",
                   choices=["h36m", "mpi", "dummy"],
                   help="Which dataset to use: h36m | mpi | dummy")
    p.add_argument("--data_root",       type=str, default=config.DATA_ROOT,
                   help="Dataset root directory")
    p.add_argument("--subjects_train",  nargs="+", default=config.SUBJECTS_TRAIN,
                   help="H36M training subject IDs, e.g. S1 S5 S6")
    p.add_argument("--subjects_test",   nargs="+", default=config.SUBJECTS_TEST,
                   help="H36M test/validation subject IDs")
    p.add_argument("--actions",         nargs="+", default=config.DEFAULT_ACTIONS,
                   help="H36M action names, or 'all'")
    p.add_argument("--mpi_subjects_train", nargs="+",
                   default=["S1", "S2", "S3", "S4"],
                   help="MPI-INF-3DHP training subjects (S1-S6)")
    p.add_argument("--mpi_subjects_test",  nargs="+",
                   default=["S5", "S6"],
                   help="MPI-INF-3DHP val subjects (S5-S6) or test set TS1-TS6")
    p.add_argument("--mpi_camera",      type=int, default=4,
                   help="Camera index to use for MPI training data (0-13)")
    p.add_argument("--dummy",           action="store_true",
                   help="Shortcut for --dataset dummy")

    # ── Model ─────────────────────────────────────────────────────────────
    p.add_argument("--d_model",     type=int, default=config.D_MODEL)
    p.add_argument("--seq_len",     type=int, default=config.SEQUENCE_LENGTH)

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",         type=int,   default=config.EPOCHS)
    p.add_argument("--batch_size",     type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=config.LR)
    p.add_argument("--weight_decay",   type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--grad_clip",      type=float, default=config.GRAD_CLIP)
    p.add_argument("--seed",           type=int,   default=config.SEED)

    # ── Checkpointing ─────────────────────────────────────────────────────
    p.add_argument("--resume",          type=str, default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--checkpoint_dir",  type=str, default="checkpoints")

    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────

def _make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a DataLoader; uses 0 workers on Windows to avoid fork issues."""
    # Windows (nt) does not support fork-based multiprocessing for DataLoader
    num_workers = 0 if os.name == "nt" else 4
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Run one full validation pass and return mean MPJPE (normalised units)."""
    model.eval()
    errors: list[float] = []
    for (poses,) in tqdm(loader, desc="  Validation", leave=False):
        poses = poses.to(device)
        pred  = model(poses)
        err   = torch.norm(pred - poses, dim=-1).mean(dim=(-2, -1))  # [B]
        errors.extend(err.cpu().tolist())
    return float(np.mean(errors))


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Datasets ──────────────────────────────────────────────────────────
    dataset_mode = "dummy" if args.dummy else args.dataset

    if dataset_mode == "dummy":
        train_ds = DummyDataset(num_samples=2000, seq_len=args.seq_len, seed=args.seed)
        val_ds   = DummyDataset(num_samples=400,  seq_len=args.seq_len, seed=args.seed + 1)

    elif dataset_mode == "mpi":
        train_ds = MPIDataset(
            data_root=args.data_root,
            split="train",
            subjects=args.mpi_subjects_train,
            seq_len=args.seq_len,
            stride=config.STRIDE,
            camera_id=args.mpi_camera,
        )
        val_ds = MPIDataset(
            data_root=args.data_root,
            split="train",          # use held-out subjects from train split
            subjects=args.mpi_subjects_test,
            seq_len=args.seq_len,
            stride=config.STRIDE,
            camera_id=args.mpi_camera,
            norm_stats=train_ds.norm_stats,
        )

    else:  # h36m
        train_ds = PoseDataset(
            data_root=args.data_root,
            subjects=args.subjects_train,
            actions=args.actions,
            seq_len=args.seq_len,
            stride=config.STRIDE,
            split="train",
        )
        val_ds = PoseDataset(
            data_root=args.data_root,
            subjects=args.subjects_test,
            actions=args.actions,
            seq_len=args.seq_len,
            stride=config.STRIDE,
            split="val",
            norm_stats=train_ds.norm_stats,
        )

    train_loader = _make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader   = _make_loader(val_ds,   args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    model = Pose3DModel(
        d_model=args.d_model,
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

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_mpjpe_val = float("inf")

    # ── Resume ────────────────────────────────────────────────────────────
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"[WARNING] Checkpoint '{args.resume}' not found -- starting fresh.")
        else:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch    = ckpt["epoch"] + 1
            best_mpjpe_val = ckpt.get("best_mpjpe", float("inf"))
            print(f"[Checkpoint] Resumed from epoch {start_epoch}  "
                  f"(best_val_mpjpe={best_mpjpe_val:.4f})")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Training summary ──────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print("=" * 62)
    print("  3D Human Pose Estimation -- Training Summary")
    print("=" * 62)
    print(f"  Dataset          : {dataset_mode.upper()}")
    print(f"  Subjects (train) : {args.subjects_train}")
    print(f"  Subjects (val)   : {args.subjects_test}")
    print(f"  Actions          : {args.actions}")
    print(f"  Train sequences  : {len(train_ds):,}")
    print(f"  Val   sequences  : {len(val_ds):,}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Learning rate    : {args.lr}")
    print(f"  Device           : {device}")
    print(f"  Model params     : {total_params:,}")
    print("=" * 62)
    print()

    # ── CSV logger ────────────────────────────────────────────────────────
    log_path = "training_log.csv"
    write_header = not os.path.exists(log_path) or start_epoch == 0
    csv_file = open(log_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["epoch", "train_loss", "val_mpjpe", "lr"])

    # ── Main loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs} [Train]")
        for batch_idx, (poses,) in enumerate(pbar):
            poses = poses.to(device)           # [B, T, J, 3]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(poses)            # [B, T, J, 3]
                loss = mpjpe(pred, poses)      # scalar

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1

            if batch_idx % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        avg_train_loss = total_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────
        val_mpjpe_val = validate(model, val_loader, device)
        current_lr    = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch {epoch+1:03d}/{args.epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_mpjpe={val_mpjpe_val:.4f}  "
            f"lr={current_lr:.2e}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_mpjpe_val < best_mpjpe_val:
            best_mpjpe_val = val_mpjpe_val
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict":    scaler.state_dict(),
                    "best_mpjpe":           best_mpjpe_val,
                    "args":                 vars(args),
                    "d_model":              args.d_model,
                },
                ckpt_path,
            )
            print(f"  >> Best model saved -> {ckpt_path}  (val_mpjpe={best_mpjpe_val:.4f})")

        # ── CSV log ───────────────────────────────────────────────────────
        csv_writer.writerow([epoch + 1, avg_train_loss, val_mpjpe_val, current_lr])
        csv_file.flush()

    csv_file.close()
    print(f"\nTraining complete.  Best val MPJPE = {best_mpjpe_val:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
