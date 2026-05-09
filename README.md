# 3D Human Pose Estimation — Spatial Transformer + Temporal LSTM

A complete deep learning pipeline for lifting 3-D human body poses from
sequences of joint positions, implemented in PyTorch.

---

## Architecture

```
Input: [B, T, J, 3]  (batch, frames, joints, xyz)
  │
  ▼
┌─────────────────────────────────────────────┐
│  JointEmbedding                             │
│  • Linear projection: 3 → d_model (128)     │
│  • + Learnable joint-type embedding [J, 128]│
│  Output: [B, T, J, 128]                     │
└─────────────────────────────────────────────┘
  │
  ▼  reshape to [B*T, J, 128]
┌─────────────────────────────────────────────┐
│  SpatialTransformerEncoder                  │
│  • 2× TransformerEncoderLayer               │
│    - nhead=4, d_ff=256, dropout=0.1         │
│  • Mean-pool over J joints → [B*T, 128]     │
│  • Reshape → [B, T, 128]                    │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  TemporalLSTM                               │
│  • 2-layer LSTM, hidden=256                 │
│  Output: [B, T, 256]                        │
└─────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────┐
│  Linear head: 256 → J×3                     │
│  Reshape: [B, T, J, 3]                      │
└─────────────────────────────────────────────┘
  │
  ▼
Output: [B, T, J, 3]   — predicted 3D joint positions
```

---

## Dataset

This project uses the **VideoPose3D preprocessed Human3.6M format**
(`data_3d_h36m.npz` + `data_2d_h36m_gt.npz`).

| Detail | Value |
|--------|-------|
| Full Human3.6M size | ~60 GB (raw videos + annotations) |
| Preprocessed .npz size (all subjects) | ~700 MB |
| **Project default subset** (S1 train / S9 test, Walking + Eating) | **~1–2 GB** |
| Sequences at default settings (T=16, stride=8) | ~15,000–20,000 |

### Downloading the data

See `data/preprocess.py` for the full step-by-step guide.  In brief:

```bash
# 1. Request access: http://vision.imar.ro/human3.6m/
# 2. Clone VideoPose3D
git clone https://github.com/facebookresearch/VideoPose3D.git
cd VideoPose3D/data
python prepare_data_h36m.py --from-source

# 3. Copy the two files into this project
cp data_3d_h36m.npz    <project>/pose3d/data/h36m/
cp data_2d_h36m_gt.npz <project>/pose3d/data/h36m/
```

---

## Setup

```bash
cd pose3d
pip install -r requirements.txt
```

---

## Quick Start — No Dataset Needed

The `--dummy` flag generates random tensors with correct shapes so you can
verify the entire pipeline instantly without any data files:

```bash
python train.py --dummy --epochs 5
```

---

## Training

### Default small subset (S1 train / S9 test, Walking + Eating)

```bash
python train.py --data_root data/h36m --subjects_train S1 --subjects_test S9 --actions Walking Eating
```

### Larger training run (more subjects + all actions)

```bash
python train.py --data_root data/h36m --subjects_train S1 S5 S6 --subjects_test S7 S8 --actions all
```

### Resume from checkpoint

```bash
python train.py --data_root data/h36m --resume checkpoints/best_model.pth
```

### Key training flags

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-3 | Initial learning rate |
| `--d_model` | 128 | Transformer / embedding dimension |
| `--seed` | 42 | Random seed for reproducibility |

---

## Evaluation

```bash
# Quick check with dummy data
python evaluate.py --checkpoint checkpoints/best_model.pth --dummy

# Real data evaluation
python evaluate.py --checkpoint checkpoints/best_model.pth \
    --data_root data/h36m --subjects_test S9 --actions Walking Eating

# Save a sample visualisation PNG
python evaluate.py --checkpoint checkpoints/best_model.pth --dummy --save_vis
```

Results are saved to `results/evaluation_report.txt`.

---

## Expected Results

On the **default small subset** (S1 train, 50 epochs):

| Metric | Expected range |
|--------|----------------|
| MPJPE  | ~60–80 mm |
| P-MPJPE| ~45–65 mm |

Performance improves significantly when adding more subjects (S1–S7) and
actions, converging toward the ~50–60 mm range reported in the literature
for similar lightweight architectures.

---

## Output Files

| Path | Description |
|------|-------------|
| `checkpoints/best_model.pth` | Best checkpoint (lowest val MPJPE) |
| `training_log.csv` | Per-epoch: loss, val MPJPE, LR |
| `data/norm_stats.npz` | Cached normalisation statistics |
| `results/evaluation_report.txt` | Full evaluation report |
| `results/sample_prediction.png` | Visualisation (with `--save_vis`) |

---

## Ablation Ideas

| Experiment | How to run |
|------------|-----------|
| Remove temporal LSTM (spatial-only) | Modify `pose_model.py`: replace LSTM with identity, pass spatial output directly to head |
| Vary transformer depth | `--` edit `config.NUM_TRANSFORMER_LAYERS` to 1 or 4 |
| Vary sequence length | `--seq_len 8` or `--seq_len 32` |
| Larger d_model | `--d_model 256` |
| Single subject vs multiple | Compare `--subjects_train S1` vs `--subjects_train S1 S5 S6 S7` |

---

## Project Structure

```
pose3d/
├── config.py                  # All hyperparameters
├── train.py                   # Training loop
├── evaluate.py                # Evaluation + metrics report
├── requirements.txt
├── data/
│   ├── dataset.py             # PoseDataset + DummyDataset
│   └── preprocess.py          # Normalization, download instructions
├── models/
│   ├── embedding.py           # JointEmbedding
│   ├── spatial_transformer.py # SpatialTransformerEncoder
│   ├── temporal_lstm.py       # TemporalLSTM
│   └── pose_model.py          # Pose3DModel (full assembled model)
└── utils/
    ├── metrics.py             # MPJPE, P-MPJPE, per-joint errors
    └── visualization.py       # 3D skeleton plotting
```
