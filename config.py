"""
config.py — Central configuration for the 3D Human Pose Estimation project.

All hyperparameters and defaults live here. CLI arguments in train.py /
evaluate.py override these values at runtime.
"""

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/h36m"
SEQUENCE_LENGTH = 16          # T — frames per sliding-window clip
STRIDE = 8                    # step between consecutive windows (50% overlap)
NUM_JOINTS = 17               # J — Human3.6M skeleton
JOINT_DIM = 3                 # x, y, z coordinates per joint

SUBJECTS_TRAIN = ["S1"]
SUBJECTS_TEST = ["S9"]
DEFAULT_ACTIONS = ["Walking", "Eating"]   # pass "all" to use every action

# ── Model ─────────────────────────────────────────────────────────────────────
D_MODEL = 128                   # embedding / transformer hidden dim
NHEAD = 4                       # attention heads in spatial transformer
NUM_TRANSFORMER_LAYERS = 2      # stacked TransformerEncoderLayer blocks
DIM_FEEDFORWARD = 256           # FFN width inside each transformer layer
LSTM_HIDDEN = 256               # LSTM hidden size
LSTM_LAYERS = 2                 # stacked LSTM layers
DROPOUT = 0.1

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SEED = 42

# ── Human3.6M joint names (17 joints, canonical order) ───────────────────────
JOINT_NAMES = [
    "Hip",        # 0  — root / pelvis
    "RHip",       # 1
    "RKnee",      # 2
    "RFoot",      # 3
    "LHip",       # 4
    "LKnee",      # 5
    "LFoot",      # 6
    "Spine",      # 7
    "Thorax",     # 8
    "Neck",       # 9
    "Head",       # 10
    "LShoulder",  # 11
    "LElbow",     # 12
    "LWrist",     # 13
    "RShoulder",  # 14
    "RElbow",     # 15
    "RWrist",     # 16
]

# ── Bone connectivity (pairs of joint indices) ────────────────────────────────
SKELETON_BONES = [
    (0, 1), (1, 2), (2, 3),            # right leg
    (0, 4), (4, 5), (5, 6),            # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),   # spine → head
    (8, 11), (11, 12), (12, 13),       # left arm
    (8, 14), (14, 15), (15, 16),       # right arm
]
