"""Shared constants for the LIBERO DINO-WM pipeline.

Every stage (data conversion, decoder/world-model/classifier training, the HJ
latent env, and the runtime safety filter) imports these so that the camera
mapping, tensor dimensions and model hyper-parameters stay in sync. Override the
paths with the LIBERO_WM_* environment variables instead of editing this file.
"""
import os

# --- raw LIBERO HDF5 layout (data/<demo>/...) -------------------------------
# DINO-WM is a two-camera model. We map the LIBERO cameras onto the existing
# "front"/"wrist" slots so the model code and dataset loader are reused as-is:
#   front  (video1, cam_zed) <- agentview_rgb
#   wrist  (video2, cam_rs)  <- eye_in_hand_rgb
FRONT_IMAGE_KEY = os.environ.get("LIBERO_WM_FRONT_KEY", "agentview_rgb")
WRIST_IMAGE_KEY = os.environ.get("LIBERO_WM_WRIST_KEY", "eye_in_hand_rgb")
# Proprio state = concat(ee_states, gripper_states); must total STATE_DIM.
EE_STATE_KEY = os.environ.get("LIBERO_WM_EE_KEY", "ee_states")
GRIPPER_STATE_KEY = os.environ.get("LIBERO_WM_GRIPPER_KEY", "gripper_states")
FAILURE_KEY = os.environ.get("LIBERO_WM_FAILURE_KEY", "failure")

# --- tensor dimensions ------------------------------------------------------
ACTION_DIM = 7          # LIBERO 7-DoF delta action (matches VideoTransformer)
STATE_DIM = 8           # proprio vector fed to the world model
DINO_IMAGE_SIZE = 224   # DINOv2 input / decoder output resolution

# --- DINO-WM model hyper-parameters (identical across all stages) -----------
WM_KWARGS = dict(
    image_size=(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE),
    dim=384,        # DINOv2 ViT-S/14 patch feature dim
    ac_dim=10,      # learned action-embedding dim
    state_dim=STATE_DIM,
    depth=6,
    heads=16,
    mlp_dim=2048,
    num_frames=3,   # context frames the transformer attends over
    dropout=0.1,
)
# Trajectory segment length: num_frames context + 1 prediction target.
SEGMENT_LENGTH = WM_KWARGS["num_frames"] + 1

DINOV2_HUB = ("facebookresearch/dinov2", "dinov2_vits14_reg")

# --- default file locations (override via env) ------------------------------
# Consolidated dataset produced by libero_to_dataset.py. World-model training
# splits THIS single file into train/test by TRAIN_FRAC (like RSSM's 90/10
# split), so no separate test file is required.
CONSOLIDATED_TRAIN = os.environ.get(
    "LIBERO_WM_DATA", "/data/libero/consolidated.h5"
)
# Fraction of trajectories used for training; the leading (1 - frac) are the
# held-out test/eval split.
TRAIN_FRAC = float(os.environ.get("LIBERO_WM_TRAIN_FRAC", "0.9"))
# Where checkpoints (decoder / world model / classifier) and action stats live.
CHECKPOINT_DIR = os.environ.get("LIBERO_WM_CKPT_DIR", "checkpoints")
ACTION_STATS_PATH = os.environ.get(
    "LIBERO_WM_ACTION_STATS", os.path.join(CHECKPOINT_DIR, "libero_action_stats.npz")
)
