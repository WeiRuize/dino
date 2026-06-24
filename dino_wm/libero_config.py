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

# DataLoader worker processes used during training (per process / per GPU under
# DDP). 0 = load in the main process (the slowest, original behaviour). The h5
# dataset opens its own file handle per item, so it is safe with >0 workers.
# Override with LIBERO_WM_NUM_WORKERS.
NUM_WORKERS = int(os.environ.get("LIBERO_WM_NUM_WORKERS", "8"))

DINOV2_HUB = ("facebookresearch/dinov2", "dinov2_vits14_reg")

# --- DINOv2 encoder loading (single source of truth) ------------------------
# Local model directory in HuggingFace/ModelScope format (config.json + weights).
# Must be a ViT-S/14 *with registers* model (hidden_size=384, num_register_tokens=4)
# so the patch features match WM_KWARGS["dim"]. Override with LIBERO_WM_DINO_PATH.
# Every stage loads the encoder through load_dino() below, so changing the path
# here (or via env) repoints all scripts at once.
DINO_MODEL_PATH = os.environ.get(
    "LIBERO_WM_DINO_PATH", "/home/admin/Workspace/latent-safety-dino/dinov2"
)


def load_dino(device=None):
    """Load the DINOv2 encoder from DINO_MODEL_PATH (HuggingFace AutoModel).

    Returned in eval mode with parameters frozen (it is only used for feature
    extraction, never trained).
    """
    from transformers import AutoModel
    dino = AutoModel.from_pretrained(DINO_MODEL_PATH).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    if device is not None:
        dino = dino.to(device)
    return dino


def dino_patch_tokens(dino, inp):
    """Patch tokens (B, num_patches, emb_dim) for a normalized image batch.

    Handles both encoder interfaces so the data-generation and runtime paths
    stay byte-for-byte consistent:
      * torch-hub DINOv2  -> forward_features(...)["x_norm_patchtokens"]
      * HuggingFace AutoModel -> last_hidden_state with the leading CLS +
        register tokens stripped.
    """
    if hasattr(dino, "forward_features"):
        return dino.forward_features(inp)["x_norm_patchtokens"]
    num_prefix = 1 + int(getattr(getattr(dino, "config", None), "num_register_tokens", 0) or 0)
    out = dino(inp)
    feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
    return feats[:, num_prefix:, :]

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
