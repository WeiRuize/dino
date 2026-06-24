"""Offline evaluation of the DINO-WM failure classifier.

Loads the world model whose failure_head was trained by train_dino_classifier.py
(checkpoints/best_classifier.pth), runs it once over the held-out TEST split, and
reports the confusion matrix (TP/FN/FP/TN) plus the rates that matter for a safety
filter. The immediate safety margin is failure_pred; margin > 0 == predicted safe.

This is an OFFLINE metric eval on cached DINO embeddings -- it does NOT launch any
environment or policy (that is stage 3 / the runtime safety filter). Run after
train_dino_classifier.py:

    python eval_dino_classifier.py
"""
import torch
from torch.utils.data import DataLoader

from test_loader import SplitTrajectoryDataset
from dino_models import VideoTransformer, normalize_acs, load_action_bounds
import libero_config as C


def confusion(pred, fail_data):
    """Confusion counts for "margin > 0 == safe".

    pred:      failure-head margin, shape (B, num_frames, 1)
    fail_data: labels (0 safe / 1 unsafe / 2 weak-unsafe), shape (B, num_frames)
    Label 2 (weak unsafe) is excluded from the matrix, matching training.
    """
    safe = torch.where(fail_data == 0.)
    unsafe = torch.where(fail_data == 1.)
    pos = pred[safe]
    neg = pred[unsafe]
    TP = torch.sum(pos > 0).item()   # safe   predicted safe
    FN = torch.sum(pos < 0).item()   # safe   predicted unsafe (false alarm)
    FP = torch.sum(neg > 0).item()   # unsafe predicted safe   (missed danger!)
    TN = torch.sum(neg < 0).item()   # unsafe predicted unsafe
    skipped = int(torch.sum(fail_data == 2.).item())
    return torch.tensor([TP, FN, FP, TN, skipped])


if __name__ == "__main__":
    device = "cuda:0"
    BS = 16
    BL = C.SEGMENT_LENGTH  # num_frames + 1

    # Action normalization must match the world model's training.
    load_action_bounds(C.ACTION_STATS_PATH)

    # Held-out test split of the same consolidated file (leading 1 - TRAIN_FRAC).
    # Confusion matrix only needs embeddings + failure labels; drop images and
    # never cache (single pass over the test split).
    expert_data = SplitTrajectoryDataset(
        C.CONSOLIDATED_TRAIN, BL, split="test", train_frac=C.TRAIN_FRAC,
        with_images=False, in_memory=False,
    )
    loader = DataLoader(expert_data, batch_size=BS, shuffle=False,
                        num_workers=C.NUM_WORKERS, pin_memory=True)

    transition = VideoTransformer(**C.WM_KWARGS).to(device)
    transition.load_state_dict(torch.load("checkpoints/best_classifier.pth", map_location=device))
    transition.eval()

    cm = torch.zeros(5, dtype=torch.long)
    with torch.no_grad():
        for data in loader:
            inputs1 = data["cam_zed_embd"][:, :-1].to(device)
            inputs2 = data["cam_rs_embd"][:, :-1].to(device)
            states = data["state"][:, :-1].to(device)
            acs = normalize_acs(data["action"][:, :-1].to(device), device)

            _, _, _, pred_fail = transition(inputs1, inputs2, states, acs)
            cm += confusion(pred_fail, data["failure"][:, 1:].to(device))

    TP, FN, FP, TN, skipped = cm.tolist()
    n_safe, n_unsafe = TP + FN, FP + TN
    print("\nConfusion matrix (margin > 0 == predicted safe):")
    print(f"  TP safe->safe     = {TP}")
    print(f"  FN safe->unsafe   = {FN}   (false alarms)")
    print(f"  FP unsafe->safe   = {FP}   (MISSED danger -- keep this low)")
    print(f"  TN unsafe->unsafe = {TN}")
    print(f"  weak-unsafe (label 2) skipped = {skipped}")
    if n_safe:
        print(f"  safe recall    TP/(TP+FN) = {TP / n_safe:.4f}")
    if n_unsafe:
        print(f"  unsafe recall  TN/(TN+FP) = {TN / n_unsafe:.4f}   (want high)")
        print(f"  miss rate (FP) FP/(FP+TN) = {FP / n_unsafe:.4f}   (want low)")
    print(f"  labeled safe/unsafe states = {n_safe + n_unsafe}")
