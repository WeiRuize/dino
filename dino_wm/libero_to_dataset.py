"""Convert raw LIBERO demo HDF5 files into the consolidated dataset DINO-WM
trains on, and dump the per-dimension action statistics.

Raw layout expected (standard LIBERO):
    <file>.hdf5
      data/
        <demo_i>/
          actions                      (T, 7)
          dones                        (T,)              [optional]
          failure                      (T,)              [optional; see label_libero.py]
          obs/
            agentview_rgb              (T, H, W, 3)      -> front  / cam_zed
            eye_in_hand_rgb            (T, H, W, 3)      -> wrist   / cam_rs
            ee_states                  (T, ...)          \
            gripper_states             (T, ...)          / concat == STATE_DIM

Output (one HDF5, one group per demo), matching test_loader.SplitTrajectoryDataset:
    trajectory_<n>/
      actions, states, labels,
      camera_0  (wrist, float[0,1] 224x224x3),  camera_1 (front, float[0,1] 224x224x3),
      cam_rs_embd (wrist DINO), cam_zed_embd (front DINO)

Also writes ACTION_STATS_PATH (.npz with 'min'/'max') consumed at train/eval time
by dino_models.load_action_bounds().

Usage:
    python libero_to_dataset.py --raw_dir /data/libero_raw --out /data/libero/consolidated.h5
"""
import argparse
import os

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import libero_config as C

# DINOv2 input: resize to 224 then ImageNet-normalize (matches training transforms).
_DINO_NORMALIZE = transforms.Compose([
    transforms.Resize((C.DINO_IMAGE_SIZE, C.DINO_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Raw image kept for decoder supervision: resize to 224, no normalization, [0,1].
_RAW_RESIZE = transforms.Compose([
    transforms.Resize((C.DINO_IMAGE_SIZE, C.DINO_IMAGE_SIZE)),
    transforms.ToTensor(),
])


def _read(group, key):
    item = group
    for part in str(key).split("/"):
        if part not in item:
            return None
        item = item[part]
    return item[()]


def _build_state(demo, length):
    """state = concat(ee_states, gripper_states) clipped to STATE_DIM."""
    ee = _read(demo, "obs/" + C.EE_STATE_KEY)
    grip = _read(demo, "obs/" + C.GRIPPER_STATE_KEY)
    if ee is None:
        raise KeyError(f"obs/{C.EE_STATE_KEY} not found in demo")
    ee = np.asarray(ee, dtype=np.float32).reshape(length, -1)
    parts = [ee]
    if grip is not None:
        parts.append(np.asarray(grip, dtype=np.float32).reshape(length, -1))
    state = np.concatenate(parts, axis=-1)
    if state.shape[-1] != C.STATE_DIM:
        raise ValueError(
            f"state dim {state.shape[-1]} != STATE_DIM {C.STATE_DIM}. "
            f"Adjust EE_STATE_KEY/GRIPPER_STATE_KEY or STATE_DIM in libero_config.py "
            f"so train- and eval-time proprio definitions match."
        )
    return state


@torch.no_grad()
def _embed(dino, images, device):
    """images: (T,H,W,3) uint8 -> (raw float[0,1] (T,224,224,3), dino emb (T,256,384))."""
    raw_list, emb_list = [], []
    # HuggingFace/ModelScope 接口的 last_hidden_state 排列为 [CLS, 寄存器token..., patch token...]，
    # 需要丢掉前缀的 CLS + 寄存器 token，只保留 patch token（与 hub 的 x_norm_patchtokens 一致）。
    num_prefix = 1 + int(getattr(getattr(dino, "config", None), "num_register_tokens", 0) or 0)
    for t in range(len(images)):
        pil = Image.fromarray(np.uint8(images[t])).convert("RGB")
        raw_list.append(_RAW_RESIZE(pil).permute(1, 2, 0).numpy())  # HWC [0,1]
        inp = _DINO_NORMALIZE(pil).unsqueeze(0).to(device)

        # 兼容两种接口，两者都返回 patch token (num_patches, emb_dim)，不做池化。
        if hasattr(dino, 'forward_features'):
            # 原来的 PyTorch Hub 接口
            emb = dino.forward_features(inp)["x_norm_patchtokens"].squeeze(0)
        else:
            # ModelScope/HuggingFace 接口：保留 patch token，丢弃 CLS + 寄存器 token
            outputs = dino(inp)
            features = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
            emb = features[:, num_prefix:, :].squeeze(0)

        emb_list.append(emb.cpu().numpy())
    emb = np.stack(emb_list).astype(np.float32)
    # 提前拦截“模型选错”这类错误：patch 数须为完全平方，维度须与世界模型一致 (ViT-S/14 = 256x384)。
    n_patches, emb_dim = emb.shape[-2], emb.shape[-1]
    if int(n_patches ** 0.5) ** 2 != n_patches or emb_dim != C.WM_KWARGS["dim"]:
        raise ValueError(
            f"DINO embedding shape (num_patches={n_patches}, dim={emb_dim}) 与世界模型不匹配，"
            f"应为 (256, {C.WM_KWARGS['dim']})。请使用 ViT-S/14 的 "
            f"dinov2-with-registers-small 模型，而不是 base/large。"
        )
    return np.stack(raw_list).astype(np.float32), emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="directory of raw LIBERO *.hdf5/*.h5")
    ap.add_argument("--out", required=True, help="output consolidated .h5 path")
    ap.add_argument("--action_stats", default=C.ACTION_STATS_PATH)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device
    model_path = "/home/admin/Workspace/latent-safety-dino/dinov2"  # 或你下载的路径

    from transformers import AutoModel
    dino = AutoModel.from_pretrained(model_path).to(device).eval()

    files = sorted(
        [os.path.join(args.raw_dir, f) for f in os.listdir(args.raw_dir)
         if f.endswith(".hdf5") or f.endswith(".h5")]
    )
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files in {args.raw_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.action_stats)), exist_ok=True)
    ac_min, ac_max = None, None
    traj_idx = 0
    with h5py.File(args.out, "w") as out:
        for fp in tqdm(files, desc="files", ncols=0):
            with h5py.File(fp, "r") as h5:
                if "data" not in h5:
                    continue
                for demo_name in sorted(h5["data"].keys()):
                    demo = h5["data"][demo_name]
                    front = _read(demo, "obs/" + C.FRONT_IMAGE_KEY)
                    wrist = _read(demo, "obs/" + C.WRIST_IMAGE_KEY)
                    actions = _read(demo, "actions")
                    if front is None or wrist is None or actions is None:
                        print(f"skip {fp}:{demo_name} (missing cameras/actions)")
                        continue
                    actions = np.asarray(actions, dtype=np.float32)
                    length = min(len(front), len(wrist), len(actions))
                    if length < C.SEGMENT_LENGTH:
                        continue
                    front, wrist, actions = front[:length], wrist[:length], actions[:length]
                    state = _build_state(demo, length)

                    failure = _read(demo, C.FAILURE_KEY)
                    if failure is None:
                        failure = _read(demo, "obs/" + C.FAILURE_KEY)
                    if failure is None:
                        print(f"WARNING {fp}:{demo_name} has no '{C.FAILURE_KEY}'; "
                              f"writing zeros. Run label_libero.py first.")
                        failure = np.zeros(length, dtype=np.float32)
                    failure = np.asarray(failure, dtype=np.float32).reshape(-1)[:length]

                    front_raw, front_emb = _embed(dino, front, device)
                    wrist_raw, wrist_emb = _embed(dino, wrist, device)

                    g = out.create_group(f"trajectory_{traj_idx}")
                    g.create_dataset("actions", data=actions)
                    g.create_dataset("states", data=state)
                    g.create_dataset("labels", data=failure)
                    g.create_dataset("camera_1", data=front_raw)   # front  -> agentview
                    g.create_dataset("camera_0", data=wrist_raw)   # wrist  -> eye_in_hand
                    g.create_dataset("cam_zed_embd", data=front_emb)
                    g.create_dataset("cam_rs_embd", data=wrist_emb)

                    cur_min, cur_max = actions.min(0), actions.max(0)
                    ac_min = cur_min if ac_min is None else np.minimum(ac_min, cur_min)
                    ac_max = cur_max if ac_max is None else np.maximum(ac_max, cur_max)
                    traj_idx += 1

    if ac_min is None:
        raise RuntimeError("No valid trajectories converted.")
    # Guard against zero-range dims (e.g. constant gripper) to avoid divide-by-zero.
    ac_max = np.where(ac_max - ac_min < 1e-6, ac_min + 1e-6, ac_max)
    np.savez(args.action_stats, min=ac_min.astype(np.float32), max=ac_max.astype(np.float32))
    print(f"\nWrote {traj_idx} trajectories to {args.out}")
    print(f"action min: {ac_min}\naction max: {ac_max}\nstats -> {args.action_stats}")


if __name__ == "__main__":
    main()
