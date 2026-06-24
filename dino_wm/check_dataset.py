"""Sanity-check a consolidated.h5 produced by libero_to_dataset.py.

The most important thing this verifies is that the cached DINO embeddings are
per-patch tokens of shape (T, 256, dim) -- NOT a pooled (T, dim) vector -- and
that the feature dim matches the world model (ViT-S/14 -> 384). It also checks
per-trajectory length consistency and finite values.

    python check_dataset.py                      # uses C.CONSOLIDATED_TRAIN
    python check_dataset.py /path/to/consolidated.h5
"""
import math
import sys

import h5py
import numpy as np

import libero_config as C

EXPECT_DIM = C.WM_KWARGS["dim"]              # 384 for ViT-S/14
EMB_KEYS = ("cam_zed_embd", "cam_rs_embd")   # front / wrist DINO features
# fields that should all share the same leading length T
LEN_KEYS = ("actions", "states", "camera_0", "camera_1",
            "cam_zed_embd", "cam_rs_embd", "labels")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else C.CONSOLIDATED_TRAIN
    print(f"checking {path}\n")
    problems = []

    with h5py.File(path, "r") as hf:
        traj_ids = list(hf.keys())
        print(f"trajectories: {len(traj_ids)}")
        if not traj_ids:
            print("ERROR: file has no trajectories")
            sys.exit(1)

        # --- detailed dump of the first trajectory --------------------------
        g0 = hf[traj_ids[0]]
        print(f"fields in {traj_ids[0]}:")
        for k in g0.keys():
            print(f"  {k:26s} shape={tuple(g0[k].shape)} dtype={g0[k].dtype}")
        print()

        # --- shape checks on every trajectory (metadata only, fast) ---------
        for tid in traj_ids:
            g = hf[tid]
            # 1) embeddings must be (T, num_patches, dim) patch tokens
            for key in EMB_KEYS:
                if key not in g:
                    problems.append(f"{tid}: missing {key}")
                    continue
                shp = g[key].shape
                if len(shp) != 3:
                    problems.append(
                        f"{tid}: {key} shape {shp} has {len(shp)} dims, expected 3 "
                        f"(T, num_patches, dim). This is POOLED data -> regenerate."
                    )
                    continue
                _, n_patch, dim = shp
                if math.isqrt(n_patch) ** 2 != n_patch:
                    problems.append(f"{tid}: {key} num_patches={n_patch} is not a perfect square")
                if dim != EXPECT_DIM:
                    problems.append(
                        f"{tid}: {key} dim={dim} != expected {EXPECT_DIM} "
                        f"(wrong DINO model -- use ViT-S/14 with registers)"
                    )
            # 2) every field shares the same length T
            lengths = {k: g[k].shape[0] for k in LEN_KEYS if k in g}
            if len(set(lengths.values())) > 1:
                problems.append(f"{tid}: inconsistent lengths {lengths}")

        # --- value sanity on the first trajectory's embeddings --------------
        for key in EMB_KEYS:
            if key in g0 and len(g0[key].shape) == 3:
                arr = np.asarray(g0[key][:])
                if not np.isfinite(arr).all():
                    problems.append(f"{traj_ids[0]}: {key} contains NaN/Inf")
                print(f"{key} value range: min={arr.min():.3f} max={arr.max():.3f} "
                      f"mean={arr.mean():.3f} std={arr.std():.3f}")

    print()
    if problems:
        print(f"FAILED with {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")
        sys.exit(1)
    print(f"OK: embeddings are (T, 256, {EXPECT_DIM}) patch tokens, lengths consistent, values finite.")


if __name__ == "__main__":
    main()
