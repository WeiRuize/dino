"""One-time, single-process slim of consolidated.h5.

Rewrites the dataset storing the bulky arrays (camera images + DINO embeddings)
as float16 instead of float32. This roughly halves the file size and the per-step
read volume with ZERO loader changes -- the loader already upcasts embeddings to
float32 and scales images by 255, so float16 is transparent. The small arrays
(actions / states / labels) are copied unchanged to keep full precision.

Runs strictly sequentially in a single process (one trajectory at a time), so it
cannot hit the h5py multiprocessing / file-locking deadlock.

    python slim_dataset.py /path/consolidated.h5 /path/consolidated_slim.h5
"""
import sys

import h5py
import numpy as np
from tqdm import tqdm

# Only these (the large arrays) are downcast to float16; everything else is copied
# verbatim so action/state normalization keeps full float32 precision.
FP16_KEYS = {"camera_0", "camera_1", "cam_zed_embd", "cam_rs_embd"}


def main():
    if len(sys.argv) != 3:
        print("usage: python slim_dataset.py <src.h5> <dst.h5>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    print(f"slimming {src} -> {dst}")

    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        traj_ids = list(fin.keys())
        for tid in tqdm(traj_ids, desc="slimming", ncols=0):
            gin = fin[tid]
            gout = fout.create_group(tid)
            for k in gin.keys():
                arr = gin[k][:]
                if k in FP16_KEYS and arr.dtype == np.float32:
                    arr = arr.astype(np.float16)
                gout.create_dataset(k, data=arr)

    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
