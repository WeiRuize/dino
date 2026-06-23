"""Label per-step safety failures on raw LIBERO demos and write them back.

Writes a `failure` dataset (one int per timestep) into every data/<demo> group:
    0 = safe, 1 = unsafe, 2 = weak-unsafe (matches train_dino_classifier.fail_loss).

These are SAFETY labels (collision / forbidden state / malicious goal reached),
not task-success labels. libero_to_dataset.py copies them into `labels`.

Interactive labeling (matplotlib): for each demo, press 0/1/2 to label the shown
frame and advance; press <space> to step back. Shows agentview | eye_in_hand.

    python label_libero.py --raw_dir /data/libero_raw            # hand-label
    python label_libero.py --raw_dir /data/libero_raw --all_safe # mark all 0

Use --all_safe only for trajectories you know are entirely safe.
"""
import argparse
import os

import h5py
import numpy as np

import libero_config as C


def _read(group, key):
    item = group
    for part in str(key).split("/"):
        if part not in item:
            return None
        item = item[part]
    return item[()]


def _write_failure(demo, labels):
    arr = np.asarray(labels, dtype=np.int64)
    if "failure" in demo:
        del demo["failure"]
    demo.create_dataset("failure", data=arr)


def _label_demo_interactive(front, wrist):
    import matplotlib.pyplot as plt

    n = len(front)
    labels = {}
    state = {"idx": 0}
    fig, ax = plt.subplots()

    def draw():
        ax.clear()
        i = state["idx"]
        joint = np.concatenate(
            [np.uint8(front[i]), np.uint8(wrist[i])], axis=1
        )
        ax.imshow(joint)
        ax.set_title(f"frame {i}/{n - 1}  (0 safe / 1 unsafe / 2 weak / space=back)")
        ax.axis("off")
        fig.canvas.draw()

    def on_key(event):
        if event.key in {"0", "1", "2"}:
            labels[state["idx"]] = int(event.key)
            state["idx"] += 1
            if state["idx"] < n:
                draw()
            else:
                plt.close(fig)
        elif event.key == " " and state["idx"] > 0:
            state["idx"] -= 1
            draw()

    draw()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)
    # Any frames left unlabeled (window closed early) default to safe.
    return [labels.get(i, 0) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--all_safe", action="store_true",
                    help="mark every timestep safe (no interaction)")
    args = ap.parse_args()

    files = sorted(
        [os.path.join(args.raw_dir, f) for f in os.listdir(args.raw_dir)
         if f.endswith(".hdf5") or f.endswith(".h5")]
    )
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files in {args.raw_dir}")

    for fp in files:
        with h5py.File(fp, "r+") as h5:
            if "data" not in h5:
                continue
            for demo_name in sorted(h5["data"].keys()):
                demo = h5["data"][demo_name]
                front = _read(demo, "obs/" + C.FRONT_IMAGE_KEY)
                actions = _read(demo, "actions")
                if front is None or actions is None:
                    continue
                length = min(len(front), len(actions))
                if "failure" in demo and demo["failure"].shape[0] == length:
                    print(f"{fp}:{demo_name} already labeled, skipping")
                    continue
                if args.all_safe:
                    labels = [0] * length
                else:
                    wrist = _read(demo, "obs/" + C.WRIST_IMAGE_KEY)
                    if wrist is None:
                        wrist = front
                    labels = _label_demo_interactive(front[:length], wrist[:length])
                _write_failure(demo, labels)
                print(f"labeled {fp}:{demo_name}  unsafe={int(np.sum(np.array(labels) > 0))}/{length}")


if __name__ == "__main__":
    main()
