import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py

class SplitTrajectoryDataset(Dataset):
    def __init__(self, hdf5_file, segment_length, split='train', num_test=100, train_frac=None,
                 in_memory=None, with_images=True):
        """
        Custom Dataset that can load either the first 1000 trajectories or the rest.

        Args:
            hdf5_file (str): Path to the HDF5 file containing the trajectories.
            segment_length (int): Length of the segments to sample (H timesteps).
            split (str): 'train' for the first 1000 trajectories, 'test' for the rest.
            num_test (int): Number of leading trajectories used as the test split.
            train_frac (float | None): If set, split a SINGLE file by fraction:
                the trailing `train_frac` go to 'train', the leading
                `1 - train_frac` go to 'test'. Overrides num_test. This mirrors
                latent-safety's RSSM world-model train/val split.
            in_memory (bool | None): preload trajectories into RAM. None -> read
                the LIBERO_WM_IN_MEMORY env var (default off).
            with_images (bool): include the raw camera_0/camera_1 frames in each
                sample. Set False for world-model training (it only needs the DINO
                embeddings); this skips the large float32 images on read AND from
                the in-memory cache, so the embeddings-only cache fits in RAM.
        """
        self.hdf5_file = hdf5_file
        self.segment_length = segment_length
        self.split = split
        self.num_test = num_test
        self.with_images = with_images

        # Open HDF5 file to get a list of trajectory groups
        with h5py.File(self.hdf5_file, 'r') as hf:
            self.trajectory_ids = list(hf.keys())

        # Fraction split of one file (train = trailing frac, test = leading rest).
        if train_frac is not None:
            total = len(self.trajectory_ids)
            self.num_test = max(1, int(round((1.0 - train_frac) * total))) if total > 1 else 0

        # Split the dataset based on the specified split
        if self.split == 'train':
            self.trajectory_ids = self.trajectory_ids[self.num_test:]
        elif self.split == 'test':
            self.trajectory_ids = self.trajectory_ids[:self.num_test]
        else:
            raise ValueError("split must be 'train' or 'test'.")
        
        # Precompute trajectory slice indices
        self.slice_indices = []
        with h5py.File(self.hdf5_file, 'r') as hf:
            for traj_id in self.trajectory_ids:
                trajectory = hf[traj_id]
                traj_len = len(trajectory['actions'])
                for start_idx in range(0, traj_len - self.segment_length + 1, 1):
                    self.slice_indices.append((traj_id, start_idx))

        # Optional: preload every trajectory into RAM so __getitem__ slices from
        # memory instead of reopening the h5 file per sample and re-reading the
        # heavily overlapping windows from disk. Off by default; enable with
        # LIBERO_WM_IN_MEMORY=1 when the (cached) data fits in RAM. With
        # with_images=False the huge float32 images are excluded, so only the
        # embeddings/state/action are cached -- small enough to fit.
        if in_memory is None:
            in_memory = os.environ.get("LIBERO_WM_IN_MEMORY", "0") == "1"
        self._cache = None
        if in_memory:
            skip = set() if self.with_images else {"camera_0", "camera_1"}
            # Store the bulky DINO features as float16 to halve the cache size;
            # _segment upcasts back to float32 so training is unaffected. Set
            # LIBERO_WM_CACHE_FP16=0 to keep float32.
            fp16 = os.environ.get("LIBERO_WM_CACHE_FP16", "1") == "1"
            emb_keys = {"cam_zed_embd", "cam_rs_embd"}
            self._cache = {}
            with h5py.File(self.hdf5_file, 'r') as hf:
                for traj_id in self.trajectory_ids:
                    g = hf[traj_id]
                    self._cache[traj_id] = {
                        k: (g[k][:].astype(np.float16) if (fp16 and k in emb_keys) else g[k][:])
                        for k in g.keys() if k not in skip
                    }


    def __len__(self):
        """Returns the number of trajectories in the selected split."""
        return len(self.slice_indices)

    def __getitem__(self, idx):
        """Randomly samples a segment from a randomly selected trajectory."""

        traj_id, start_idx = self.slice_indices[idx]
        if self._cache is not None:
            return self._segment(self._cache[traj_id], start_idx)
        with h5py.File(self.hdf5_file, 'r') as hf:
            return self._segment(hf[traj_id], start_idx)

    def _segment(self, trajectory, start_idx):
        """Build one training sample from an h5 group or an in-RAM dict.

        Both support `trajectory[key][start:end]`, so the same slicing works for
        the on-disk and cached paths.
        """
        end_idx = start_idx + self.segment_length

        segment_obs_tensor = {}
        if self.with_images:
            segment_obs_tensor["robot0_eye_in_hand_image"] = torch.tensor(np.array(trajectory["camera_0"][start_idx:end_idx])*255., dtype=torch.uint8)
            segment_obs_tensor["agentview_image"] = torch.tensor(np.array(trajectory["camera_1"][start_idx:end_idx])*255., dtype=torch.uint8)
        segment_obs_tensor["cam_rs_embd"] = torch.tensor(np.array(trajectory["cam_rs_embd"][start_idx:end_idx]), dtype=torch.float32)
        segment_obs_tensor["cam_zed_embd"] = torch.tensor(np.array(trajectory["cam_zed_embd"][start_idx:end_idx]), dtype=torch.float32)
        segment_obs_tensor["state"] = torch.tensor(np.array(trajectory["states"][start_idx:end_idx]), dtype=torch.float32)
        segment_obs_tensor["action"] = torch.tensor(np.array(trajectory["actions"][start_idx:end_idx]), dtype=torch.float32)
        if "labels" in trajectory.keys():
            segment_obs_tensor["failure"] = torch.tensor(np.array(trajectory["labels"][start_idx:end_idx]), dtype=torch.float32)
        segment_obs_tensor["is_first"] = torch.zeros(self.segment_length)
        segment_obs_tensor["is_last"] = torch.zeros(self.segment_length)
        segment_obs_tensor["is_first"][0] = 1.
        segment_obs_tensor["is_terminal"] = segment_obs_tensor["is_last"]
        segment_obs_tensor["discount"] = torch.ones(self.segment_length, dtype=torch.float32)

        return segment_obs_tensor
    
if __name__ == '__main__':
    # Path to your HDF5 file
    hdf5_file = '/home/kensuke/data/skittles_trajectories_dreamer.h5'
    segment_length = 32  # Number of timesteps per segment
    batch_size = 32      # Number of trajectories per batch

    # Create the dataset
    train_dataset = SplitTrajectoryDataset(hdf5_file, segment_length, split='train', num_train=1000)
    test_dataset = SplitTrajectoryDataset(hdf5_file, segment_length, split='test', num_train=1000)


    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Example usage:
    for batch_idx, data in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(data.keys())
        print(data['agentview_image'].shape)
        print(data['agentview_image'].max())
        print(f"Observations: {data['cam_zed_right_embd'].shape}")
        print(f"Actions: {data['action'].shape}")

        break  # Just print one batch

    for batch_idx, data in enumerate(test_loader):
        print(f"Batch {batch_idx}:")
        print(data.keys())
        print(f"Observations: {data['cam_zed_right_embd'].shape}")
        print(f"Actions: {data['action'].shape}")


        
        break  # Just print one batch