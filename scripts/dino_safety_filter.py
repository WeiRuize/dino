"""Runtime DINO-WM safety filter for the OpenVLA / LIBERO online evaluation.

This is the DINO-WM counterpart of latent-safety's `libero_safety_filter.py`.
It exposes the SAME adapter interface the OpenVLA eval harness
(`openvla/run_libero_eval_wm.py`) talks to, so that script can switch from the
RSSM filter to this one by changing a single import:

    # from libero_safety_filter import build_libero_safety_filter
    from dino_safety_filter import build_libero_safety_filter

Unlike the RSSM filter (which carries a recurrent posterior), DINO-WM is a
fixed-context video transformer, so we keep a sliding window of the last
`num_frames` DINO embeddings / proprio states / actions and roll the model one
step forward to score each proposed action.

Convention (matches the latent env): the safety score is positive when safe;
an action is rejected when score <= threshold.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "dino_wm")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dino_wm.dino_models import VideoTransformer, normalize_acs, load_action_bounds
import dino_wm.libero_config as C


_DINO_NORMALIZE = transforms.Compose([
    transforms.Resize((C.DINO_IMAGE_SIZE, C.DINO_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def quat_to_axis_angle(quat):
    quat = np.asarray(quat, dtype=np.float32)
    quat = quat / max(np.linalg.norm(quat), 1e-8)
    if quat[0] < 0.0:
        quat = -quat
    sin_theta = np.linalg.norm(quat[1:])
    if sin_theta < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_theta, quat[0])
    return (quat[1:] / sin_theta * angle).astype(np.float32)


class DinoLatentHJSafetyFilter:
    """Slides a DINO-WM context window and scores proposed actions."""

    def __init__(self, wm, dino, policy=None, action_dim=C.ACTION_DIM,
                 threshold=0.0, device="cuda:0"):
        self.wm = wm
        self.dino = dino
        self.policy = policy
        self.action_dim = int(action_dim)
        self.threshold = float(threshold)
        self.device = device
        self.num_frames = C.WM_KWARGS["num_frames"]
        self.reset()

    def reset(self):
        self.front = []   # list of (256, 384) np arrays
        self.wrist = []
        self.state = []   # list of (STATE_DIM,) np arrays
        self.actions = []  # list of (action_dim,) np arrays (normalized)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

    @torch.no_grad()
    def _encode(self, image):
        pil = Image.fromarray(np.uint8(image)).convert("RGB")
        inp = _DINO_NORMALIZE(pil).unsqueeze(0).to(self.device)
        emb = self.dino.forward_features(inp)["x_norm_patchtokens"].squeeze(0)
        return emb.cpu().numpy().astype(np.float32)

    def _push(self, lst, item):
        lst.append(item)
        if len(lst) > self.num_frames:
            lst.pop(0)
        # left-pad with the oldest frame until the window is full
        while len(lst) < self.num_frames:
            lst.insert(0, lst[0])

    def observe(self, obs, prev_action=None, is_first=False, is_terminal=False):
        """obs = {'front': HxWx3, 'wrist': HxWx3, 'state': (STATE_DIM,)}."""
        act = self.prev_action if prev_action is None else np.asarray(prev_action, dtype=np.float32)
        act = normalize_acs(
            torch.as_tensor(act.reshape(1, self.action_dim), device=self.device)
        )[0].cpu().numpy()
        self._push(self.front, self._encode(obs["front"]))
        self._push(self.wrist, self._encode(obs["wrist"]))
        self._push(self.state, np.asarray(obs["state"], dtype=np.float32))
        self._push(self.actions, act)

    def _hist_tensors(self, candidate_action):
        front = torch.as_tensor(np.stack(self.front)[None], device=self.device)          # (1,F,256,384)
        wrist = torch.as_tensor(np.stack(self.wrist)[None], device=self.device)
        state = torch.as_tensor(np.stack(self.state)[None], device=self.device)          # (1,F,STATE)
        acs = np.stack(self.actions).copy()
        cand = normalize_acs(
            torch.as_tensor(np.asarray(candidate_action, dtype=np.float32).reshape(1, self.action_dim),
                            device=self.device)
        )[0].cpu().numpy()
        acs[-1] = cand                                                                   # eval candidate at the latest step
        acs = torch.as_tensor(acs[None], device=self.device)                             # (1,F,action_dim)
        return front, wrist, state, acs

    @torch.no_grad()
    def score_action(self, action):
        if not self.front:
            raise RuntimeError("Call observe() before scoring actions.")
        front, wrist, state, acs = self._hist_tensors(action)
        latent = self.wm.forward_features(front, wrist, state, acs)
        if self.policy is not None:
            feat = latent[:, [-1]].mean(dim=2).reshape(1, -1).cpu().numpy()
            act = np.asarray(action, dtype=np.float32).reshape(1, self.action_dim)
            return float(self.policy.critic(feat, act).detach().cpu().numpy().squeeze())
        margin = torch.tanh(2 * self.wm.failure_pred(latent)[0, -1])
        return float(margin.detach().cpu().numpy().squeeze())

    def filter_action(self, action):
        score = self.score_action(action)
        truncated = score <= self.threshold
        info = {"truncated": bool(truncated), "safety_score": score, "threshold": self.threshold}
        if truncated:
            return np.zeros(self.action_dim, dtype=np.float32), info
        return np.asarray(action, dtype=np.float32), info


class LiberoDinoSafetyAdapter:
    """Bridge raw LIBERO env obs -> DinoLatentHJSafetyFilter."""

    def __init__(self, hj_filter, front_image_key="agentview_image",
                 wrist_image_key="robot0_eye_in_hand_image", obs_state_key=None):
        self.hj_filter = hj_filter
        self.front_image_key = front_image_key
        self.wrist_image_key = wrist_image_key
        self.obs_state_key = obs_state_key

    def reset(self):
        self.hj_filter.reset()

    def _state(self, obs):
        if self.obs_state_key is not None:
            return np.asarray(obs[self.obs_state_key], dtype=np.float32)
        # Match libero_to_dataset convention: [eef_pos(3), axis_angle(3), gripper(2)].
        return np.concatenate([
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat_to_axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ])

    def _to_wm_obs(self, obs):
        return {
            "front": np.asarray(obs[self.front_image_key]),
            "wrist": np.asarray(obs[self.wrist_image_key]),
            "state": self._state(obs),
        }

    def observe(self, libero_obs, prev_action=None, is_first=False, is_terminal=False):
        self.hj_filter.observe(self._to_wm_obs(libero_obs), prev_action=prev_action,
                               is_first=is_first, is_terminal=is_terminal)

    def filter_action(self, action, reject_action_mode="zero"):
        raw = np.asarray(action, dtype=np.float32)
        safe, info = self.hj_filter.filter_action(raw)
        if info["truncated"]:
            if reject_action_mode == "clip":
                executed = np.clip(raw, -1.0, 1.0).astype(np.float32)
            elif reject_action_mode == "identity":
                executed = raw.copy()
            else:
                executed = np.zeros_like(raw, dtype=np.float32)
        else:
            executed = safe
        info = dict(info)
        info["reject_action_mode"] = reject_action_mode
        return executed, info


def _build_hj_policy(hj_policy_path, device, state_shape=(1, 1, 786), action_shape=(C.ACTION_DIM,)):
    from PyHJ.exploration import GaussianNoise
    from PyHJ.policy import avoid_DDPGPolicy_annealing_dinowm as DDPGPolicy
    from PyHJ.utils.net.common import Net
    from PyHJ.utils.net.continuous import Actor, Critic
    import gymnasium

    critic_net = Net(state_shape, action_shape, hidden_sizes=[512, 512, 512, 512],
                     activation=torch.nn.ReLU, concat=True, device=device)
    critic = Critic(critic_net, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3, weight_decay=1e-3)
    actor_net = Net(state_shape, hidden_sizes=[512, 512, 512, 512],
                    activation=torch.nn.ReLU, device=device)
    actor = Actor(actor_net, action_shape, max_action=1.0, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
    act_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=action_shape, dtype=np.float32)
    policy = DDPGPolicy(
        critic, critic_optim, tau=0.005, gamma=0.9999,
        exploration_noise=GaussianNoise(sigma=0.1), reward_normalization=False,
        estimation_step=1, action_space=act_space,
        actor=actor, actor_optim=actor_optim, actor_gradient_steps=1,
    )
    policy.load_state_dict(torch.load(hj_policy_path, map_location=device))
    policy.eval()
    return policy


def build_libero_dino_safety_filter(
    wm_ckpt_path, hj_policy_path=None, action_stats_path=C.ACTION_STATS_PATH,
    device="cuda:0", threshold=0.0,
    front_image_key="agentview_image", wrist_image_key="robot0_eye_in_hand_image",
    obs_state_key=None,
):
    """Load the DINO-WM (+ optional HJ critic) and return a ready safety adapter."""
    if not os.path.isfile(wm_ckpt_path):
        raise FileNotFoundError(f"DINO-WM ckpt not found: {wm_ckpt_path}")
    load_action_bounds(action_stats_path)

    wm = VideoTransformer(**C.WM_KWARGS)
    wm.load_state_dict(torch.load(wm_ckpt_path, map_location=device))
    wm = wm.to(device).eval()
    dino = wm.dino  # reuse the DINOv2 already loaded inside the world model

    policy = None
    if hj_policy_path is not None:
        policy = _build_hj_policy(hj_policy_path, device)

    hj_filter = DinoLatentHJSafetyFilter(
        wm=wm, dino=dino, policy=policy, action_dim=C.ACTION_DIM,
        threshold=threshold, device=device,
    )
    return LiberoDinoSafetyAdapter(
        hj_filter, front_image_key=front_image_key,
        wrist_image_key=wrist_image_key, obs_state_key=obs_state_key,
    )


def build_libero_safety_filter(
    configs_path=None, rssm_ckpt_path=None, hj_policy_path=None,
    device="cuda:0", threshold=0.0, image_size=128, image_key="agentview_image",
    obs_state_key=None, **kwargs,
):
    """Drop-in alias matching latent-safety's signature.

    `rssm_ckpt_path` is reinterpreted as the DINO-WM checkpoint path; `image_key`
    maps to the front camera. `configs_path` / `image_size` are accepted for
    call-site compatibility but unused (DINO-WM resizes to 224 internally).
    """
    if rssm_ckpt_path is None:
        raise ValueError("Pass the DINO-WM checkpoint via rssm_ckpt_path=...")
    return build_libero_dino_safety_filter(
        wm_ckpt_path=rssm_ckpt_path, hj_policy_path=hj_policy_path,
        action_stats_path=kwargs.get("action_stats_path", C.ACTION_STATS_PATH),
        device=device, threshold=threshold, front_image_key=image_key,
        wrist_image_key=kwargs.get("wrist_image_key", "robot0_eye_in_hand_image"),
        obs_state_key=obs_state_key,
    )
