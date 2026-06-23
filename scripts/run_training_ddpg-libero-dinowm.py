"""Train the latent HJ avoid critic/actor on the LIBERO DINO-WM world model.

Loads a DINO-WM checkpoint whose failure head is already trained
(dino_wm/checkpoints/best_classifier.pth), wraps it in the `libero_wm_DINO-v0`
latent env, and runs PyHJ avoid-DDPG. The first (warmup) episode fixes gamma=0
so the critic first fits the immediate margin, then gamma is annealed up.

    python scripts/run_training_ddpg-libero-dinowm.py \
        --wm_ckpt dino_wm/checkpoints/best_classifier.pth \
        --data /data/libero/consolidated.h5 \
        --logdir logs/libero_dinowm
"""
import argparse
import os
import sys

import gymnasium
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from dino_wm.dino_models import VideoTransformer, load_action_bounds
from dino_wm.test_loader import SplitTrajectoryDataset
import dino_wm.libero_config as C

from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing_dinowm as DDPGPolicy


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wm_ckpt", default="dino_wm/checkpoints/best_classifier.pth")
    p.add_argument("--data", default=C.CONSOLIDATED_TRAIN)
    p.add_argument("--action_stats", default=C.ACTION_STATS_PATH)
    p.add_argument("--logdir", default="logs/libero_dinowm")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup_eps", type=int, default=1)
    p.add_argument("--total_eps", type=int, default=15)
    p.add_argument("--warmup_steps", type=int, default=10000)
    p.add_argument("--steps_per_epoch", type=int, default=40000)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--critic_lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--buffer_size", type=int, default=40000)
    return p.parse_args()


def main():
    args = get_args()
    device = args.device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # LIBERO action normalization must match the world model's training.
    load_action_bounds(args.action_stats)

    wm = VideoTransformer(**C.WM_KWARGS)
    wm.load_state_dict(torch.load(args.wm_ckpt, map_location=device))
    wm = wm.to(device).eval()

    expert_data = SplitTrajectoryDataset(
        args.data, C.WM_KWARGS["num_frames"], split="train", num_test=0
    )

    def make_env():
        return gymnasium.make("libero_wm_DINO-v0", params=[wm, expert_data])

    env = make_env()
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    max_action = env.action_space.high[0]

    train_envs = DummyVectorEnv([make_env for _ in range(1)])
    test_envs = DummyVectorEnv([make_env for _ in range(1)])
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    critic_net = Net(
        state_shape, action_shape,
        hidden_sizes=[512, 512, 512, 512],
        activation=torch.nn.ReLU, concat=True, device=device,
    )
    critic = Critic(critic_net, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=1e-3)

    actor_net = Net(state_shape, hidden_sizes=[512, 512, 512, 512],
                    activation=torch.nn.ReLU, device=device)
    actor = Actor(actor_net, action_shape, max_action=max_action, device=device).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    policy = DDPGPolicy(
        critic, critic_optim,
        tau=0.005, gamma=0.9999,
        exploration_noise=GaussianNoise(sigma=0.1),
        reward_normalization=False, estimation_step=1,
        action_space=env.action_space,
        actor=actor, actor_optim=actor_optim, actor_gradient_steps=1,
    )

    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)

    def stop_fn(mean_rewards):
        return False

    epoch = 0
    for it in range(args.warmup_eps + args.total_eps):
        if it < args.warmup_eps:
            policy._gamma = 0          # warmup: fit immediate margin only
            policy.warmup = True
            steps = args.warmup_steps
        else:
            policy._gamma = args.gamma
            policy.warmup = False
            steps = args.steps_per_epoch

        epoch += 1
        epoch_dir = os.path.join(args.logdir, f"epoch_id_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        print(f"episode {it} -> {epoch_dir}")

        writer = SummaryWriter(epoch_dir)
        logger = WandbLogger()
        logger.load(writer)

        offpolicy_trainer(
            policy, train_collector, test_collector,
            1, steps, 8, 1, args.batch_size,
            update_per_step=0.125, stop_fn=stop_fn, logger=logger,
        )
        torch.save(policy.state_dict(), os.path.join(epoch_dir, "policy.pth"))


if __name__ == "__main__":
    main()
