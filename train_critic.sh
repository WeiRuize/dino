#!/usr/bin/env bash
# ============================================================================
# 阶段③：在 DINO-WM latent space 训练 HJ avoid 安全 critic / actor
#   载入带 failure_head 的世界模型 (best_classifier.pth)，包进 libero_wm_DINO-v0
#   环境，跑 PyHJ avoid-DDPG。首个 warmup episode 固定 gamma=0 先拟合即时 margin。
#   产物：logs/libero_dinowm/epoch_id_<N>/policy.pth
# 运行环境：Linux + GPU。
# ============================================================================
set -e
export CUDA_VISIBLE_DEVICES=0

# ---- 路径（与前两段一致）--------------------------------------------------
WM_CKPT="dino_wm/checkpoints/best_classifier.pth"   # 阶段②产出的世界模型
DATA="/data/libero/consolidated.h5"                 # <FILL> 训练集（与 train_wm.sh 同一份）
ACTION_STATS="dino_wm/checkpoints/libero_action_stats.npz"
LOGDIR="logs/libero_dinowm"
# ---------------------------------------------------------------------------

# 训练超参默认值取自脚本/原 Franka 代码（critic_lr=1e-3, actor_lr=1e-4,
# tau=0.005, gamma=0.95, batch=512, buffer=40000, warmup=1 ep/1e4 步,
# 之后 15 ep/4e4 步）。不确定的值先用默认，按需在此覆盖。
python scripts/run_training_ddpg-libero-dinowm.py \
  --wm_ckpt "$WM_CKPT" \
  --data "$DATA" \
  --action_stats "$ACTION_STATS" \
  --logdir "$LOGDIR" \
  --device cuda:0 \
  --seed 0 \
  --warmup_eps 1 \
  --total_eps 15 \
  --warmup_steps 10000 \
  --steps_per_epoch 40000 \
  --gamma 0.95 \
  --actor_lr 1e-4 \
  --critic_lr 1e-3 \
  --batch_size 512 \
  --buffer_size 40000

# 可选：可视化 BRT（注意 eval_dino_brt.py 内部目前是写死的 ckpt 路径，按需修改后再跑）
# cd dino_wm && python eval_dino_brt.py
