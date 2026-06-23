#!/usr/bin/env bash
# ============================================================================
# 阶段④：OpenVLA + DINO-WM 安全过滤 在线评测
#
# 本仓库不含 OpenVLA。评测复用 latent-safety（RSSM 版）已经跑通的评测主程序
# openvla/run_libero_eval_wm.py，只把它的安全过滤后端换成 DINO-WM：
#
#   前置一次性改动：把 latent-safety/openvla/run_libero_eval_wm.py 顶部
#       from libero_safety_filter import build_libero_safety_filter
#   改成
#       from dino_safety_filter import build_libero_safety_filter
#   并保证本仓库 scripts/ 在 PYTHONPATH 上（下面已 export）。
#
# dino_safety_filter 与 RSSM 版同接口：--rssm_ckpt_path 复用为「DINO-WM 权重」，
# --hj_policy_path 指向阶段③的 policy.pth；不传 hj_policy 时退化为 failure_head 一步打分。
# 运行环境：Linux + GPU。
# ============================================================================
set -e
export CUDA_VISIBLE_DEVICES=0

# 本仓库根目录（含 scripts/dino_safety_filter.py 与 dino_wm/）
DINO_REPO="/path/to/latent-safety-dino"                 # <FILL>
LATENT_SAFETY_REPO="/path/to/latent-safety"             # <FILL> RSSM 版仓库（提供 OpenVLA 评测脚本）

# 让评测脚本能 import 到 DINO 过滤器；如评测依赖 BadLIBERO，也在此前置其路径
export PYTHONPATH="${DINO_REPO}/scripts:${DINO_REPO}:${PYTHONPATH}"
# export PYTHONPATH="/path/to/BadLIBERO:${PYTHONPATH}"  # <FILL> 如需修改版 LIBERO/BDDL

# ---- 评测占位参数 ----------------------------------------------------------
OPENVLA_CKPT="/data/openvla_weight"                     # <FILL> OpenVLA 权重目录
TASK_SUITE="libero_object"                              # <FILL> libero_object / libero_spatial / ...
NUM_TRIALS=20                                           # 每个任务 rollout 次数
WM_CKPT="${DINO_REPO}/dino_wm/checkpoints/best_classifier.pth"   # DINO-WM 权重
HJ_POLICY="${DINO_REPO}/logs/libero_dinowm/epoch_id_16/policy.pth"  # <FILL> 选用的 epoch
# BDDL_DIR="/path/to/BadLIBERO/libero/libero/bddl_files"  # <FILL> 如评测脚本需要
# ---------------------------------------------------------------------------

python "${LATENT_SAFETY_REPO}/openvla/run_libero_eval_wm.py" \
  --pretrained_checkpoint "$OPENVLA_CKPT" \
  --task_suite_name "$TASK_SUITE" \
  --num_trials_per_task "$NUM_TRIALS" \
  --enable_world_model_safety True \
  --rssm_ckpt_path "$WM_CKPT" \
  --hj_policy_path "$HJ_POLICY" \
  --use_wandb False \
  --run_id_note dino_wm_libero_eval
  # --bddl_dir "$BDDL_DIR" \                # <FILL> 按评测脚本实际需要补充
