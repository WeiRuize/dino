#!/usr/bin/env bash
# ============================================================================
# 阶段②：训练 DINO-WM 世界模型
#   1) train_dino_decoder.py    VQVAE decoder（仅用于可视化重建）
#   2) train_dino_wm.py         DINO-WM 动力学（front/wrist/state 预测）
#   3) train_dino_classifier.py failure_head（即时安全 margin，冻结其余权重微调）
#   4) eval_dino_classifier.py  在测试集上算混淆矩阵，检查 FP/FN
# 这几个脚本不吃命令行参数，路径/开关通过环境变量传入（见 dino_wm/libero_config.py）。
# 运行环境：Linux + GPU。
# ============================================================================
set -e
# 单卡默认用 GPU0；多卡时改成例如 "0,1,2,3" 并把 NUM_GPUS 设为卡数。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# 世界模型（train_dino_wm.py）支持 torchrun 多卡 DDP；NUM_GPUS>1 时用 torchrun 启动。
export NUM_GPUS="${NUM_GPUS:-1}"

# ---- 数据 / 权重 路径（与 prepare_data.sh 保持一致）-----------------------
export LIBERO_WM_DATA="/home/admin/data/libero/consolidated.h5"   # <FILL> 唯一数据文件，按比例切 train/test
export LIBERO_WM_TRAIN_FRAC="0.9"               # 训练占比；前 (1-frac) 条做测试/eval
export LIBERO_WM_CKPT_DIR="checkpoints"         # 相对 dino_wm/；存放 *.pth + 动作统计
export LIBERO_WM_ACTION_STATS="checkpoints/libero_action_stats.npz"
export WANDB_MODE="disabled"                    # 想看 wandb 曲线就删掉这一行并登录 wandb
# ---------------------------------------------------------------------------

cd dino_wm
mkdir -p checkpoints

# 1) decoder：脚本内默认 lr=3e-4, BS=64, train_iter=5000 -> checkpoints/testing_decoder.pth
#python train_dino_decoder.py

# 2) world model：默认 BS=16, BL=4(=num_frames+1), train_iter=100000,
#    transformer lr=5e-5 / action&embedding lr=5e-4 -> checkpoints/best_testing.pth
#    单卡：python；多卡：torchrun（NUM_GPUS=卡数，每卡各跑 BS 个样本 -> 有效 BS=BS*NUM_GPUS）。
if [ "${NUM_GPUS}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" train_dino_wm.py
else
  python train_dino_wm.py
fi

# 3) failure 分类器：加载 best_testing.pth，只训 failure_head，
#    默认 lr=5e-5, train_iter=10000 -> checkpoints/best_classifier.pth
python train_dino_classifier.py

# 4) 评测分类器混淆矩阵（FP/FN 都低再进入阶段③）
python eval_dino_classifier.py
