#!/usr/bin/env bash
# ============================================================================
# 阶段①：LIBERO 原始 HDF5 -> DINO-WM 训练数据
#   1) label_libero.py     给每个 demo 写 failure 安全标签 (0 安全/1 危险/2 弱危险)
#   2) libero_to_dataset.py 双相机 DINOv2 embedding + state + label -> consolidated.h5
#                           并写出动作统计 libero_action_stats.npz
# 只需准备 **一份** 数据：世界模型训练时按 TRAIN_FRAC 自动切 train/test
# （见 train_wm.sh / libero_config.py），无需单独的测试集文件。
# 需在 Linux + GPU 上运行（DINOv2 编码需要 GPU）。
# ============================================================================
set -e
export CUDA_VISIBLE_DEVICES=0

# ---- 占位路径：改成你自己的目录 -------------------------------------------
RAW_DIR="/home/admin/data/libero_hj_labeled/libero_object_no_noops"                # <FILL> 原始 LIBERO demo (*.hdf5) 所在目录
OUT="/data/libero/consolidated.h5"        # <FILL> 转换后的数据（train/test 由比例切分共用此文件）
CKPT_DIR="dino_wm/checkpoints"            # 权重 + 动作统计目录（相对仓库根）
# ---------------------------------------------------------------------------

mkdir -p "$(dirname "$OUT")" "$CKPT_DIR"

cd dino_wm

# 1) 标注 failure。交互式：0 安全 / 1 危险 / 2 弱危险 / 空格回退。
#    已确认整段安全的轨迹可改用 --all_safe 跳过人工标注。
python label_libero.py --raw_dir "$RAW_DIR"
# python label_libero.py --raw_dir "$RAW_DIR" --all_safe   # 全部标为安全

# 2) 转换 + 计算动作统计（写到 checkpoints/libero_action_stats.npz，训练/HJ/评测共用）。
python libero_to_dataset.py \
  --raw_dir "$RAW_DIR" \
  --out "$OUT" \
  --action_stats "checkpoints/libero_action_stats.npz" \
  --device cuda:0
