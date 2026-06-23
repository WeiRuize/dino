# latent-safety-dino: LIBERO 上的 DINO-WM 安全过滤

本仓库是 `latent-safety`（RSSM 版）的 **DINO-WM 版本**：把世界模型从 Dreamer-v3 的
RSSM 换成 [DINO-WM](https://arxiv.org/abs/2411.04983)（DINOv2 patch 特征 + 视频
Transformer），并在 **LIBERO 机械臂** 上完成「离线学习世界模型 → 在 latent space 训练
HJ 安全 critic → 在 OpenVLA 在线 rollout 时过滤动作」的同一条三段式链路。

与 RSSM 版的主要区别：

| | RSSM 版 (latent-safety) | DINO-WM 版 (本仓库) |
|---|---|---|
| 世界模型 | Dreamer RSSM，单相机 agentview | DINO-WM 视频 Transformer，**双相机** |
| 相机 | `agentview_rgb` | front=`agentview_rgb` + wrist=`eye_in_hand_rgb` |
| latent | RSSM posterior（递归） | Transformer 固定上下文窗口（`num_frames` 帧） |
| 即时安全 | margin head | `failure_head`（正值更安全） |
| 安全 score | RSSM 想象一步 / HJ critic | DINO-WM 前推一步 / HJ critic |

> 双相机对齐：DINO-WM 的 `VideoTransformer` 本身就是双相机结构。我们把
> LIBERO 的 `agentview_rgb` 映射到 front（`cam_zed`），`eye_in_hand_rgb` 映射到
> wrist（`cam_rs`），因此模型与数据加载器（`test_loader.py`）无需改结构即可复用。

## 安装

推荐 Python 3.12 + Linux + CUDA GPU（DINOv2 / LIBERO / OpenVLA 都需要 GPU）。

```bash
cd latent-safety-dino
pip install -e .
conda install -c conda-forge ffmpeg
# 首次会通过 torch.hub 下载 facebookresearch/dinov2 (dinov2_vits14_reg)
```

## 配置入口：`dino_wm/libero_config.py`

整条管线的相机/字段名、张量维度、模型超参、默认路径都集中在
`dino_wm/libero_config.py`，**所有阶段共享同一份**，避免训练/评测维度不一致。
路径用环境变量覆盖，不用改源码：

```bash
export LIBERO_WM_DATA=/data/libero/consolidated.h5         # 唯一数据文件
export LIBERO_WM_TRAIN_FRAC=0.9                            # 训练占比，前 (1-frac) 做测试/eval
export LIBERO_WM_CKPT_DIR=dino_wm/checkpoints              # 权重 + 动作统计目录
export WANDB_MODE=disabled                                 # 不想用 wandb 时
```

> 世界模型训练（decoder / wm / classifier）只需要 **一份** `consolidated.h5`，
> 内部按 `TRAIN_FRAC` 切 train/test（尾部 `frac` 训练、头部 `1-frac` 测试），
> 与 RSSM 版 `dreamer_offline.py` 的 90/10 train/val 划分一致。

关键约定（必须保证训练与评测一致）：

- 动作 7 维；proprio `state = concat(ee_states, gripper_states)` 必须等于
  `STATE_DIM=8`。`libero_to_dataset.py` 会在维度不符时报错。
- `state` 在评测期由 live LIBERO obs 拼成 `[eef_pos(3), axis_angle(eef_quat)(3),
  gripper_qpos(2)]`。请保证你的 `ee_states` 与此**语义一致**（位置 3 + 轴角 3），
  否则世界模型在评测时看到的 proprio 与训练分布不匹配。
- 动作归一化范围来自数据：`libero_to_dataset.py` 计算逐维 min/max 存到
  `libero_action_stats.npz`，训练/HJ/评测都用 `load_action_bounds()` 载入它。

## 数据约定

原始 LIBERO HDF5（标准格式）：

```text
<file>.hdf5
  data/<demo_i>/
    actions                 # (T,7)
    dones                   # (T,) 可选
    failure                 # (T,) 安全标签，可选；由 label_libero.py 写入
    obs/
      agentview_rgb         # (T,H,W,3) front
      eye_in_hand_rgb       # (T,H,W,3) wrist
      ee_states             # 与 gripper_states 拼成 8 维 state
      gripper_states
```

`failure` 是 **安全标签**（碰撞 / 禁区 / 恶意目标达成），不是任务成功标签：
0=safe，1=unsafe，2=weak-unsafe（与分类器 `fail_loss` 对应）。

## 一键脚本

仓库根目录提供 4 个 shell 脚本，对应下面四个阶段；里面的数据集目录均为
`<FILL>` 占位，请按注释改成你自己的路径后运行：

```bash
bash prepare_data.sh          # ① 标注 + 转换数据
bash train_wm.sh              # ② decoder / world model / classifier
bash train_critic.sh          # ③ HJ 安全 critic
bash run_libero_eval_wm.sh    # ④ OpenVLA + DINO-WM 在线评测（需先做一行 import 替换，见脚本头注释）
```

## 三段式流程

```text
原始 LIBERO HDF5
  │  ① 数据准备
  ├─ dino_wm/label_libero.py      给每个 demo 写 failure 标签
  └─ dino_wm/libero_to_dataset.py 双相机 DINOv2 embedding + state + label
        → consolidated.h5 + libero_action_stats.npz
  │  ② 世界模型
  ├─ dino_wm/train_dino_decoder.py    训 VQVAE decoder（可视化用）
  ├─ dino_wm/train_dino_wm.py         训 DINO-WM 动力学（front/wrist/state 预测）
  ├─ dino_wm/train_dino_classifier.py 训 failure_head（即时安全 margin）
  └─ dino_wm/eval_dino_classifier.py  混淆矩阵检查 FP/FN
  │  ③ 安全 critic
  ├─ scripts/run_training_ddpg-libero-dinowm.py  在 libero_wm_DINO-v0 里训 HJ avoid critic
  └─ dino_wm/eval_dino_brt.py                    可视化 BRT
  │  ④ 在线评测
  └─ scripts/dino_safety_filter.py + OpenVLA LIBERO 评测
```

### ① 数据准备

```bash
cd dino_wm
# 1. 标注 failure（交互式：0 安全 / 1 危险 / 2 弱危险 / 空格回退）
python label_libero.py --raw_dir /data/libero_raw
#    已知整条安全的轨迹可直接：python label_libero.py --raw_dir /data/libero_raw --all_safe

# 2. 转成 DINO-WM 训练用的 consolidated 数据 + 动作统计（只需一份，训练时按比例切）
python libero_to_dataset.py --raw_dir /data/libero_raw --out $LIBERO_WM_DATA
```

### ② 训练世界模型

```bash
cd dino_wm
mkdir -p checkpoints
python train_dino_decoder.py      # -> checkpoints/testing_decoder.pth
python train_dino_wm.py           # -> checkpoints/best_testing.pth
python train_dino_classifier.py   # -> checkpoints/best_classifier.pth
python eval_dino_classifier.py    # 看 TP/FN/FP/TN，FP、FN 都低再继续
```

### ③ 训练 latent HJ 安全 critic

```bash
cd ..   # 回到仓库根目录
python scripts/run_training_ddpg-libero-dinowm.py \
    --wm_ckpt dino_wm/checkpoints/best_classifier.pth \
    --data $LIBERO_WM_DATA \
    --logdir logs/libero_dinowm
# 产物：logs/libero_dinowm/epoch_id_<N>/policy.pth

python dino_wm/eval_dino_brt.py   # 可视化 BRT（按需修改其中的 ckpt 路径）
```

### ④ OpenVLA + DINO-WM 安全过滤 在线评测

本仓库不内置 OpenVLA。在线评测复用 `latent-safety` 里已经跑通的 OpenVLA/LIBERO
评测主程序 `openvla/run_libero_eval_wm.py`，**只需把它的安全过滤后端从 RSSM 换成
DINO-WM**——`scripts/dino_safety_filter.py` 已提供与之**同名同接口**的
`build_libero_safety_filter(...)` 以及 adapter 的 `reset()/observe()/filter_action()`。

把评测脚本顶部的导入改为指向本仓库的过滤器（保证 `scripts/` 在 `PYTHONPATH` 上）：

```python
# 原: from libero_safety_filter import build_libero_safety_filter
from dino_safety_filter import build_libero_safety_filter
```

然后照常运行评测，把 `--rssm_ckpt_path` 指向 **DINO-WM 的 classifier 权重**、
`--hj_policy_path` 指向上一步的 `policy.pth`：

```bash
export PYTHONPATH=/path/to/latent-safety-dino/scripts:$PYTHONPATH
python /path/to/latent-safety/openvla/run_libero_eval_wm.py \
    --pretrained_checkpoint /data/openvla_weight \
    --task_suite_name libero_object \
    --enable_world_model_safety True \
    --rssm_ckpt_path /path/to/latent-safety-dino/dino_wm/checkpoints/best_classifier.pth \
    --hj_policy_path /path/to/latent-safety-dino/logs/libero_dinowm/epoch_id_16/policy.pth
```

接口约定：`safety_score` 正值更安全，`score <= threshold`（默认 0）时拒绝动作；
拒绝模式 `zero`（默认）/`clip`/`identity`。不传 `--hj_policy_path` 时退化为只用
`failure_head` 的一步 margin 打分。

## 修改/新增清单（相对原 Franka 版 dino 仓库）

新增：
- `dino_wm/libero_config.py`：管线共享常量与路径。
- `dino_wm/libero_to_dataset.py`：原始 LIBERO HDF5 → consolidated 数据 + 动作统计。
- `dino_wm/label_libero.py`：LIBERO demo 的 failure 标注。
- `PyHJ/reach_rl_gym_envs/libero-DINOwm.py`：LIBERO latent 环境（注册为 `libero_wm_DINO-v0`）。
- `scripts/run_training_ddpg-libero-dinowm.py`：LIBERO HJ critic 训练（无硬编码路径）。
- `scripts/dino_safety_filter.py`：评测期 DINO-WM 安全过滤器（与 RSSM 版同接口）。

改动：
- `dino_wm/dino_models.py`：`normalize_acs/unnormalize_acs` 改为可被
  `set_action_bounds()/load_action_bounds()` 覆盖；默认仍是 Franka 范围（不破坏旧脚本）。
- `dino_wm/train_dino_{decoder,wm,classifier}.py`：硬编码数据路径改为读
  `libero_config`；wm/classifier 启动时 `load_action_bounds()`。
- `PyHJ/reach_rl_gym_envs/__init__.py`：注册 `libero_wm_DINO-v0`。

Franka 相关文件（`franka-DINOwm.py`、`run_training_ddpg-dinowm.py`、
`hdf5_to_dataset.py`、`label.py`）保留未删，作为参考。

## 注意事项

1. 全程在 Linux + GPU 运行；本次代码改动在无 GPU 环境只做了语法级检查，**未做端到端运行验证**。
2. 训练（`num_frames`=3，segment_length=4）、HJ、评测必须用同一套
   `libero_config` 与同一份 `libero_action_stats.npz`。
3. 评测期 `state` 的拼法必须与转换期 `ee_states` 的语义一致，见上文「配置入口」。
4. 第④段依赖 `latent-safety` 的 OpenVLA/LIBERO 评测树（本仓库不含 OpenVLA）。
