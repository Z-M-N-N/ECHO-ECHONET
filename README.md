# ECHO-ECHONET — EchoNet-Dynamic 回归任务

基于 EchoNet-Dynamic 官方 `FileList.csv` 的 3 个回归目标（**EF / ESV / EDV**），构建三个可独立运行的子项目：

| 子项目 | 目录 | 视觉骨干 | 训练部分 | 特征 |
|---|---|---|---|---|
| A. Qwen 冻结视觉塔 + MLP | `model/qwen/linear_probe/` | Qwen2.5-VL-3B `visual.*`，冻结 | 仅 MLP | 一次性缓存 |
| B. Qwen LoRA + MLP | `model/qwen/lora/` | Qwen2.5-VL-3B `visual.*` | LoRA(qkv/proj) + MLP | 不缓存，端到端；结束后保存 pooled 特征 |
| C. VideoMAE 冻结 + MLP | `model/videomae/` | VideoMAE-base，冻结 | 仅 MLP | 一次性缓存 |

> Qwen 两个子项目**只加载视觉塔**（`config.json -> vision_config` + safetensors 中 `visual.*`，390 个张量，约 0.67B 参数），不加载语言模型。

## 数据

- `/home/mzhao/Datasets/EchoNet-Dynamic/FileList.csv`：10030 个视频。
- 官方划分：TRAIN 7465 / VAL 1288 / TEST 1277（代码内统一小写）。
- 回归目标：

| 目标 | train 均值 ± std | 范围 |
|---|---|---|
| EF（射血分数，%） | 55.78 ± 12.41 | 6.9 – 97.0 |
| ESV（收缩末期容积，mL） | 43.35 ± 35.97 | 4.4 – 612.5 |
| EDV（舒张末期容积，mL） | 91.24 ± 45.91 | 12.6 – 695.0 |

- 视频路径由 `video_root / f"{FileName}.mp4"` 生成；原始视频多为 112×112、50 fps、约 176 帧，统一 resize 到 224×224。
- 均匀采样固定帧数（可配置，默认 16 帧）。
- 标签按 **train split 的 mean/std 做 z-score**，loss 在标准化空间计算；评估反标准化回原单位。
- 三个子项目共用同一官方划分；每次运行保存 `split_manifest.csv`。

## 环境

```bash
source /home/mzhao/miniconda3/etc/profile.d/conda.sh && conda activate Qwen_back
```

已验证：Python 3.10 / torch 2.5.1+cu124 / transformers 4.57.0 / peft 0.20.0 / decord 0.6.0。

GPU 任务需指定卡号：

```bash
CUDA_VISIBLE_DEVICES=<卡号> python train.py --config config.yaml --device cuda
```

## 快速开始

### A. Qwen 冻结视觉塔 + MLP

```bash
cd model/qwen/linear_probe
python extract_features.py --config config.yaml --device cuda   # 可选；train.py 会自动补
python train.py --config config.yaml --device cuda
```

### B. Qwen LoRA + MLP

```bash
cd model/qwen/lora
python train.py --config config.yaml --device cuda
```

### C. VideoMAE 冻结 + MLP

```bash
cd model/videomae
python extract_features.py --config config.yaml --device cuda   # 可选；train.py 会自动补
python train.py --config config.yaml --device cuda
```

常用参数：`--seed`、`--epochs`、`--device cpu|cuda|auto`、`--run-name`、`--max-videos N`（调试）、
`--overwrite-features`、`--reuse-features`；提取脚本支持 `--num-shards/--shard-id/--merge`。

## 输出结构

```text
out/
├── features/
│   ├── qwen_linear_probe/<tag>.npz + .json
│   ├── qwen_lora/<run>.npz + .json
│   └── videomae/<tag>.npz + .json
├── results/<子项目>/
│   ├── <时间戳>_<run_name>/
│   │   ├── metrics.json            # 每任务 MAE/RMSE/R²（train/val/test）+ macro
│   │   ├── predictions.csv         # 长表：FileName/task/y_true/y_pred
│   │   ├── config.yaml / split_manifest.csv / label_stats.json / history.csv
│   │   ├── mlp_head.pt
│   │   └── adapter/                # 仅 LoRA
│   ├── latest -> <最近一次运行>
│   └── metrics.json / predictions.csv / config.yaml -> latest/*
└── figures/<子项目>/<时间戳>_<run_name>/
    ├── loss_curve.png/.pdf
    └── pred_vs_true_<task>.png/.pdf   # test；val 为 pred_vs_true_val_<task>
```

`pred_vs_true` 图包含 `y=x` 理想线、最小二乘拟合线，并在图内标注 n / MAE / RMSE / R² / 斜率 / 截距。

## 训练与评估设置

- MLP：LayerNorm → Linear → GELU → Dropout（2 层）→ 3 维输出。
- 损失：masked MSE（可切 Huber）；标签缺失不参与 loss/指标。
- 优化器：AdamW + cosine，早停基于 val masked loss。
- 指标：每任务 MAE / RMSE / R²，另给跨任务 macro 平均。
- 固定 seed（Python/NumPy/PyTorch/CUDA）；保存 config、split manifest、环境快照、git commit。
- 每次运行自动追加 `EXPERIMENT_LOG.md`。

## 日志

```bash
LOG=/tmp/echo_logs/logs_<任务名>.log
screen -dmS <任务名> bash -lc "python -u train.py --config config.yaml --device cuda > \"$LOG\" 2>&1"

tail -f /tmp/echo_logs/logs_<任务名>.log
grep -aE "epoch |\[val\]|\[test\]|early stopping|done|Traceback" /tmp/echo_logs/logs_<任务名>.log | tail -n 40
```

## 说明

- 本目录的 `task.txt` 是从 ECHO-MIMIC-IV 模板复制的，其中 `data/regression.csv`、`/home/mzhao/ECHO-MIMIC-IV/...`
  等路径为旧路径；本仓库已按 **EchoNet-Dynamic + /home/mzhao/ECHO-ECHONET** 适配。
- VideoMAE 权重实际位于 `/home/mzhao/模型权重/VideoMAE/pytorch_model.bin`
  （task.txt 中的 `VideoMAE_pth/` 子目录不存在）。
