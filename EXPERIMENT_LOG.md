## [2026-10-25 14:00] ECHO-ECHONET 项目构建 + 流程验证

- 目的：基于 EchoNet-Dynamic FileList.csv 构建 Qwen 冻结 / Qwen LoRA / VideoMAE 冻结三个回归子项目。
- 执行命令：三个子项目均以 `--device cpu --max-videos 8~24 --epochs 1~2` 做小样本流程验证。
- 关键参数：EF/ESV/EDV 三任务；train-split z-score；masked MSE；VideoMAE 16 帧；Qwen 16 帧。
- 输入：/home/mzhao/Datasets/EchoNet-Dynamic/FileList.csv + Videos/*.mp4。
- 输出：验证产物在 /tmp/echo_logs/en_smoke_out/；仓库 out/ 保持为空。
- 结果/指标：数据读取、特征提取、MLP/LoRA 训练、metrics/predictions/图 全流程跑通；无正式指标。
- 结论与备注：正式实验需在用户指定 GPU 上运行。

## [2026-10-25 14:07] VideoMAE 冻结特征 + MLP 回归（videomae_reg）

- 目的：VideoMAE 冻结 backbone 提取特征，训练 MLP 多任务回归头，报告各任务 MAE/RMSE/R2。
- 执行命令：model/videomae/train.py --config model/videomae/config.yaml --device cuda
- 关键参数：seed=42, frames=16, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, best_epoch=108
- 输入：/home/mzhao/Datasets/EchoNet-Dynamic/FileList.csv
- 输出：/home/mzhao/ECHO-ECHONET/out/results/videomae/2026-10-25_14-04-49_videomae_reg
- 结果/指标：
```
test macro MAE=14.9370, RMSE=21.2590, R2=0.5394
task                        n        MAE       RMSE       R2
EF                       1277     6.0745     8.3409   0.5345
ESV                      1277    14.5960    22.4494   0.6048
EDV                      1277    24.1405    32.9866   0.4788
macro                            14.9370    21.2590   0.5394
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-ECHONET/out/features/videomae/videomae_16f_224px_mean.npz；最佳 epoch=108；详细指标见 metrics.json。

## [2026-10-25 14:16] Qwen2.5-VL 冻结视觉塔 + MLP 回归（qwen_linear_probe_reg）

- 目的：仅加载 Qwen2.5-VL 视觉塔并冻结，缓存 pooled 特征后训练 MLP 多任务回归头。
- 执行命令：model/qwen/linear_probe/train.py --config model/qwen/linear_probe/config.yaml --device cuda
- 关键参数：seed=42, frames=16, size=224, pooling=mean, hidden=[512, 256], lr=0.001, bs=128, epochs=200, best_epoch=63
- 输入：/home/mzhao/Datasets/EchoNet-Dynamic/FileList.csv
- 输出：/home/mzhao/ECHO-ECHONET/out/results/qwen_linear_probe/2026-10-25_14-04-49_qwen_linear_probe_reg
- 结果/指标：
```
test macro MAE=16.1808, RMSE=23.2731, R2=0.4264
task                        n        MAE       RMSE       R2
EF                       1277     7.0961     9.7948   0.3581
ESV                      1277    16.1183    25.2369   0.5006
EDV                      1277    25.3280    34.7877   0.4204
macro                            16.1808    23.2731   0.4264
```
- 结论与备注：特征缓存 /home/mzhao/ECHO-ECHONET/out/features/qwen_linear_probe/qwen2vl3b_16f_224px_mean_frozen.npz；LM 未加载；最佳 epoch=63。
