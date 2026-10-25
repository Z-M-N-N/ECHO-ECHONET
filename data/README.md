# 数据位置

本项目的唯一数据源是 EchoNet-Dynamic 官方划分：

- `FileList.csv`：`/home/mzhao/Datasets/EchoNet-Dynamic/FileList.csv`
- 视频目录：`/home/mzhao/Datasets/EchoNet-Dynamic/Videos/*.mp4`

`FileList.csv` 字段：`FileName, EF, ESV, EDV, FrameHeight, FrameWidth, FPS, NumberOfFrames, Split`。

数据不复制到本仓库；`model/*/config.yaml` 中的 `csv` 与 `video_root` 指向上述绝对路径。
