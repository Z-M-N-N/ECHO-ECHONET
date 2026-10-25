"""Dataset / CSV utilities for the EchoNet-Dynamic regression subprojects.

The source of truth is ``Datasets/EchoNet-Dynamic/FileList.csv``:

    FileName, EF, ESV, EDV, FrameHeight, FrameWidth, FPS, NumberOfFrames, Split

``FileName`` is mapped to ``<video_root>/<FileName>.mp4``.  Splits use the
official TRAIN/VAL/TEST assignment (lower-cased internally).

Every item returned by :class:`VideoRegressionDataset` is::

    {"frames": uint8 [T,3,H,W],
     "labels": float32 [K] (NaN = missing),
     "meta": {"row_id", "file_name", "video_path", "split", ...}}

Frame sampling is deterministic uniform sampling, so a row always yields the
same frames for a given ``num_frames``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TASK_COLS: tuple[str, ...] = ("EF", "ESV", "EDV")
SPLITS: tuple[str, ...] = ("train", "val", "test")
# Optional per-record metadata kept in features / predictions when available.
OPTIONAL_META: tuple[str, ...] = (
    "FileName", "subject_id", "study_id", "measurement_id",
    "FrameHeight", "FrameWidth", "FPS", "NumberOfFrames",
)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def load_records(
    csv_path: str | Path,
    tasks: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    video_root: str | Path | None = None,
    video_ext: str = ".mp4",
    require_any_label: bool = True,
) -> pd.DataFrame:
    """Load ``FileList.csv`` and build absolute ``video_path`` values.

    ``row_id`` is the original CSV row index and is the join key between cached
    features, predictions and labels.
    """
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))

    if "split" not in df.columns:
        if "Split" not in df.columns:
            raise ValueError(f"no Split/split column in {csv_path}")
        df["split"] = df["Split"].astype(str).str.lower()
    else:
        df["split"] = df["split"].astype(str).str.lower()

    tasks = list(tasks) if tasks is not None else list(TASK_COLS)
    unknown = [t for t in tasks if t not in TASK_COLS]
    if unknown:
        raise ValueError(f"unknown task column(s): {unknown}; available={list(TASK_COLS)}")
    missing_cols = [t for t in tasks if t not in df.columns]
    if missing_cols:
        raise ValueError(f"missing label column(s) in {csv_path}: {missing_cols}")

    if "video_path" not in df.columns:
        if video_root is None:
            raise ValueError("CSV has no video_path; pass video_root")
        root = Path(video_root).expanduser()
        if "FileName" not in df.columns:
            raise ValueError("CSV has no FileName column")
        df["video_path"] = [str(root / f"{fn}{video_ext}") for fn in df["FileName"].astype(str)]

    bad_splits = sorted(set(df["split"]) - set(SPLITS))
    if bad_splits:
        raise ValueError(f"unexpected split value(s): {bad_splits}")

    if splits is not None:
        df = df[df["split"].isin(list(splits))].copy()
    if require_any_label:
        df = df[df[tasks].notna().any(axis=1)].copy()
    return df.reset_index(drop=True)


def label_stats(df: pd.DataFrame, tasks: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-task mean/std computed on the *train* split (NaN-safe)."""
    stats: dict[str, dict[str, float]] = {}
    for t in tasks:
        col = df.loc[df["split"] == "train", t].dropna().astype(float)
        mean = float(col.mean()) if len(col) else 0.0
        std = float(col.std(ddof=0)) if len(col) else 1.0
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        stats[t] = {"mean": mean, "std": std, "n_train": int(len(col))}
    return stats


def save_split_manifest(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("row_id", "FileName", "subject_id", "study_id", "measurement_id", "split") if c in df.columns]
    df[cols].to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Video decoding / sampling
# ---------------------------------------------------------------------------
def sample_indices(num_frames: int, count: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("video has no decodable frames")
    if count <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames >= count:
        idx = np.linspace(0, num_frames - 1, count)
    else:
        idx = np.concatenate([np.arange(num_frames), np.full(count - num_frames, num_frames - 1)])
    return np.clip(np.round(idx).astype(np.int64), 0, num_frames - 1)


def decode_video(path: str | Path, num_frames: int) -> np.ndarray:
    """Return ``uint8`` frames ``[T, H, W, 3]`` sampled uniformly over the clip."""
    path = str(path)
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        idx = sample_indices(len(vr), num_frames)
        frames = vr.get_batch(idx).asnumpy()
    except Exception:  # pragma: no cover - fallback only
        from torchvision.io import read_video

        vframes, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
        idx = torch.from_numpy(sample_indices(int(vframes.shape[0]), num_frames)).long()
        frames = vframes[idx].numpy()
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"unexpected decoded shape {frames.shape} for {path}")
    return np.ascontiguousarray(frames)


def resize_uint8_frames(frames: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(frames.shape[-2:]) == tuple(size):
        return frames
    out = torch.nn.functional.interpolate(
        frames.float(), size=tuple(size), mode="bilinear", align_corners=False, antialias=True
    )
    return out.round().clamp_(0, 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VideoRegressionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tasks: Sequence[str],
        num_frames: int = 16,
        image_size: int | tuple[int, int] | None = 224,
        check_paths: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tasks = list(tasks)
        self.num_frames = int(num_frames)
        self.meta_cols = [c for c in OPTIONAL_META if c in self.df.columns]
        if image_size is None:
            self.image_size = None
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)

        if check_paths:
            missing = [p for p in self.df["video_path"] if not Path(p).exists()]
            if missing:
                preview = "\n  ".join(missing[:5])
                raise FileNotFoundError(f"{len(missing)} video(s) missing, e.g.:\n  {preview}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        rec = self.df.iloc[i]
        frames = decode_video(rec["video_path"], self.num_frames)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        if self.image_size is not None:
            tensor = resize_uint8_frames(tensor, self.image_size)

        labels = torch.tensor(
            [np.nan if pd.isna(rec[t]) else float(rec[t]) for t in self.tasks], dtype=torch.float32
        )
        meta = {
            "row_id": int(rec["row_id"]),
            "file_name": str(rec["FileName"]) if "FileName" in rec.index else Path(rec["video_path"]).stem,
            "video_path": str(rec["video_path"]),
            "split": str(rec["split"]),
        }
        for c in self.meta_cols:
            if c == "FileName":
                continue
            val = rec[c]
            meta[c] = None if pd.isna(val) else (int(val) if float(val).is_integer() else float(val))
        return {"frames": tensor, "labels": labels, "meta": meta}


def collate_batch(items: list[dict]) -> dict:
    frames = torch.stack([it["frames"] for it in items], dim=0)
    labels = torch.stack([it["labels"] for it in items], dim=0)
    keys = items[0]["meta"].keys()
    meta = {k: [it["meta"][k] for it in items] for k in keys}
    return {"frames": frames, "labels": labels, "meta": meta}


def dataframe_from_records(
    df: pd.DataFrame, tasks: Sequence[str], indices: Iterable[int] | None = None
) -> pd.DataFrame:
    if indices is not None:
        df = df.iloc[list(indices)]
    return df.reset_index(drop=True)
