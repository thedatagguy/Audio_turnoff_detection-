"""
Dataset + feature extraction for turn detection.

Each item: load the 16kHz WAV, right-align it to a fixed `max_seconds`
window (keep the LAST N seconds -- the turn boundary lives at the end of the
clip, so if we must truncate we drop the beginning, not the end; shorter
clips are front-padded with zeros), then compute a Whisper-compatible
log-mel spectrogram via the official WhisperFeatureExtractor so the features
exactly match what the pretrained encoder was trained on.

Labels returned:
- endpoint: 1.0 if the turn ended (endpoint_bool), else 0.0
- endfiller: 1.0/0.0, or masked (endfiller_mask=0) when the source didn't
  annotate fillers (value is null in metadata).
"""

import os

# The whisper-tiny feature extractor is already cached locally. Force offline
# so from_pretrained never blocks on an HF Hub network call -- critical on
# Windows, where DataLoader workers spawn fresh and each re-imports this
# module; a network stall here starves the GPU (workers never produce a
# batch). Must be set before importing transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

SR = 16_000
WHISPER_NAME = "openai/whisper-tiny"


class TurnDataset(Dataset):
    def __init__(self, data_dir: str, split: str, max_seconds: float = 8.0):
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(self.data_dir / f"metadata_{split}.csv")
        self.max_seconds = max_seconds
        self.max_samples = int(round(max_seconds * SR))
        self.n_mel_frames = int(round(max_seconds * 100))
        self.fe = WhisperFeatureExtractor.from_pretrained(WHISPER_NAME)

    def __len__(self) -> int:
        return len(self.df)

    def _right_align(self, y: np.ndarray) -> np.ndarray:
        """Keep the last max_samples; front-pad with zeros if shorter."""
        if len(y) >= self.max_samples:
            return y[-self.max_samples :]
        pad = np.zeros(self.max_samples - len(y), dtype=y.dtype)
        return np.concatenate([pad, y])

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        y, sr = sf.read(self.data_dir / row["path"])
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        assert sr == SR, f"expected {SR}Hz, got {sr}"

        y = self._right_align(y)

        # WhisperFeatureExtractor on an exactly-N-second clip -> (80, N*100)
        feats = self.fe(
            y,
            sampling_rate=SR,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
        ).input_features[0]  # (80, n_mel_frames)

        endpoint = 1.0 if bool(row["endpoint_bool"]) else 0.0

        ef = row["endfiller"]
        if pd.isna(ef):
            endfiller, endfiller_mask = 0.0, 0.0
        else:
            endfiller, endfiller_mask = (1.0 if bool(ef) else 0.0), 1.0

        return {
            "input_features": feats,
            "endpoint": torch.tensor(endpoint, dtype=torch.float32),
            "endfiller": torch.tensor(endfiller, dtype=torch.float32),
            "endfiller_mask": torch.tensor(endfiller_mask, dtype=torch.float32),
            "language": row["language"],
        }


def collate(batch: list[dict]) -> dict:
    return {
        "input_features": torch.stack([b["input_features"] for b in batch]),
        "endpoint": torch.stack([b["endpoint"] for b in batch]),
        "endfiller": torch.stack([b["endfiller"] for b in batch]),
        "endfiller_mask": torch.stack([b["endfiller_mask"] for b in batch]),
        "language": [b["language"] for b in batch],
    }
