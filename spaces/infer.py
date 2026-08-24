"""
Reusable single-clip inference wrapper around a trained checkpoint.

Mirrors the training-time preprocessing exactly: mono 16kHz, right-align to
the last `max_seconds` (the turn boundary lives at the clip end), Whisper log-
mel features. Returns calibrated-ish probabilities via sigmoid on the logits.
"""

from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import TurnDetector  # noqa: E402

SR = 16_000


class TurnPredictor:
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        self.max_seconds = cfg.get("max_seconds", 8.0)
        self.max_samples = int(round(self.max_seconds * SR))
        self.model = TurnDetector(
            max_seconds=self.max_seconds,
            pooling=cfg.get("pooling", "attention"),
            use_endfiller_head=not cfg.get("no_endfiller_head", False),
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        from transformers import WhisperFeatureExtractor
        self.fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def _prep(self, y: np.ndarray, sr: int) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        # normalize int16 range if needed
        if np.abs(y).max() > 1.5:
            y = y / 32768.0
        if sr != SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        # right-align to last max_seconds
        if len(y) >= self.max_samples:
            y = y[-self.max_samples:]
        else:
            y = np.concatenate([np.zeros(self.max_samples - len(y), dtype=y.dtype), y])
        return y

    @torch.no_grad()
    def predict(self, y: np.ndarray, sr: int) -> dict:
        y = self._prep(y, sr)
        feats = self.fe(
            y, sampling_rate=SR, return_tensors="pt",
            padding="max_length", max_length=self.max_samples, truncation=True,
        ).input_features.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            out = self.model(feats)
        endpoint_prob = torch.sigmoid(out["endpoint_logit"].float()).item()
        result = {"endpoint_prob": endpoint_prob}
        if "endfiller_logit" in out:
            result["endfiller_prob"] = torch.sigmoid(out["endfiller_logit"].float()).item()
        return result
