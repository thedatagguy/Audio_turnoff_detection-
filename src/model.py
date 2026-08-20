"""
Turn-detection model: Whisper-tiny encoder (pretrained "ear") + a small
classification head we train ourselves.

Design choices (see README "Methodology"):
- We reuse only Whisper's ENCODER, not its text decoder. No text is ever
  generated; we read the turn-completion signal straight off the encoder's
  audio representation.
- The encoder normally expects a fixed 30s (3000 mel frames -> 1500 encoder
  positions). For a tiny+fast turn detector that's wasteful (our clips
  average ~7s). We rebuild the encoder for a shorter window by truncating
  the positional-embedding table and copying every other pretrained weight
  unchanged. Verified: outputs match the standard encoder's shape/behaviour
  on the first N positions.
- Two output heads: `endpoint` (main task) and `endfiller` (auxiliary,
  motivated by EDA showing endfiller=True => turn-not-ended 100% of the
  time). The auxiliary head is masked out for rows lacking the annotation.
"""

import copy

import torch
import torch.nn as nn
from transformers import WhisperConfig, WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder

WHISPER_NAME = "openai/whisper-tiny"
MEL_FRAMES_PER_SEC = 100  # hop_length=160 @ 16kHz
ENC_POS_PER_SEC = 50      # after conv stride 2


def build_encoder(max_seconds: float) -> tuple[WhisperEncoder, int]:
    """Load pretrained whisper-tiny encoder, rebuilt for a `max_seconds`
    window by slicing the positional-embedding table. Returns the encoder
    and the number of mel frames it expects as input."""
    full = WhisperModel.from_pretrained(WHISPER_NAME)
    enc_full = full.get_encoder()

    n_mel_frames = int(round(max_seconds * MEL_FRAMES_PER_SEC))
    max_src = n_mel_frames // 2  # encoder positions after conv stride 2

    if max_src >= full.config.max_source_positions:
        # Requested window >= 30s: just use the full pretrained encoder as-is.
        return enc_full, full.config.max_source_positions * 2

    cfg = copy.deepcopy(full.config)
    cfg.max_source_positions = max_src
    enc_small = WhisperEncoder(cfg)

    sd_full = enc_full.state_dict()
    new_sd = {}
    for k, v in sd_full.items():
        if k == "embed_positions.weight":
            new_sd[k] = v[:max_src].clone()
        else:
            new_sd[k] = v
    missing, unexpected = enc_small.load_state_dict(new_sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return enc_small, n_mel_frames


class AttentionPool(nn.Module):
    """Collapse the time axis (frames x d_model) into a single vector via a
    learned query attending over frames. Motivated by turn detection caring
    most about the final moments of the clip -- let the model learn which
    frames matter rather than averaging everything equally."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        scores = (x @ self.query) * self.scale       # (B, T)
        weights = torch.softmax(scores, dim=1)         # (B, T)
        return (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)


class TurnDetector(nn.Module):
    def __init__(
        self,
        max_seconds: float = 8.0,
        pooling: str = "attention",
        head_hidden: int = 128,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        use_endfiller_head: bool = True,
    ):
        super().__init__()
        self.encoder, self.n_mel_frames = build_encoder(max_seconds)
        d_model = self.encoder.config.d_model

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if pooling == "attention":
            self.pool = AttentionPool(d_model)
        elif pooling == "mean":
            self.pool = None
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling

        def make_head():
            return nn.Sequential(
                nn.Linear(d_model, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

        self.endpoint_head = make_head()
        self.use_endfiller_head = use_endfiller_head
        self.endfiller_head = make_head() if use_endfiller_head else None

    def forward(self, input_features: torch.Tensor) -> dict:
        # input_features: (B, 80, n_mel_frames)
        if self.freeze_encoder:
            with torch.no_grad():
                hidden = self.encoder(input_features).last_hidden_state
        else:
            hidden = self.encoder(input_features).last_hidden_state  # (B, T, D)

        if self.pooling == "mean":
            pooled = hidden.mean(dim=1)
        else:
            pooled = self.pool(hidden)

        out = {"endpoint_logit": self.endpoint_head(pooled).squeeze(-1)}
        if self.use_endfiller_head:
            out["endfiller_logit"] = self.endfiller_head(pooled).squeeze(-1)
        return out

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
