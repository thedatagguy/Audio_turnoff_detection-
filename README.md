# Audio Turn Detection (Hinglish-focused)

Tiny, fast, audio-based **turn detection** model: given a short audio clip that
cuts off at some point, predict whether the speaker is genuinely done talking
(end of turn) or just paused mid-thought (not end of turn). Built for a
Shiprocket open challenge, with a focus on Indian Hinglish speech, filler
words, and natural pauses.

## Problem statement

A live voice bot has to decide, the instant a user stops talking, whether to
respond now or keep listening. A naive silence timer gets this wrong
constantly — e.g. "Can you send me the invoice by..." (paused mid-sentence)
looks identical to a silence-timer as "...send me the invoice by Friday."
(actually finished). This project trains a small classifier on raw audio to
make that call using prosody/content, not just pause length, with particular
attention to how this plays out in Hinglish speech (code-switching, filler
words like "matlab", "toh", "wo hi", different pause rhythms than English).

**Input:** an audio clip ending at some cutoff point.
**Output:** probability that the cutoff is a genuine end-of-turn.

## Dataset

[`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
— 270,946 clips, 41.4 GB, single `train` split.

Key columns:
- `audio` — the waveform
- `audioduration (s)` — 0.36s to 32.6s
- `language` — 23 languages (includes Hindi, not explicit Hinglish)
- `endpoint_bool` — **main label**: True = clip ends at a genuine turn boundary
- `midfiller` / `endfiller` — filler word present mid-clip / at the very end
  (endfiller ≈ strong "not actually done" signal, e.g. trailing off on "umm")
- `synthetic` — whether the clip is TTS-generated vs. real recording
- `dataset` — which of 12 source sub-datasets a row came from

No explicit Hinglish label exists in the base dataset — this is a gap we plan
to fill by filtering/oversampling the Hindi subset and, if needed,
supplementing with our own Hinglish clips or code-switched TTS for eval.

## Hardware (this machine)

| Resource | Spec |
|---|---|
| GPU | NVIDIA RTX PRO 5000 Blackwell, 48 GB VRAM, driver 573.42, CUDA 12.8 |
| CPU | Intel Xeon Gold 6530, 32 cores / 64 threads |
| RAM | 128 GB |
| Disk | ~5.7 TB free |

Comfortably sufficient for fine-tuning Whisper-tiny (39M params) or even
whisper-base/small if needed. The only real constraint hit during setup:
the system's default Python (3.14.3) is too new for current PyTorch wheels,
so the project is pinned to **Python 3.12** via `uv`.

## Environment setup

Dependency management is via [`uv`](https://github.com/astral-sh/uv).

```bash
# from the project root
uv sync
```

This creates `.venv` (Python 3.12, pinned via `.python-version`) and installs
everything pinned in `uv.lock`. PyTorch is installed from the CUDA 12.8 wheel
index (`https://download.pytorch.org/whl/cu128`) to match the driver on this
machine — see `[[tool.uv.index]]` in `pyproject.toml`.

To run any script:

```bash
uv run python src/<script>.py
```

### Installed stack (key packages, see `pyproject.toml`/`uv.lock` for full pins)

| Package | Version | Purpose |
|---|---|---|
| torch / torchaudio | 2.11.0+cu128 | model + audio tensor ops, GPU accel |
| transformers | 5.15.1 | Whisper encoder + model utilities |
| datasets | 2.19.1 | streaming/loading the HF dataset |
| accelerate | 1.14.0 | training loop device/precision handling |
| huggingface-hub | 1.28.0 | HF Hub access (model/dataset downloads) |
| librosa / soundfile | 1.0.0 / 0.14.0 | audio I/O, resampling |
| scikit-learn | 1.9.0 | metrics (F1, PR curves, etc.) |
| evaluate | 0.4.6 | metric utilities |
| gradio | 6.25.0 | demo UI |
| numpy / pandas / matplotlib | 2.5.2 / 3.0.5 / 3.11.1 | data wrangling + plots |

Note: `datasets==2.19.1` is noticeably older than `huggingface-hub==1.28.0`
(uv's resolver picked this pairing for compatibility). Verified working —
both import cleanly together and successfully stream real rows from
`pipecat-ai/smart-turn-data-v3.2-train` (see verification below). If this
pairing ever causes friction later (e.g. new dataset features), revisit with
`uv add "datasets>=X"` and re-resolve.

### Verifying the setup works

GPU visibility:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Confirmed output on this machine: `2.11.0+cu128 True NVIDIA RTX PRO 5000 Blackwell` (compute capability 12.0 / Blackwell / sm_120).

Dataset streaming (no full 41GB download needed to test):

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('pipecat-ai/smart-turn-data-v3.2-train', split='train', streaming=True)
print(next(iter(ds)).keys())
"
```

Both checks pass as of this setup (2026-08-20).

## Project layout

```
src/            training/inference/data-prep code
data/           local dataset cache / processed splits (gitignored)
checkpoints/    saved model weights (gitignored)
notebooks/      exploratory analysis
reports/        write-up, eval results, figures
```

## Status

- [x] Environment scaffolded (uv, Python 3.12, CUDA-enabled torch, full stack verified)
- [ ] Data loading + Hinglish subset curation
- [ ] Model (Whisper-tiny encoder + classification head)
- [ ] Training loop
- [ ] Evaluation (general + Hinglish-specific split, latency benchmark)
- [ ] Gradio demo
- [ ] Write-up
