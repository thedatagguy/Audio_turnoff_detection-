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

## Data preparation

`src/data_prep.py` streams `pipecat-ai/smart-turn-data-v3.2-train` (no need
to download the full 41.4GB up front) and curates a working subset:

```bash
uv run python src/data_prep.py --out-dir data/processed --max-scan 150000
```

- Applies per-language sample quotas (see `LANGUAGE_QUOTAS` in the script).
  **Hindi is weighted far above its natural frequency** (quota 12,000, vs.
  an estimated ~12k total available in the full dataset — i.e. we try to
  take essentially all of it) since it's the closest available proxy for
  the Hinglish target domain. Everything else is a thin general-multilingual
  baseline slice, not a training priority.
- The dataset has **no explicit Hinglish (code-switched) label** — only
  plain `language == "hin"`. The Hindi pool curated here is a proxy, not
  the real thing. A small hand-checked/supplemented Hinglish eval set is
  still a to-do (see Status below).
- Resamples every clip to 16kHz mono (Whisper's expected input) and writes
  WAV files to `data/processed/audio/`.
- Writes a stratified (by `language` + `endpoint_bool`) 80/10/10
  train/val/test split to `metadata_{train,val,test}.csv`, plus a
  `summary.json` with actual scanned/kept counts (the per-language numbers
  above are estimates from HF's *partial* dataset statistics — this script
  reports real counts from the actual streamed run).
- `--max-scan` is a safety cap on rows scanned (the stream isn't shuffled
  by language, so quotas may not all fill exactly); raise it if a run ends
  with under-filled quotas for languages you care about.

Verified end-to-end on a small scan (300 rows) before the full run: streaming,
resampling, WAV writing, and the stratified split all work correctly,
including the edge case where a language+label stratum has too few members
to split normally (handled via `safe_stratified_split`, which falls back to
an unstratified split for just those thin strata instead of crashing).

### Full run results (`--max-scan 150000`, 2026-08-20)

- **150,000 rows scanned → 34,755 clips kept**, 7.7GB on disk. File counts
  verified consistent: 34,755 WAVs = 27,804 train + 3,475 val + 3,476 test
  rows across the metadata CSVs.
- `endpoint_bool` stayed balanced in the kept set: 17,471 False / 17,284 True.
- **Hindi came in at 6,655 clips — under the 12,000 quota.** The scan hit its
  150,000-row cap before finding enough Hindi rows to fill it (the dataset
  isn't shuffled by language, so a 150k-row prefix doesn't sample every
  language proportionally). English hit its full 6,000-clip cap. Still,
  6,655 is ~4-6x the size of any other single non-English language pool
  here, so it remains the clear second-largest language and usable for the
  Hinglish-focused fine-tuning — raising `--max-scan` (or scanning the full
  270,946 rows) would close the gap further if the model needs more Hindi
  data after initial experiments.

## Model & training

**Architecture** (`src/model.py`, `src/dataset.py`, `src/train.py`):

- **Whisper-tiny encoder** (pretrained, the "ear") + **attention pooling** +
  a small MLP **endpoint head**. Only the encoder is reused — Whisper's text
  decoder is discarded; no text is ever generated. ~7.9M params total.
- The encoder normally expects a fixed 30s input (1500 positions). We rebuild
  it for an **8s window** by slicing the positional-embedding table
  (1500 -> 400) and copying every other pretrained weight unchanged, cutting
  wasted compute for our ~7s clips (supports the "tiny + fast" goal).
- Clips are **right-aligned to the last 8s** — the turn boundary lives at the
  END of the clip, so truncation drops the beginning, not the end. Features
  are Whisper-compatible log-mel spectrograms.
- **Auxiliary endfiller head** (multi-task), motivated directly by the EDA
  finding that endfiller=True => turn-not-ended 100% of the time. Masked for
  the ~14.5% of rows lacking the annotation.
- Loss = BCE(endpoint) + 0.3 * masked_BCE(endfiller). Discriminative LRs
  (1e-5 encoder / 1e-3 heads), bf16 autocast.

```bash
uv run python src/train.py --out-dir checkpoints/finetune --epochs 4 --batch-size 128 --num-workers 6
uv run python src/train.py --out-dir checkpoints/frozen   --epochs 4 --batch-size 128 --num-workers 6 --freeze-encoder
```

**Results** (validation, 4 epochs, ~4 min on the RTX PRO 5000; full
per-epoch/per-language history in `reports/training/`):

| Model | Val Acc | Val F1 | **Hindi Acc** |
|---|---|---|---|
| Frozen encoder (head-only baseline) | 0.871 | 0.877 | 0.853 |
| **Full fine-tune** | **0.901** | **0.906** | **0.922** |

- Fine-tuning the encoder buys ~3 F1 points overall and **~7 points on Hindi**
  — the biggest gain lands exactly on the target domain (Whisper's pretrained
  Hindi representation is weaker, so adapting it helps Hindi most). This is the
  core experiment justifying full fine-tuning over off-the-shelf features.
- Hindi improved every epoch (0.898 -> 0.904 -> 0.917 -> 0.922) and ends
  **above the overall average** — the Hindi-weighted curation paid off.
- Weakest languages: Vietnamese (~0.72), Chinese/Bengali/Marathi (~0.81) —
  low-resource/tonal languages lag, but they're out of scope for this
  Hindi-focused challenge.

> Note on a Windows gotcha fixed along the way: `DataLoader` workers spawn
> fresh on Windows and were stalling on HF Hub network calls while re-loading
> the feature extractor, starving the GPU (1% util). Fixed by forcing HF
> offline (model is cached) + `persistent_workers`. See `src/dataset.py`.

## Project layout

```
src/            training/inference/data-prep code
data/           local dataset cache / processed splits (gitignored)
checkpoints/    saved model weights (gitignored)
reports/        EDA + training results, write-up, figures
```

## Status

- [x] Environment scaffolded (uv, Python 3.12, CUDA-enabled torch, full stack verified)
- [x] Data loading + curation script (`src/data_prep.py`), verified end-to-end
- [x] Full data-prep pass run — 34,755 clips curated (`data/processed/`, see below)
- [x] EDA on curated data (`src/eda.py`, `reports/eda/`)
- [x] Model — Whisper-tiny encoder + attention pooling + endpoint/endfiller heads
- [x] Training loop + fine-tune vs frozen baseline experiment
- [ ] Test-set evaluation (final held-out metrics)
- [ ] Latency benchmark (CPU + GPU ms/inference)
- [ ] Gradio demo
- [ ] Supplement with a hand-checked Hinglish eval set (dataset has no native Hinglish label)
- [ ] Write-up
