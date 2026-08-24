# Report outline — Audio Turn Detection

> Writing scaffold. Each section: what to cover + the exact numbers + why it
> matters. Write the prose yourself, in your own words — this is the skeleton
> and the facts, not the essay. Aim ~2–4 pages. Lead with decisions and
> reasoning, not just results (the brief cares about methodology > results).

---

## 1. Problem framing (½ page)
- What turn detection is: after a speaker pauses, decide **done vs just
  pausing** — a live voice bot has to call it in real time.
- Why a silence timer is not enough: "send me the invoice by…" (paused) looks
  identical to "…by Friday." (done) to a timer. Needs prosody/content, not
  pause length.
- The Hinglish angle: code-switching, filler words ("matlab", "toh", "umm"),
  different pause rhythms throw off English-centric models.
- Input → output in one line: **audio clip ending at a cutoff → probability
  the cutoff is a real end-of-turn.**

## 2. Data understanding & preparation (1 page — this is weighted heavily)
- Dataset: `pipecat-ai/smart-turn-data-v3.2-train` — 270,946 clips, 41.4 GB,
  23 languages. Labels: `endpoint_bool` (main), `midfiller`/`endfiller`,
  `synthetic`, `dataset` (12 sources).
- **Key decision — subset, not all 271k, and Hindi-weighted.** Why:
  - Natural distribution is English-heavy, Hindi only ~4–5%. Training on it
    as-is would barely see the target domain.
  - Curated 34,755 clips (scanned 150k rows). Hindi 6,655 (largest non-English
    pool, ~4–6× any other), English 6,000, rest 800–1,500.
  - Whisper-tiny + a head doesn't need 271k to converge; subset = faster
    iteration. Honest caveat: Hindi hit 6,655 not the 12k target because the
    stream isn't shuffled by language.
- 16 kHz mono resampling — why 16k (speech tops ~8 kHz; Nyquist; matches
  Whisper's training).
- Stratified 80/10/10 split (by language × endpoint). `endpoint_bool` balanced
  (17,471 F / 17,284 T).

## 3. EDA — findings that shaped the model (½–¾ page, use the figures)
- **Finding 1 — `endfiller` is a near-perfect signal:** ends-on-filler → NOT
  done, **100%** of the time. The dataset has dedicated `orpheus_endfiller_1`
  hard-negative sources. → motivated the auxiliary endfiller head.
  *(figure: `reports/eda/endfiller_vs_endpoint.png`)*
- **Finding 2 — Hindi is 100% synthetic (TTS);** English only 33%. A real
  limitation — model learns clean TTS Hindi, not messy real Hinglish. → the
  honest gap in the write-up.
- **Finding 3 — duration is near-identical across labels** (7.27 vs 7.37 s
  mean) → duration is not a shortcut; the model must actually listen.
- 14.5% of rows lack filler annotations → masked in the aux loss, not treated
  as False.
- Include: `waveform → log-mel spectrogram` figure to explain what the encoder
  actually sees (the audio-becomes-an-image step).

## 4. Approach & model (¾ page)
- **Transfer learning from Whisper-tiny.** Reuse only the **encoder** (the
  "ear"); discard the text decoder. No text is generated — we read the
  turn-completion signal straight off the audio representation.
- Architecture: encoder → **attention pooling** (learns which frames matter;
  turn boundary is at the end) → endpoint head + **auxiliary endfiller head**
  (multi-task, from EDA). **7.9M params total.**
- **8-second window** via slicing the positional-embedding table (1500→400) +
  copying other weights — cuts wasted compute vs Whisper's default 30 s
  (supports "tiny + fast").
- **Right-align to the last 8 s** — the decision boundary is at the clip end,
  so truncation drops the beginning, not the end. (A turn-detection-specific
  choice, opposite of generic Whisper usage.)
- Loss = BCE(endpoint) + 0.3·masked_BCE(endfiller); discriminative LR (low
  encoder 1e-5 / high heads 1e-3); bf16.

## 5. Experiments & results (¾ page — tables/figures)
- **Fine-tune vs frozen-encoder baseline** (the core experiment):

  | Model | Val Acc | Val F1 | Hindi Acc |
  |---|---|---|---|
  | Frozen (head-only) | 0.871 | 0.877 | 0.853 |
  | Full fine-tune | 0.901 | 0.906 | 0.922 |

  → fine-tuning buys ~3 F1 overall but **~7 points on Hindi** — biggest gain on
  the target domain (Whisper's pretrained Hindi is weaker). Data curation +
  modeling decision reinforce each other, *proven* by the baseline.
- **Convergence** (`reports/training/convergence.png`): val plateaus at ~0.90
  by ~step 150 (mid epoch 1) while train loss keeps falling → converges within
  1 epoch (expected for transfer learning); mild overfit after. Hindi is the
  one metric still climbing through all 4 epochs → justifies keeping them.
- **Held-out TEST** (untouched until the end): **0.894 acc / 0.899 F1**,
  Hindi **0.899**, endfiller head **0.956**. Matches val → no overfit to val.
  *(figure: `reports/eval/per_language_test_acc.png`)*
- **Error analysis:** confusion matrix skews to **false positives** (287 FP vs
  83 FN) — model leans toward declaring "done". In production a FP interrupts
  the user, so raise the decision threshold >0.5 to trade recall for precision
  (no retraining). Weakest langs: Bengali/Marathi ~0.75, Chinese 0.82 (out of
  scope).

## 6. Efficiency / "is it fast?" (¼ page)
- Latency at batch=1 (one real-time decision): **~25 ms CPU / ~11 ms GPU**
  end-to-end (feature extraction + forward). vs ~100–200 ms conversational
  budget → huge headroom, runs on CPU (no GPU needed in production).
- **7.9M params, 31.6 MB** checkpoint. → tiny + fast + accurate, all three.

## 7. Deployment (¼ page)
- Live demo on **Streamlit Community Cloud**:
  https://audioturnoff.streamlit.app/ — mic/upload → live decision with the
  threshold slider and endfiller readout.
- Note the pivot: HF Spaces now charges for Gradio compute, so moved to
  Streamlit's free tier (a real deployment-constraint decision worth
  mentioning).

## 8. Limitations & next steps (¼ page — shows judgment)
- Hindi training data is 100% synthetic TTS → build a small **real-Hinglish
  eval set** to measure the sim-to-real gap.
- Low-resource languages lag (Bengali/Marathi/Chinese) — more data or a bigger
  encoder needed to push past ~90%.
- Streaming: current model scores a fixed clip; production needs a sliding
  window running every ~N ms. Latency budget already supports it.
- Threshold calibration per deployment (precision/recall trade-off).

## Assets to reference
- `reports/eda/*.png` — language dist, endfiller vs endpoint, spectrograms
- `reports/training/convergence.png` — loss + val curves
- `reports/eval/per_language_test_acc.png`, `test_results.json`
- Repo: all code in `src/`, live demo link above.
