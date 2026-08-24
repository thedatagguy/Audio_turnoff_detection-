# Audio Turn Detection for Hinglish Speech — Project Report

> DRAFT to rephrase in your own words. Every number is real and traceable to
> the code/results in this repo. Make it yours before submitting.

**Live demo:** https://audioturnoff.streamlit.app/ · **Code + weights:** (GitHub repo link)

---

## 1. The problem

Turn detection decides, the instant a speaker stops talking, whether they have
actually finished their turn or are just pausing mid-thought. It is a core
piece of any voice-AI pipeline: get it wrong and the assistant either
interrupts the user or leaves awkward silences.

The naive approach — a silence timer — fails because silence alone is
ambiguous. "Can you send me the invoice by…" (a mid-sentence pause) sounds, to
a timer, exactly like "…send me the invoice by Friday." (a finished thought).
Distinguishing them requires listening to the *content and prosody* of the
speech — intonation, rhythm, whether the grammar is complete — not just
measuring how long the pause is.

This is harder for Indian **Hinglish**, where code-switching between Hindi and
English, filler words ("matlab", "toh", "umm"), and different pause rhythms
break models tuned on clean English.

**Task:** given an audio clip that ends at some cutoff point, output the
probability that the cutoff is a genuine end-of-turn.

## 2. Data preparation

I used `pipecat-ai/smart-turn-data-v3.2-train` — 270,946 clips, 41.4 GB, 23
languages. The main label is `endpoint_bool` (did the turn end); auxiliary
labels flag filler words in the middle (`midfiller`) or at the very end
(`endfiller`) of a clip.

**Key decision: a curated subset, weighted toward Hindi — not the full 271k.**
In the natural distribution, English dominates and Hindi is only ~4–5% of
clips. Training on that as-is would produce a model that mostly learns English
turn-taking and barely sees the target domain. Instead I streamed the dataset
and applied per-language quotas that oversample Hindi far above its natural
frequency. The result: **34,755 clips** (7.7 GB), with Hindi at **6,655** — the
largest non-English pool by 4–6× — and English at 6,000. (Hindi fell short of
its 12k quota because the dataset stream is not shuffled by language and the
scan hit its row cap first; this is documented honestly and is easy to extend.)

A second reason for a subset: a Whisper-tiny encoder plus a small head does not
need 271k examples to converge, and a smaller set makes the full
prep→train→eval loop fast enough to iterate on several times.

Every clip was resampled to **16 kHz mono** — speech energy tops out around
8 kHz (so by Nyquist, 16 kHz captures it), and it matches what the Whisper
encoder expects. I split the data 80/10/10 into train/val/test, **stratified by
language × endpoint** so every split has the same balance. The final label
balance is even (17,471 not-ended / 17,284 ended).

## 3. Exploratory analysis — findings that shaped the model

Three findings from EDA directly influenced design decisions:

1. **`endfiller` is a near-perfect signal.** When a clip ends on a filler word,
   the turn is labelled *not ended* **100%** of the time. This is by design —
   the dataset includes dedicated hard-negative sources (`orpheus_endfiller_1`)
   built to teach "trailing off on a filler is never a real end of turn." This
   directly motivated adding an auxiliary filler-detection head.

2. **The Hindi data is 100% synthetic (TTS).** Every Hindi clip in the curated
   set is machine-generated, versus only 33% for English. This is an honest
   limitation: the model learns clean synthetic Hindi and may not fully
   generalise to messy real Hinglish.

3. **Clip duration does not leak the label.** Mean duration is nearly identical
   for ended vs not-ended clips (7.37 s vs 7.27 s). So the model cannot cheat
   by using length as a shortcut — it has to actually listen.

*(Reference figures: `reports/eda/endfiller_vs_endpoint.png`,
`reports/eda/waveform_spectrogram_*.png`.)*

## 4. Approach and model

**Transfer learning from Whisper-tiny.** Whisper is a speech model pretrained
on ~680k hours of audio across many languages, including Hindi. It has two
halves: an **encoder** that turns audio into a rich numerical representation,
and a **decoder** that generates text. I reuse only the **encoder** — the
"ear" — and discard the decoder entirely. No text is ever generated; the
turn-completion signal is read straight off the encoder's representation by a
small classifier I train myself.

**Architecture:** log-mel spectrogram → Whisper-tiny encoder → attention
pooling → two heads (endpoint + endfiller). Total: **7.9M parameters.**

Design choices specific to turn detection:
- **Attention pooling** over the encoder's time steps lets the model learn
  which moments matter, rather than averaging everything equally.
- **An 8-second window.** Whisper defaults to a fixed 30 s input, which wastes
  compute on ~7 s clips. I rebuilt the encoder for 8 s by slicing its
  positional-embedding table (1500→400 positions) and copying all other
  pretrained weights unchanged.
- **Right-alignment.** The turn boundary lives at the *end* of a clip, so when
  a clip is longer than 8 s I keep the last 8 s, not the first — the opposite
  of standard Whisper usage. Shorter clips are front-padded.
- **Multi-task loss:** `BCE(endpoint) + 0.3 · masked_BCE(endfiller)`. The
  filler term is masked for the ~14.5% of rows without that annotation.
  Encoder and heads use different learning rates (low for the pretrained
  encoder, high for the fresh heads).

## 5. Experiments and results

**Main experiment — does fine-tuning the encoder help?** I trained two
versions: a frozen-encoder baseline (only the head learns) and a full
fine-tune (encoder adapts too).

| Model | Val Acc | Val F1 | Hindi Acc |
|---|---|---|---|
| Frozen encoder (head only) | 0.871 | 0.877 | 0.853 |
| **Full fine-tune** | **0.901** | **0.906** | **0.922** |

Fine-tuning gained ~3 F1 points overall but **~7 points on Hindi
specifically.** The interpretation: Whisper saw relatively little Hindi in
pretraining, so its frozen representation of Hindi is weak; letting the encoder
adapt is exactly what improves the target domain. The data decision
(oversample Hindi) and the modelling decision (fine-tune) reinforce each other,
and the baseline comparison proves it rather than asserting it.

**Convergence.** With fine-grained logging, validation accuracy reaches its
~0.90 plateau by roughly step 150 — within the first epoch, before one full
pass through the data — while training loss keeps falling. That is textbook
transfer-learning behaviour: the pretrained encoder already understands speech,
so the head learns the turn boundary almost immediately. Epochs 2–4 are mild
overfitting on most languages, but Hindi keeps improving through all four
epochs, which justifies not stopping early. *(Figure:
`reports/training/convergence.png`.)*

**Held-out test set.** On the untouched 3,476-clip test split: **0.894
accuracy, 0.899 F1, Hindi 0.899**, and the endfiller head at **0.956** on
annotated clips. These match validation, confirming the model did not overfit
to the validation set during experimentation. *(Figure:
`reports/eval/per_language_test_acc.png`.)*

**Error analysis.** The errors are lopsided — 287 false positives vs 83 false
negatives. The model leans toward declaring "done" (recall 0.95 > precision
0.85). In a real voice bot a false positive means interrupting the user
mid-sentence, so a deployment would raise the decision threshold above 0.5 to
trade recall for precision — no retraining needed. The weakest languages are
Bengali and Marathi (~0.75) and Chinese (0.82), all outside this challenge's
Hindi focus.

## 6. Efficiency — tiny and fast

Measured at batch size 1 (a single real-time decision), with feature
extraction and model forward timed separately:

| | End-to-end latency (median) |
|---|---|
| GPU | ~11 ms |
| CPU | ~25 ms |

Conversational turn-taking has roughly a 100–200 ms budget before latency
feels unnatural, so ~25 ms on CPU leaves large headroom — and it means no GPU
is required in production. The model is **7.9M parameters, 31.6 MB** on disk.
Together with the ~90% accuracy, this satisfies all three requirements: tiny,
fast, and accurate.

## 7. Deployment

The model is deployed as a live Gradio-style demo on **Streamlit Community
Cloud**: https://audioturnoff.streamlit.app/. Users record from the mic or
upload a clip and see the end-of-turn decision, the probability, and the
endfiller readout, with a slider to adjust the decision threshold live.

A deployment note: Hugging Face Spaces now requires a paid plan to run Gradio
compute, so I moved to Streamlit's free tier, which deploys directly from the
GitHub repo (including the committed model weights).

## 8. Limitations and next steps

- **Synthetic Hindi data.** The Hindi training clips are entirely TTS. The
  clear next step is a small, hand-checked **real-Hinglish evaluation set** to
  measure the sim-to-real gap.
- **Low-resource languages** (Bengali, Marathi, Chinese) lag; closing that gap
  needs more data or a larger encoder.
- **Streaming.** The current model scores a fixed clip; a production system
  would run it in a sliding window every ~N ms. The latency budget already
  supports this.
- **Threshold calibration** per deployment, using the precision/recall lever
  from the error analysis.

## Summary

A 7.9M-parameter Whisper-tiny-based turn detector, fine-tuned on a
Hindi-weighted subset of smart-turn-data-v3.2, reaching ~0.90 test accuracy
(0.899 on Hindi) at ~25 ms per decision on CPU. The work's emphasis is on the
reasoning: curating data toward the target domain and proving it helped,
turning an EDA finding into an auxiliary head, and aligning the input window to
where the turn boundary actually is.
