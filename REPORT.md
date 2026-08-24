# Audio Turn Detection

## Problem

In audio AI infra, the model has to decide whether the speaker is done
talking or just taking a pause in between.

This is an important problem to solve because the audio model has to hand the
person's speech to an LLM to get an answer. If we hand it over midway, it
creates a bad user experience from the user's point of view — and from the
company's point of view, every time the AI agent/model calls the LLM the cost
increases.

So we have to train a model that decides whether the speaker is done talking
or just taking a pause, considering the trade-off between latency and
accuracy.

## Data preparation

1. The original dataset has around **271k audio clips (42 GB)**, which is a lot
   for quick experimentation.
2. So I took a sample of the full data containing around **35k clips**, with
   **6,655 Hindi clips**.
3. Every clip was resampled to **16 kHz mono**, which matches the Whisper-tiny
   model — it was trained at the same rate.
4. The data is split **80/10/10 (train/val/test)**, stratified by language and
   endpoint so every split gets the same balance.
5. The final label balance is even: **17,471 not-ended vs 17,284 ended**.

## Findings from the data

1. **Endfiller is a perfect signal.** Whenever a clip ends on a filler word,
   the turn is labelled *not ended* 100% of the time. So alongside the main
   classification task I also predict whether the clip ends on a filler word.
   Our final prediction is therefore **two things**: whether the turn has ended,
   and whether the clip ends on a filler word.
2. **Hindi data is 100% synthetic**, while English is 33% synthetic.
3. **Clip duration does not leak any information** — the mean duration for
   ended and not-ended clips is almost identical (**7.37 s ended vs 7.27 s
   not-ended**).
4. **5,050 clips had no label in the filler column**, so those unlabeled clips
   were **masked out of the auxiliary (endfiller) loss** — they still train the
   main endpoint task, but contribute nothing to the filler head.

## Model development

I use **transfer learning on Whisper-tiny**. Whisper is an encoder–decoder
model; I use only the **encoder**, since our goal is not to generate text from
audio, so the decoder is not needed.

1. I use **attention pooling** over the encoder's time steps, which lets the
   model learn which moments matter rather than averaging everything equally.
2. I use an **8-second window** for training — if the clip is longer I keep the
   last 8 seconds, and if it is shorter I pad the beginning with zeros. This is
   motivated by the data: the median clip is ~6.8 s and the 75th percentile is
   ~9.2 s, so 8 s fits most clips whole while trimming little. I also shrink the
   positional embeddings from **1500 to 400**, since the clip duration is
   reduced to 8 s from Whisper's original 30 s training duration.
3. **Two output heads** share the pooled encoder representation: the main
   `endpoint` head and the auxiliary `endfiller` head. The training objective is
   `loss = BCE(endpoint) + 0.3 × masked_BCE(endfiller)`, and I use
   **discriminative learning rates** — a low rate for the pretrained encoder and
   a higher rate for the freshly initialised heads.

## Results

I trained two versions of the encoder: a **frozen** encoder (head-only) and a
**fully fine-tuned** encoder.

The fine-tuned version gave significantly better results than the frozen one —
especially on Hindi, where validation accuracy went from **85 to 92**, and
overall validation accuracy increased from **87 to 90**.

| Model | Val Acc | Val F1 | Hindi Acc |
| :--- | :---: | :---: | :---: |
| Frozen encoder (head only) | 0.871 | 0.877 | 0.853 |
| Full fine-tune | 0.901 | 0.906 | 0.922 |

### Held-out test set (final, never used in training)

Validation was used only to select the best checkpoint, so the honest final
numbers come from the **3,476-clip test set** (fine-tuned model):

| Metric | Test |
| :--- | :---: |
| Accuracy | **0.894** |
| F1 | **0.899** |
| Precision | 0.851 |
| Recall | **0.952** |
| Hindi accuracy | **0.899** |
| Endfiller head accuracy (on annotated clips) | 0.956 |

**Confusion matrix:** TN 1461, FP 287, FN 83, TP 1645.

The model is deliberately **recall-heavy** (recall 0.952 vs precision 0.851):
it catches almost every real end-of-turn, at the cost of some false alarms.
For a voice assistant this is the right bias — replying a beat early is better
than leaving a long, awkward silence.

**Decision threshold.** The model outputs an end-of-turn *probability*; the
deployed demo exposes the cutoff as a slider. All metrics above use the default
**0.50** threshold. Raising it makes the model more conservative (higher
precision, fewer interruptions, slightly slower to respond); lowering it makes
it more eager (higher recall, faster, more interruptions).

### Latency and size

The end-to-end latency (median) is **11 ms on GPU** and **~25 ms on CPU**.

Conversational turn-taking has roughly a **100–200 ms budget** before the
latency feels buggy, so ~25 ms on CPU is very fast and does not require a GPU.

The entire model is **7.9M parameters (31.6 MB on disk)**, which balances both
accuracy and latency.

## Limitations

- The Hindi training data is **100% synthetic (TTS-generated)**, so real-world
  Hinglish performance with natural speakers is not yet directly validated.
- The dataset has no explicit code-switched "Hinglish" label; plain Hindi is
  used as the closest available proxy.
