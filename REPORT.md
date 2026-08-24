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

## Model development

I use **transfer learning on Whisper-tiny**. Whisper is an encoder–decoder
model; I use only the **encoder**, since our goal is not to generate text from
audio, so the decoder is not needed.

1. I use **attention pooling** over the encoder's time steps, which lets the
   model learn which moments matter rather than averaging everything equally.
2. I use an **8-second window** for training — if the clip is longer I keep the
   last 8 seconds, and if it is shorter I pad the beginning with zeros. I also
   shrink the positional embeddings from **1500 to 400**, since the clip
   duration is reduced to 8 s from Whisper's original 30 s training duration.

## Results

I trained two versions of the encoder: a **frozen** encoder (head-only) and a
**fully fine-tuned** encoder.

The fine-tuned version gave significantly better results than the frozen one —
especially on Hindi, where accuracy went from **85 to 92**, and overall
accuracy of the model increased from **87 to 90**.

| Model | Val Acc | Val F1 | Hindi Acc |
| :--- | :---: | :---: | :---: |
| Frozen encoder (head only) | 0.871 | 0.877 | 0.853 |
| Full fine-tune | 0.901 | 0.906 | 0.922 |

The end-to-end latency (median) is **11 ms on GPU** and **~25 ms on CPU**.

Conversational turn-taking has roughly a **100–200 ms budget** before the
latency feels buggy, so ~25 ms on CPU is very fast and does not require a GPU.

The entire model is **7.9M parameters (31.6 MB on disk)**, which balances both
accuracy and latency.
