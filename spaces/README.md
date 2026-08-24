---
title: Audio Turn Detection
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: mit
short_description: Tiny Whisper-based turn detector for Hinglish speech
---

# 🎙️ Audio Turn Detection

Tiny (7.9M-param) Whisper-based model that decides whether a speaker has
**finished their turn** (respond now) or is just **pausing** (keep listening).
Tuned for Indian Hinglish, filler words, and natural pauses. ~25 ms per
decision on CPU.

- **Test accuracy:** 0.894 (F1 0.899); **Hindi:** 0.899
- Built on the Whisper-tiny encoder + attention pooling + endpoint/endfiller
  heads, fine-tuned on `pipecat-ai/smart-turn-data-v3.2-train`.
- Full code, EDA, training, and evaluation:
  see the accompanying GitHub repository.

The threshold slider exposes the precision/recall trade-off — raise it to make
the model more conservative about declaring "done", reducing the chance of
interrupting the speaker mid-sentence.
