"""
Hugging Face Spaces entry point for the Audio Turn Detection demo.

Spaces runs this file; the module-level `demo` Gradio app is auto-served.
Loads the fine-tuned weights shipped in this Space (best.pt) and runs on CPU
(free-tier Spaces have no GPU) — inference is ~25ms/clip, so CPU is plenty.
"""

import gradio as gr

from infer import TurnPredictor

CKPT = "best.pt"
PREDICTOR = TurnPredictor(CKPT, device="cpu")


def analyze(audio, threshold):
    if audio is None:
        return {"No audio": 1.0}, "Record or upload a clip first."
    sr, y = audio
    res = PREDICTOR.predict(y, sr)
    p = res["endpoint_prob"]
    ended = p >= threshold

    label = {"🟢 Turn ENDED — respond now": p,
             "🟡 Still talking — keep listening": 1 - p}

    dur = len(y) / sr
    lines = [
        f"**Decision:** {'🟢 Turn ENDED — respond' if ended else '🟡 Still talking — wait'}",
        f"**End-of-turn probability:** {p:.1%}  (threshold {threshold:.0%})",
        f"**Clip duration:** {dur:.1f}s  (model uses the last {PREDICTOR.max_seconds:.0f}s)",
    ]
    if "endfiller_prob" in res:
        ef = res["endfiller_prob"]
        lines.append(f"**Ends on a filler word?** {ef:.1%} "
                     f"{'— likely trailing off, so probably NOT done' if ef > 0.5 else ''}")
    return label, "\n\n".join(lines)


with gr.Blocks(title="Audio Turn Detection") as demo:
    gr.Markdown(
        "# 🎙️ Audio Turn Detection\n"
        "Tiny Whisper-based model that decides whether a speaker has "
        "**finished their turn** or is just **pausing**. Tuned for Indian "
        "Hinglish, filler words, and natural pauses. ~25ms per decision on CPU.\n\n"
        "Record a clip that ends where you'd want a voice bot to decide — "
        "try finishing a sentence vs. cutting off mid-thought or on a filler "
        "like *\"matlab...\"* / *\"umm...\"*."
    )
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["microphone", "upload"], type="numpy",
                             label="Speech clip")
            threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05,
                                  label="Decision threshold (higher = more "
                                        "conservative, fewer interruptions)")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_label = gr.Label(label="Prediction", num_top_classes=2)
            out_detail = gr.Markdown()

    btn.click(analyze, inputs=[audio, threshold], outputs=[out_label, out_detail])
    audio.stop_recording(analyze, inputs=[audio, threshold],
                         outputs=[out_label, out_detail])


if __name__ == "__main__":
    demo.launch()
