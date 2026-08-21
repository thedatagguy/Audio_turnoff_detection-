"""
Gradio demo for the turn-detection model.

Record from the mic (or upload a clip) and the model decides whether the
speaker has finished their turn (respond now) or is just pausing (keep
listening). A threshold slider exposes the precision/recall trade-off:
raising it makes the model more conservative about declaring "done", which
in production reduces the chance of interrupting the user mid-sentence.

Usage:
    uv run python src/demo.py --ckpt checkpoints/finetune/best.pt
    uv run python src/demo.py --share      # public link
"""

import argparse
from pathlib import Path

import gradio as gr

import sys
sys.path.insert(0, str(Path(__file__).parent))
from infer import TurnPredictor  # noqa: E402

PREDICTOR: TurnPredictor | None = None


def analyze(audio, threshold):
    if audio is None:
        return {"No audio": 1.0}, "Record or upload a clip first."
    sr, y = audio
    res = PREDICTOR.predict(y, sr)
    p = res["endpoint_prob"]
    ended = p >= threshold

    # show both classes with their raw probabilities (independent of threshold)
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


def build_app():
    with gr.Blocks(title="Audio Turn Detection") as app:
        gr.Markdown(
            "# 🎙️ Audio Turn Detection\n"
            "Tiny Whisper-based model that decides whether a speaker has "
            "**finished their turn** or is just **pausing**. Tuned for Indian "
            "Hinglish, filler words, and natural pauses. Runs in ~25ms on CPU.\n\n"
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
    return app


def main():
    global PREDICTOR
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/finetune/best.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    print(f"Loading model from {args.ckpt} ...")
    PREDICTOR = TurnPredictor(args.ckpt, device=args.device)
    print(f"Ready on {PREDICTOR.device}.")

    build_app().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
