"""
Streamlit Community Cloud demo for the Audio Turn Detection model.

Deploys straight from this GitHub repo: Streamlit Cloud clones everything,
so we load the committed weights (checkpoints/finetune/best.pt) and reuse the
src/ inference code directly. Runs on CPU (~25ms/clip).

Local run:  uv run streamlit run streamlit_app.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from infer import TurnPredictor  # noqa: E402

CKPT = ROOT / "checkpoints" / "finetune" / "best.pt"


@st.cache_resource(show_spinner="Loading model (first run downloads Whisper-tiny)…")
def load_predictor():
    return TurnPredictor(str(CKPT), device="cpu")


def read_audio(file_bytes: bytes):
    """Decode uploaded/recorded audio bytes to (waveform, sample_rate)."""
    y, sr = sf.read(io.BytesIO(file_bytes))
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr


st.set_page_config(page_title="Audio Turn Detection", page_icon="🎙️")
st.title("🎙️ Audio Turn Detection")
st.markdown(
    "Tiny (7.9M-param) Whisper-based model that decides whether a speaker has "
    "**finished their turn** (respond now) or is just **pausing** (keep "
    "listening). Tuned for Indian **Hinglish**, filler words, and natural "
    "pauses. ~25 ms per decision on CPU.\n\n"
    "Record a clip that ends where you'd want a voice bot to decide — try "
    "finishing a sentence vs. cutting off mid-thought or on a filler like "
    "*“matlab…”* / *“umm…”*."
)

predictor = load_predictor()

threshold = st.slider(
    "Decision threshold (higher = more conservative, fewer interruptions)",
    0.05, 0.95, 0.50, 0.05,
)

tab_mic, tab_upload = st.tabs(["🎤 Record", "📁 Upload"])
audio_bytes = None
with tab_mic:
    rec = st.audio_input("Record a speech clip")
    if rec is not None:
        audio_bytes = rec.getvalue()
with tab_upload:
    up = st.file_uploader("Upload a clip", type=["wav", "flac", "ogg"])
    if up is not None:
        audio_bytes = up.getvalue()

if audio_bytes:
    y, sr = read_audio(audio_bytes)
    res = predictor.predict(y, sr)
    p = res["endpoint_prob"]
    ended = p >= threshold

    st.divider()
    if ended:
        st.success(f"🟢 **Turn ENDED — respond now**  ({p:.0%} confidence)")
    else:
        st.warning(f"🟡 **Still talking — keep listening**  ({1 - p:.0%} confidence)")

    st.progress(p, text=f"End-of-turn probability: {p:.1%}  (threshold {threshold:.0%})")

    cols = st.columns(2)
    cols[0].metric("Clip duration", f"{len(y) / sr:.1f}s",
                   help=f"Model uses the last {predictor.max_seconds:.0f}s")
    if "endfiller_prob" in res:
        ef = res["endfiller_prob"]
        cols[1].metric("Ends on a filler word?", f"{ef:.0%}",
                       help="High = trailing off on a filler, so probably NOT done")
        if ef > 0.5:
            st.caption("↳ The clip appears to end on a filler word — a strong "
                       "cue the speaker is still composing their thought.")

st.divider()
st.caption("Whisper-tiny encoder + attention pooling + endpoint/endfiller "
           "heads · test acc 0.894, Hindi 0.899 · see the GitHub repo for "
           "data prep, EDA, training, and evaluation.")
