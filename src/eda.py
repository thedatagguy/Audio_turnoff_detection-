"""
Exploratory analysis of the curated data/processed subset, plus a visual
walkthrough of how a raw audio clip becomes model input (waveform -> log-mel
spectrogram), since that's the actual mechanism Whisper's encoder operates on.

Outputs everything to reports/eda/ as PNGs, and a numeric summary as
reports/eda/eda_summary.json.

Usage:
    uv run python src/eda.py --data-dir data/processed --out-dir reports/eda
"""

import argparse
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


def load_all_metadata(data_dir: Path) -> pd.DataFrame:
    parts = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(data_dir / f"metadata_{split}.csv")
        df["split"] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--out-dir", default="reports/eda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_metadata(data_dir)
    print(f"Loaded {len(df)} rows across train/val/test")

    summary = {}
    summary["total_clips"] = len(df)
    summary["split_counts"] = df["split"].value_counts().to_dict()

    # --- language distribution ---
    lang_counts = df["language"].value_counts()
    summary["language_counts"] = lang_counts.to_dict()

    fig, ax = plt.subplots(figsize=(10, 6))
    lang_counts.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Clips per language (curated subset)")
    ax.set_xlabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "language_distribution.png", dpi=120)
    plt.close(fig)

    # --- endpoint_bool balance, overall and per language ---
    df["endpoint_bool"] = df["endpoint_bool"].astype(bool)
    summary["endpoint_bool_overall"] = df["endpoint_bool"].value_counts().to_dict()

    endpoint_by_lang = df.groupby("language")["endpoint_bool"].mean().sort_values()
    summary["endpoint_true_rate_by_language"] = endpoint_by_lang.round(3).to_dict()

    fig, ax = plt.subplots(figsize=(10, 6))
    endpoint_by_lang.plot(kind="barh", ax=ax, color="steelblue")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Fraction of clips labeled 'turn ended' (endpoint_bool=True), by language")
    ax.set_xlabel("fraction True")
    fig.tight_layout()
    fig.savefig(out_dir / "endpoint_rate_by_language.png", dpi=120)
    plt.close(fig)

    # --- duration distribution ---
    summary["duration_stats_overall"] = df["duration_s"].describe().round(3).to_dict()
    summary["duration_stats_by_endpoint"] = (
        df.groupby("endpoint_bool")["duration_s"].describe().round(3).to_dict()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        df.loc[df["endpoint_bool"], "duration_s"],
        bins=50, alpha=0.6, label="endpoint_bool=True (turn ended)", range=(0, 20),
    )
    ax.hist(
        df.loc[~df["endpoint_bool"], "duration_s"],
        bins=50, alpha=0.6, label="endpoint_bool=False (mid-turn)", range=(0, 20),
    )
    ax.set_xlabel("clip duration (s)")
    ax.set_ylabel("count")
    ax.set_title("Clip duration distribution by label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "duration_distribution.png", dpi=120)
    plt.close(fig)

    # --- filler labels vs endpoint_bool (key hypothesis: endfiller correlates with NOT ended) ---
    filler_df = df.dropna(subset=["midfiller", "endfiller"]).copy()
    summary["rows_with_filler_annotations"] = len(filler_df)
    summary["rows_missing_filler_annotations"] = len(df) - len(filler_df)

    if len(filler_df) > 0:
        filler_df["endfiller"] = filler_df["endfiller"].astype(bool)
        filler_df["midfiller"] = filler_df["midfiller"].astype(bool)
        endfiller_crosstab = pd.crosstab(
            filler_df["endfiller"], filler_df["endpoint_bool"], normalize="index"
        )
        summary["endpoint_true_rate_by_endfiller"] = endfiller_crosstab[True].round(3).to_dict()

        midfiller_crosstab = pd.crosstab(
            filler_df["midfiller"], filler_df["endpoint_bool"], normalize="index"
        )
        summary["endpoint_true_rate_by_midfiller"] = midfiller_crosstab[True].round(3).to_dict()

        fig, ax = plt.subplots(figsize=(6, 5))
        endfiller_crosstab[True].plot(kind="bar", ax=ax, color=["salmon", "seagreen"])
        ax.set_xticklabels(["endfiller=False", "endfiller=True"], rotation=0)
        ax.set_ylabel("fraction with endpoint_bool=True")
        ax.set_title("Does ending on a filler word predict an unfinished turn?")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out_dir / "endfiller_vs_endpoint.png", dpi=120)
        plt.close(fig)

    # --- synthetic vs real ---
    df["synthetic"] = df["synthetic"].astype(bool)
    summary["synthetic_counts"] = df["synthetic"].value_counts().to_dict()
    summary["synthetic_by_language"] = (
        df.groupby("language")["synthetic"].mean().round(3).to_dict()
    )

    # --- source dataset breakdown ---
    summary["source_dataset_counts"] = df["source_dataset"].value_counts().to_dict()

    # --- Hindi-specific slice, since that's the target domain ---
    hin = df[df["language"] == "hin"]
    summary["hindi_subset"] = {
        "count": len(hin),
        "endpoint_bool_balance": hin["endpoint_bool"].value_counts().to_dict(),
        "synthetic_fraction": round(float(hin["synthetic"].mean()), 3),
        "duration_mean_s": round(float(hin["duration_s"].mean()), 3),
    }

    with open(out_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))

    # --- visual: raw waveform -> log-mel spectrogram for a few real clips ---
    examples = []
    true_row = df[df["endpoint_bool"]].iloc[0]
    false_row = df[~df["endpoint_bool"]].iloc[0]
    examples.append(("endpoint_true", true_row))
    examples.append(("endpoint_false", false_row))
    if len(filler_df[filler_df["endfiller"]]) > 0:
        examples.append(("endfiller_true", filler_df[filler_df["endfiller"]].iloc[0]))

    for name, row in examples:
        wav_path = data_dir / row["path"]
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=400, hop_length=160)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        fig, axes = plt.subplots(2, 1, figsize=(9, 6))
        t = np.arange(len(y)) / sr
        axes[0].plot(t, y, linewidth=0.5)
        axes[0].set_title(f"Waveform: {name} (lang={row['language']}, dur={row['duration_s']:.2f}s)")
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("amplitude")

        img = librosa.display.specshow(
            log_mel, sr=sr, hop_length=160, x_axis="time", y_axis="mel", ax=axes[1]
        )
        axes[1].set_title("Log-mel spectrogram (what the encoder actually sees)")
        fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
        fig.tight_layout()
        fig.savefig(out_dir / f"waveform_spectrogram_{name}.png", dpi=120)
        plt.close(fig)

    print(f"\nWrote plots + eda_summary.json to {out_dir}/")


if __name__ == "__main__":
    main()
