"""
Stream pipecat-ai/smart-turn-data-v3.2-train and curate a working subset.

The full dataset is 270,946 clips / 41.4GB across 23 languages. For fast
iteration we don't train on all of it up front: this script streams the
train split, applies per-language quotas (Hindi weighted heavily since it's
our target domain and the smallest well-represented language pool), writes
each kept clip to disk as 16kHz mono WAV, and produces a stratified
train/val/test metadata split.

Note: the dataset has no explicit "Hinglish" (code-switched) label, only
plain `language == "hin"`. This script curates the Hindi pool as the closest
available proxy; a separate small hand-checked Hinglish eval set is a
follow-up (see README "Status").

Usage:
    uv run python src/data_prep.py --out-dir data/processed --max-scan 150000
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"
TARGET_SR = 16_000  # Whisper's expected input sample rate

# Per-language sample caps. Hindi is weighted far above its natural
# frequency since it's the closest proxy we have for the Hinglish target
# domain. Everything else is a thin general-multilingual baseline slice,
# not a training priority.
LANGUAGE_QUOTAS = {
    "hin": 12_000,  # take essentially all available Hindi
    "eng": 6_000,
    "spa": 1_500,
    "rus": 1_500,
    "por": 1_500,
    "fra": 1_500,
    "deu": 1_500,
    "nld": 1_000,
    "pol": 1_000,
    "kor": 1_000,
    "ind": 1_000,
    "nor": 1_000,
    "vie": 1_000,
    "fin": 1_000,
    "ben": 1_000,
    "tur": 1_000,
    "ita": 800,
    "jpn": 800,
    "ukr": 800,
    "ara": 800,
    "dan": 800,
    "mar": 800,
    "zho": 800,
}


def safe_stratified_split(df: pd.DataFrame, strat_key: pd.Series, test_size: float, seed: int):
    """train_test_split, but strata with <2 members fall back to an
    unstratified split instead of crashing (small language pools + a
    two-level split make this common here)."""
    vc = strat_key.value_counts()
    stratifiable_idx = strat_key.isin(vc[vc >= 2].index)
    strat_part = df[stratifiable_idx]
    rest_part = df[~stratifiable_idx]

    a, b = train_test_split(
        strat_part,
        test_size=test_size,
        random_state=seed,
        stratify=strat_key.loc[strat_part.index],
    )
    if len(rest_part) == 1:
        a = pd.concat([a, rest_part], ignore_index=True)
    elif len(rest_part) > 1:
        rest_a, rest_b = train_test_split(rest_part, test_size=test_size, random_state=seed)
        a = pd.concat([a, rest_a], ignore_index=True)
        b = pd.concat([b, rest_b], ignore_index=True)
    return a, b


def resample_if_needed(audio_array: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return audio_array.astype(np.float32)
    import librosa

    return librosa.resample(
        audio_array.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument(
        "--max-scan",
        type=int,
        default=150_000,
        help="Safety cap on rows scanned from the stream, so a run always terminates "
        "even if quotas can't all be filled (dataset isn't shuffled by language).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {DATASET_NAME} (train split)...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    counts: Counter = Counter()
    endpoint_counts: dict = {}  # per-language True/False balance tracking
    rows = []
    scanned = 0

    quotas_remaining = dict(LANGUAGE_QUOTAS)

    pbar = tqdm(total=args.max_scan, desc="scanning")
    for row in ds:
        scanned += 1
        pbar.update(1)
        lang = row["language"]

        if lang not in quotas_remaining or quotas_remaining[lang] <= 0:
            if scanned >= args.max_scan or not any(
                v > 0 for v in quotas_remaining.values()
            ):
                break
            continue

        quotas_remaining[lang] -= 1
        counts[lang] += 1

        audio = row["audio"]
        arr = resample_if_needed(np.asarray(audio["array"]), audio["sampling_rate"])
        clip_id = row["id"]
        wav_path = audio_dir / f"{clip_id}.wav"
        sf.write(wav_path, arr, TARGET_SR)

        rows.append(
            {
                "id": clip_id,
                "path": wav_path.relative_to(out_dir).as_posix(),
                "language": lang,
                "endpoint_bool": row["endpoint_bool"],
                "midfiller": row["midfiller"],
                "endfiller": row["endfiller"],
                "synthetic": row["synthetic"],
                "source_dataset": row["dataset"],
                "duration_s": len(arr) / TARGET_SR,
            }
        )

        if scanned >= args.max_scan:
            break

    pbar.close()
    print(f"Scanned {scanned} rows, kept {len(rows)} clips.")
    print("Per-language kept counts:", dict(counts))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows collected — check quotas/streaming connectivity.")

    df["endpoint_bool"] = df["endpoint_bool"].astype(bool)
    strat_key = df["language"] + "_" + df["endpoint_bool"].astype(str)

    train_df, temp_df = safe_stratified_split(df, strat_key, test_size=0.2, seed=args.seed)
    temp_strat_key = temp_df["language"] + "_" + temp_df["endpoint_bool"].astype(str)
    val_df, test_df = safe_stratified_split(temp_df, temp_strat_key, test_size=0.5, seed=args.seed)

    train_df.to_csv(out_dir / "metadata_train.csv", index=False)
    val_df.to_csv(out_dir / "metadata_val.csv", index=False)
    test_df.to_csv(out_dir / "metadata_test.csv", index=False)

    summary = {
        "scanned_rows": scanned,
        "kept_clips": len(rows),
        "per_language_kept": dict(counts),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "endpoint_bool_balance_kept": df["endpoint_bool"]
        .value_counts()
        .to_dict(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote splits + audio to {out_dir}/")


if __name__ == "__main__":
    main()
