"""
Final evaluation of a trained checkpoint on the held-out TEST split, plus a
latency benchmark.

Test metrics: accuracy / precision / recall / F1 + confusion matrix for the
endpoint task, per-language accuracy (Hindi called out), and endfiller-head
accuracy on annotated rows.

Latency is measured the way it matters in production: batch=1 (one clip, the
real-time decision), on GPU and CPU separately, with feature extraction
(log-mel) and model forward timed independently. Reports mean / median / p95.

Usage:
    uv run python src/evaluate.py --ckpt checkpoints/finetune/best.pt
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import TurnDataset, collate  # noqa: E402
from model import TurnDetector  # noqa: E402


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TurnDetector(
        max_seconds=cfg.get("max_seconds", 8.0),
        pooling=cfg.get("pooling", "attention"),
        use_endfiller_head=not cfg.get("no_endfiller_head", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


def test_metrics(model, loader, device):
    ep_logits, ep_labels, langs = [], [], []
    ef_logits, ef_labels, ef_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["input_features"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(feats)
            ep_logits.append(out["endpoint_logit"].float().cpu())
            ep_labels.append(batch["endpoint"])
            langs.extend(batch["language"])
            if "endfiller_logit" in out:
                ef_logits.append(out["endfiller_logit"].float().cpu())
                ef_labels.append(batch["endfiller"])
                ef_masks.append(batch["endfiller_mask"])

    logits = torch.cat(ep_logits).numpy()
    labels = torch.cat(ep_labels).numpy().astype(int)
    preds = (logits > 0).astype(int)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()  # [[TN, FP], [FN, TP]]

    lang_arr = np.array(langs)
    per_lang = {}
    for lang in sorted(set(langs)):
        m = lang_arr == lang
        per_lang[lang] = {"n": int(m.sum()), "acc": round(float(accuracy_score(labels[m], preds[m])), 4)}

    result = {
        "n_test": len(labels),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": {"TN": cm[0][0], "FP": cm[0][1], "FN": cm[1][0], "TP": cm[1][1]},
        "per_language": per_lang,
    }

    if ef_logits:
        efl = torch.cat(ef_logits).numpy()
        efy = torch.cat(ef_labels).numpy().astype(int)
        efm = torch.cat(ef_masks).numpy().astype(bool)
        if efm.sum() > 0:
            efp = (efl[efm] > 0).astype(int)
            result["endfiller_head_acc_on_annotated"] = round(float(accuracy_score(efy[efm], efp)), 4)
            result["endfiller_annotated_n"] = int(efm.sum())
    return result


def benchmark_latency(model, dataset, device, n=200, warmup=20):
    """Batch=1 single-clip latency: feature extraction vs model forward."""
    idxs = np.random.default_rng(0).integers(0, len(dataset), size=n + warmup)

    # Pre-load raw items so we time extraction + forward, not disk.
    import soundfile as sf
    raws = []
    for i in idxs:
        row = dataset.df.iloc[int(i)]
        y, sr = sf.read(dataset.data_dir / row["path"])
        if y.ndim > 1:
            y = y.mean(axis=1)
        raws.append(y.astype(np.float32))

    fe = dataset.fe
    max_samples = dataset.max_samples

    def extract(y):
        if len(y) >= max_samples:
            y = y[-max_samples:]
        else:
            y = np.concatenate([np.zeros(max_samples - len(y), dtype=y.dtype), y])
        return fe(y, sampling_rate=16000, return_tensors="pt", padding="max_length",
                  max_length=max_samples, truncation=True).input_features

    fe_times, fwd_times = [], []
    model.eval()
    with torch.no_grad():
        for j, y in enumerate(raws):
            t0 = time.perf_counter()
            feats = extract(y)
            t1 = time.perf_counter()
            feats = feats.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _ = model(feats)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            if j >= warmup:
                fe_times.append((t1 - t0) * 1000)
                fwd_times.append((t3 - t2) * 1000)

    def stats(a):
        a = np.array(a)
        return {"mean_ms": round(float(a.mean()), 2),
                "median_ms": round(float(np.median(a)), 2),
                "p95_ms": round(float(np.percentile(a, 95)), 2)}

    return {
        "device": device.type,
        "n_measured": len(fwd_times),
        "feature_extraction": stats(fe_times),
        "model_forward": stats(fwd_times),
        "total_end_to_end": stats(np.array(fe_times) + np.array(fwd_times)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/finetune/best.pt")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--out", default="reports/eval/test_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, cfg = load_model(args.ckpt, device)
    max_seconds = cfg.get("max_seconds", 8.0)

    test_ds = TurnDataset(args.data_dir, "test", max_seconds)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=args.num_workers, pin_memory=True)

    print("Evaluating on test split...")
    metrics = test_metrics(model, test_loader, device)

    print("Benchmarking latency (GPU)...")
    lat_gpu = benchmark_latency(model, test_ds, device) if device.type == "cuda" else None

    print("Benchmarking latency (CPU)...")
    model_cpu = model.to(torch.device("cpu")).float()
    lat_cpu = benchmark_latency(model_cpu, test_ds, torch.device("cpu"))

    n_params = sum(p.numel() for p in model.parameters())
    ckpt_mb = round(Path(args.ckpt).stat().st_size / 1e6, 1)

    report = {
        "checkpoint": str(args.ckpt),
        "model": {"total_params": n_params, "checkpoint_size_mb": ckpt_mb,
                  "max_seconds": max_seconds, "pooling": cfg.get("pooling")},
        "test_metrics": metrics,
        "latency": {"gpu": lat_gpu, "cpu": lat_cpu},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
