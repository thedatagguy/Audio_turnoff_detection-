"""
Plot convergence curves from a training run's curve.json:
- per-step training loss (raw + smoothed)
- intra-epoch validation accuracy / F1 / Hindi accuracy

Usage:
    uv run python src/plot_curves.py --run-dir checkpoints/finetune --out reports/training/convergence.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth(y, k=15):
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="checkpoints/finetune")
    p.add_argument("--out", default="reports/training/convergence.png")
    p.add_argument("--steps-per-epoch", type=int, default=217)
    args = p.parse_args()

    curve = json.load(open(Path(args.run_dir) / "curve.json"))
    steps, losses = zip(*curve["step_losses"])
    steps = np.array(steps)
    losses = np.array(losses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- training loss ---
    ax1.plot(steps, losses, color="lightsteelblue", linewidth=0.8, label="raw (per step)")
    sm = smooth(losses)
    ax1.plot(steps[len(steps) - len(sm):], sm, color="navy", linewidth=2, label="smoothed")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("train loss")
    ax1.set_title("Training loss")
    for e in range(1, 5):
        x = e * args.steps_per_epoch
        if x <= steps.max():
            ax1.axvline(x, color="gray", linestyle=":", linewidth=1)
            ax1.text(x, ax1.get_ylim()[1], f"ep{e}", fontsize=8, va="top", ha="right", color="gray")
    ax1.legend()

    # --- validation curve ---
    if curve["curve_evals"]:
        ev = np.array([(s, a, f, (h if h is not None else np.nan))
                       for s, a, f, h in curve["curve_evals"]], dtype=float)
        ax2.plot(ev[:, 0], ev[:, 1], "-o", markersize=3, label="val accuracy")
        ax2.plot(ev[:, 0], ev[:, 2], "-o", markersize=3, label="val F1")
        ax2.plot(ev[:, 0], ev[:, 3], "-o", markersize=3, label="Hindi accuracy")
        ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
        ax2.set_xlabel("training step")
        ax2.set_ylabel("metric")
        ax2.set_title("Validation metrics during training")
        for e in range(1, 5):
            x = e * args.steps_per_epoch
            if x <= ev[:, 0].max():
                ax2.axvline(x, color="gray", linestyle=":", linewidth=1)
        ax2.legend(loc="lower right")
        ax2.set_ylim(0.45, 1.0)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
