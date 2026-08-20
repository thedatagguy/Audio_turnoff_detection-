"""
Train the turn-detection model.

Loss = BCE(endpoint) + aux_weight * masked_BCE(endfiller). The endfiller term
is masked to the ~85% of rows that carry the annotation.

Fine-tunes the whole encoder by default (cheap at this scale); pass
--freeze-encoder for the fast head-only baseline. Uses discriminative
learning rates (lower for the pretrained encoder, higher for the fresh
heads) and bfloat16 autocast on GPU.

Metrics each epoch: accuracy / precision / recall / F1 for the endpoint task,
plus per-language endpoint accuracy (Hindi called out, since it's the target
domain). Best checkpoint by val endpoint-F1 is saved.

Usage:
    uv run python src/train.py --out-dir checkpoints/run1 --epochs 4
    uv run python src/train.py --freeze-encoder --out-dir checkpoints/frozen
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import TurnDataset, collate  # noqa: E402
from model import TurnDetector  # noqa: E402


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels, all_langs = [], [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["input_features"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(feats)
            all_logits.append(out["endpoint_logit"].float().cpu())
            all_labels.append(batch["endpoint"])
            all_langs.extend(batch["language"])
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (logits > 0).astype(int)  # logit>0 <=> prob>0.5

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    per_lang = {}
    lang_arr = np.array(all_langs)
    for lang in sorted(set(all_langs)):
        m = lang_arr == lang
        per_lang[lang] = {
            "n": int(m.sum()),
            "acc": round(float(accuracy_score(labels[m], preds[m])), 4),
        }

    return {
        "acc": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "per_language": per_lang,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seconds", type=float, default=8.0)
    p.add_argument("--pooling", default="attention", choices=["attention", "mean"])
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--no-endfiller-head", action="store_true")
    p.add_argument("--aux-weight", type=float, default=0.3)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-train-batches", type=int, default=0, help="0=all; >0 for smoke tests")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = TurnDataset(args.data_dir, "train", args.max_seconds)
    val_ds = TurnDataset(args.data_dir, "val", args.max_seconds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
    )

    model = TurnDetector(
        max_seconds=args.max_seconds,
        pooling=args.pooling,
        freeze_encoder=args.freeze_encoder,
        use_endfiller_head=not args.no_endfiller_head,
    ).to(device)
    print(f"Params: total={model.num_total_params():,} trainable={model.num_trainable_params():,}")

    # Discriminative LR: low for pretrained encoder, high for fresh heads.
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder.")]
    param_groups = [{"params": head_params, "lr": args.head_lr}]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": args.encoder_lr})
    optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    bce = nn.BCEWithLogitsLoss()
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

    history = []
    best_f1 = -1.0
    config_dump = vars(args) | {"device": str(device)}
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dump, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = defaultdict(float)
        n_batches = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for i, batch in enumerate(pbar):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            feats = batch["input_features"].to(device)
            ep = batch["endpoint"].to(device)
            ef = batch["endfiller"].to(device)
            ef_mask = batch["endfiller_mask"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(feats)
                loss_ep = bce(out["endpoint_logit"], ep)
                loss = loss_ep
                if "endfiller_logit" in out and ef_mask.sum() > 0:
                    per = bce_none(out["endfiller_logit"], ef)
                    loss_ef = (per * ef_mask).sum() / ef_mask.sum()
                    loss = loss + args.aux_weight * loss_ef
                else:
                    loss_ef = torch.tensor(0.0)

            optim.zero_grad()
            loss.backward()
            optim.step()

            running["loss"] += loss.item()
            running["loss_ep"] += loss_ep.item()
            running["loss_ef"] += loss_ef.detach().item()
            n_batches += 1
            pbar.set_postfix(loss=running["loss"] / n_batches)

        train_time = time.time() - t0
        val_metrics = evaluate(model, val_loader, device)
        epoch_log = {
            "epoch": epoch,
            "train_loss": round(running["loss"] / max(n_batches, 1), 4),
            "train_loss_endpoint": round(running["loss_ep"] / max(n_batches, 1), 4),
            "train_loss_endfiller": round(running["loss_ef"] / max(n_batches, 1), 4),
            "train_time_s": round(train_time, 1),
            "val": val_metrics,
        }
        history.append(epoch_log)
        hin = val_metrics["per_language"].get("hin", {})
        print(f"[epoch {epoch}] train_loss={epoch_log['train_loss']} "
              f"val_acc={val_metrics['acc']} val_f1={val_metrics['f1']} "
              f"hin_acc={hin.get('acc')}")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {"model_state": model.state_dict(), "config": config_dump,
                 "epoch": epoch, "val_metrics": val_metrics},
                out_dir / "best.pt",
            )
            print(f"  saved best (f1={best_f1})")

    print(f"Done. Best val F1: {best_f1}")


if __name__ == "__main__":
    main()
