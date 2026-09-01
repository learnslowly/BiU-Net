"""Turn a training log into a durable learning-curve table.

Usage: python scripts/extract_training_curve.py <log> [<log> ...]
Writes Results/training_curve_{runId}.csv per log.
"""
import os
import re
import sys

import pandas as pd

try:
    import torch
except ImportError:
    torch = None

OUTDIR = "Results"
LINE = re.compile(
    r"Epoch (\d+)/(\d+) - Dataset (\d+) (Val )?Loss: ([\d.]+), Acc: ([\d.]+)")
RUNID = re.compile(r"runId\s*[:=]\s*(\S+)")
CURR = re.compile(r"curriculumWarmupEpochs\s*[:=]\s*(\d+)")


def parse(path):
    """{(epoch, bucket, is_val): (loss, acc)} keeping the last value seen."""
    seen, run_id, curriculum = {}, None, 0
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            for line in raw.replace("\r", "\n").split("\n"):
                m = LINE.search(line)
                if m:
                    ep, _, ds, is_val, loss, acc = m.groups()
                    seen[(int(ep), int(ds), bool(is_val))] = (float(loss), float(acc))
                    continue
                if run_id is None:
                    r = RUNID.search(line)
                    if r:
                        run_id = r.group(1)
                c = CURR.search(line)
                if c:
                    curriculum = int(c.group(1))
    return seen, run_id, curriculum


for log_path in sys.argv[1:]:
    seen, run_id, curriculum = parse(log_path)
    if not seen:
        print(f"{log_path}: no epoch lines found, skipping")
        continue
    run_id = run_id or os.path.basename(log_path).split("_")[0]

    rows = {}
    for (ep, ds, is_val), (loss, acc) in seen.items():
        r = rows.setdefault(ep, {"train": [], "val": []})
        r["val" if is_val else "train"].append((loss, acc))
    table = []
    for ep in sorted(rows):
        tr, va = rows[ep]["train"], rows[ep]["val"]
        table.append({
            "epoch": ep,
            "n_buckets": len(tr) or len(va),
            "train_loss": sum(l for l, _ in tr) / len(tr) if tr else None,
            "train_acc": sum(a for _, a in tr) / len(tr) if tr else None,
            "val_loss": sum(l for l, _ in va) / len(va) if va else None,
            "val_acc": sum(a for _, a in va) / len(va) if va else None,
        })
    df = pd.DataFrame(table)

    val = df[df.val_loss.notna() & (df.epoch > curriculum)]
    summary = ""
    if not val.empty:
        best = val.loc[val.val_loss.idxmin()]
        # saturation: first validated epoch within 1% of the best loss reached
        near = val[val.val_loss <= best.val_loss * 1.01]
        sat = int(near.epoch.min())
        df["is_best"] = df.epoch == int(best.epoch)
        summary = (f"best epoch {int(best.epoch)} "
                   f"(val loss {best.val_loss:.4f}, acc {best.val_acc:.4f}); "
                   f"within 1% of best from epoch {sat}; "
                   f"{len(val)} validated epochs up to {int(val.epoch.max())}")
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"training_curve_{run_id}.csv")
    df.to_csv(out, index=False)
    print(f"{run_id}: {out}")
    if summary:
        print(f"  {summary}")

    # independent check: the epoch stored inside the run's own best checkpoint
    if torch is not None:
        hits = [p for p in os.listdir("checkpoints")
                if p.startswith(f"checkpoint_{run_id}_") and p.endswith("_epoch_best.pth")] \
            if os.path.isdir("checkpoints") else []
        for h in hits[:1]:
            try:
                ck = torch.load(os.path.join("checkpoints", h), map_location="cpu",
                                weights_only=False)
                print(f"  checkpoint {h} reports epoch {ck.get('epoch')} "
                      f"(best_val_loss {ck.get('best_val_loss')})")
            except Exception as e:
                print(f"  could not read {h}: {e}")
