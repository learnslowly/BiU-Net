"""Per-chromosome fine-tuning: GW-M against GW-M+FT across all 22 autosomes.

Usage: python scripts/build_rev2_chrft.py
Writes Results/rev2_chrft_per_chromosome_revision2.csv
       Results/rev2_chrft_summary_revision2.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_cells

ANALYSIS, OUTDIR = "analysis", "Results"
RANDS, MISSING, CHRS = [0, 42, 1024], ["5%", "15%", "25%"], list(range(1, 23))
SEG = "seg1024_overlap128"
BIN_ORDER = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%",
             "30%~40%", "40%~50%"]

# dataset -> {arm: (runId, reporting epoch)}
ARMS = {
    "1KGP": {"GW-M": ("v3low_ref_g22", 70), "GW-M+FT": ("v3low_ftchr", 30)},
    "SGDP": {"GW-M": ("ft_sgdp_full", "best"), "GW-M+FT": ("ft_sgdp_ftchr", "best")},
}


def cells(run_id, epoch, dataset, chrom, miss):
    """Seed cells for one chromosome, with degenerate ones rejected whole.

    "Overall R2" is a per-cell scalar repeated on every row of that cell, so
    dropping only the offending rows leaves the bad overall value behind on the
    rows that survive. A cell showing the constant-prediction signature anywhere
    is therefore discarded entirely rather than repaired.
    """
    frames = []
    for rs in RANDS:
        p = analysis_cells.resolve(ANALYSIS, run_id, "phased", "biunet", "BAT",
                                   rs, dataset, chrom, "ALL", SEG, miss,
                                   epoch=epoch)
        if p is None:
            continue
        one = pd.read_csv(p)
        one["MAF_bin"] = one["MAF_bin"].astype(str).str.strip()
        bad = analysis_cells.degenerate_rows(one)
        if len(bad):
            DEGENERATE.append(f"{dataset} {run_id} chr{chrom} {miss} rand{rs}: "
                              + ", ".join(sorted(set(bad["MAF_bin"]))))
            continue
        frames.append(one)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


DEGENERATE = []
rows, missing_cells = [], []
for ds, arms in ARMS.items():
    for arm, (run_id, epoch) in arms.items():
        for c in CHRS:
            for miss in MISSING:
                df = cells(run_id, epoch, ds, c, miss)
                if df is None:
                    missing_cells.append(f"{ds} {arm} chr{c} {miss}")
                    continue
                ov = df[df.MAF_bin == "Overall"]
                rows.append({"dataset": ds, "arm": arm, "chr": c,
                             "missing": miss,
                             "R2": df["Overall R2"].mean(),
                             "Num_SNPs": ov["Num_SNPs"].mean()})
                for b in BIN_ORDER:
                    sub = df[df.MAF_bin == b]
                    if sub.empty:
                        continue
                    rows.append({"dataset": ds, "arm": arm, "chr": c,
                                 "missing": miss, "MAF_bin": b,
                                 "R2": sub["Bin_R2"].mean(),
                                 "Num_SNPs": sub["Num_SNPs"].mean()})

if not rows:
    sys.exit("No benchmark cells found for either arm.")
long = pd.DataFrame(rows)
long["MAF_bin"] = long.get("MAF_bin", pd.Series(dtype=object)).fillna("Overall")

os.makedirs(OUTDIR, exist_ok=True)
p1 = os.path.join(OUTDIR, "rev2_chrft_per_chromosome_revision2.csv")
wide = (long[long.MAF_bin == "Overall"]
        .pivot_table(index=["dataset", "chr", "missing"], columns="arm",
                     values=["R2", "Num_SNPs"]))
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
if "R2_GW-M" in wide and "R2_GW-M+FT" in wide:
    wide["delta"] = wide["R2_GW-M+FT"] - wide["R2_GW-M"]
wide.reset_index().to_csv(p1, index=False)
print(f"wrote {p1} ({len(wide)} rows)")


def weighted(g):
    w = g["Num_SNPs"].to_numpy(dtype=float)
    return np.average(g["R2"], weights=w) if w.sum() else np.nan


summary = (long.groupby(["dataset", "arm", "missing", "MAF_bin"])
           .apply(weighted, include_groups=False).rename("R2").reset_index())
counts = (long.groupby(["dataset", "arm", "missing", "MAF_bin"])
          .agg(chromosomes=("chr", "nunique"),
               Num_SNPs=("Num_SNPs", "sum")).reset_index())
summary = summary.merge(counts, on=["dataset", "arm", "missing", "MAF_bin"])
p2 = os.path.join(OUTDIR, "rev2_chrft_summary_revision2.csv")
summary.to_csv(p2, index=False)
print(f"wrote {p2} ({len(summary)} rows)")

if DEGENERATE:
    print(f"NOTE: {len(DEGENERATE)} cell(s) dropped as degenerate "
          f"(constant prediction: high accuracy with zero R2):")
    for line in DEGENERATE:
        print(f"  {line}")
if missing_cells:
    print(f"NOTE: {len(missing_cells)} cell groups absent, first few: "
          + "; ".join(missing_cells[:6]))
for ds in ARMS:
    sub = summary[(summary.dataset == ds) & (summary.MAF_bin == "Overall")]
    for miss in MISSING:
        s = sub[sub.missing == miss].set_index("arm")
        if {"GW-M", "GW-M+FT"} <= set(s.index):
            d = s.loc["GW-M+FT", "R2"] - s.loc["GW-M", "R2"]
            print(f"  {ds} {miss}: GW-M {s.loc['GW-M','R2']:.4f} -> "
                  f"GW-M+FT {s.loc['GW-M+FT','R2']:.4f} ({d:+.4f}), "
                  f"{int(s.loc['GW-M+FT','chromosomes'])} chromosomes")
