"""Aggregate revision-2 genome-wide benchmark results into summary tables.

Reads the per-cell CSVs benchmark.py writes to analysis/:
    1KGP_genomewide_phased_{biunet|beagle}_BAT_rand{RS}_1KGP_chr{C}_ALL_seg1024_overlap128_missing{M}.csv
(columns: MAF_bin, Num_SNPs, Bin_Acc, Bin_R2, Bin_Precision, Bin_Recall, Bin_F1,
plus constant "Overall *" columns) and writes NEW files under Results/ with a
 suffix — never overwrites manuscript originals.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_cells

DATASET = sys.argv[1] if len(sys.argv) > 1 else "1KGP"
POP = "ALL"
CHRS = list(range(1, 23))  # full autosome set
# label -> (runId prefix of the analysis CSV, benchmark.py method_tag, seg tag).
# SCDA runs through test.py, so benchmark.py tags it .biunet.; runs are told
# apart by runId prefix and seg tag.
METHODS_BY_DATASET = {
    "1KGP": {
        "biunet": ("1KGP_genomewide", "biunet", "seg1024_overlap128"),
        "beagle": ("1KGP_genomewide", "beagle", "seg1024_overlap128"),
        "scda":   ("1KGP_scda_genomewide", "biunet", "seg-1_overlap0"),
    },
    # SGDP is read from the standalone runs. Beagle.s cells carry no epoch.
    "SGDP": {
        "biunet": ("SGDP_genomewide", "biunet", "seg1024_overlap128"),
        "beagle": ("SGDP_ablation_aligned", "beagle", "seg1024_overlap128"),
        "scda":   ("SGDP_scda_genomewide", "biunet", "seg-1_overlap0"),
    },
}
METHODS = METHODS_BY_DATASET[DATASET]
# Pin the checkpoint each method is reported at. Without a pin the resolver takes
# whatever epoch scored highest-numbered per cell, which silently mixes epochs
# while an evaluation is still filling in -- a table half at epoch 24 and half at
# epoch 70 looks finished and is not. A pinned epoch that has no cell yet is
# reported as missing instead of being substituted.
EPOCHS_BY_DATASET = {
    "1KGP": {"biunet": 70, "scda": 286},
    # both SGDP runs held their val split out, so each reports its val-selected
    # checkpoint; Beagle has no epoch and resolves through the legacy name.
    "SGDP": {"biunet": "best", "scda": "best"},
}
EPOCHS = EPOCHS_BY_DATASET.get(DATASET, {})
RANDS = [0, 42, 1024]
MISSING = ["5%", "15%", "25%"]
BIN_METRICS = ["Bin_Acc", "Bin_R2", "Bin_Precision", "Bin_Recall", "Bin_F1"]
OVERALL_METRICS = ["Overall Acc", "Overall R2", "Overall Precision",
                   "Overall Recall", "Overall F1"]

ANALYSIS = "analysis"
OUTDIR = "Results"


def cell_path(method, rs, chrom, miss):
    run_id, tag, seg = METHODS[method]
    return analysis_cells.resolve(
        ANALYSIS, run_id, "phased", tag, "BAT", rs, DATASET, chrom, POP, seg,
        miss, epoch=EPOCHS.get(method))


def load_cells():
    rows, missing_files, used = [], [], []
    for method in METHODS:
        for rs in RANDS:
            for chrom in CHRS:
                for miss in MISSING:
                    p = cell_path(method, rs, chrom, miss)
                    if p is None:
                        missing_files.append(
                            f"{METHODS[method][0]} {DATASET} chr{chrom} "
                            f"rand{rs} missing{miss}")
                        continue
                    used.append(p)
                    df = pd.read_csv(p)
                    df["method"], df["rand"], df["chr"], df["missing"] = \
                        method, rs, chrom, miss
                    rows.append(df)
    print(f"Cell provenance: {analysis_cells.describe(used)}")
    if missing_files:
        print(f"WARNING: {len(missing_files)} expected result cells absent "
              f"(listed below); tables cover only what exists.")
        for p in missing_files:
            print(f"  missing: {p}")
    if not rows:
        sys.exit("No result files found — nothing to aggregate.")
    return pd.concat(rows, ignore_index=True)


def per_chromosome_tables(cells):
    """Mean +/- sd across rand states of each metric, per (method, missing, chr, bin)."""
    out = []
    grp = cells.groupby(["method", "missing", "chr", "MAF_bin"], sort=False)
    agg = grp.agg(
        Num_SNPs=("Num_SNPs", "mean"),
        **{m: (m, "mean") for m in BIN_METRICS},
        **{f"{m}_sd": (m, "std") for m in BIN_METRICS},
        **{m: (m, "mean") for m in OVERALL_METRICS},
    ).reset_index()
    return agg


def genomewide_table(cells):
    """Num_SNPs-weighted mean across chromosomes within each rand state,
    then mean +/- sd across rand states."""
    recs = []
    for (method, miss, rs, maf), g in cells.groupby(
            ["method", "missing", "rand", "MAF_bin"], sort=False):
        w = g["Num_SNPs"].to_numpy(dtype=float)
        rec = {"method": method, "missing": miss, "rand": rs, "MAF_bin": maf,
               "Num_SNPs": w.sum()}
        for m in BIN_METRICS:
            v = g[m].to_numpy(dtype=float)
            ok = np.isfinite(v)
            rec[m] = np.average(v[ok], weights=w[ok]) if ok.any() else np.nan
        recs.append(rec)
    per_rand = pd.DataFrame(recs)
    grp = per_rand.groupby(["method", "missing", "MAF_bin"], sort=False)
    agg = grp.agg(
        Num_SNPs=("Num_SNPs", "mean"),
        **{m: (m, "mean") for m in BIN_METRICS},
        **{f"{m}_sd": (m, "std") for m in BIN_METRICS},
    ).reset_index()
    return agg


def head_to_head(gw):
    """BiUNet minus Beagle on Bin_R2 / Bin_Acc per (missing, MAF_bin)."""
    piv = gw.pivot_table(index=["missing", "MAF_bin"], columns="method",
                         values=["Bin_R2", "Bin_Acc"], sort=False)
    out = pd.DataFrame(index=piv.index)
    for m in ["Bin_R2", "Bin_Acc"]:
        if ("biunet" in piv[m]) and ("beagle" in piv[m]):
            out[f"{m}_biunet"] = piv[m]["biunet"]
            out[f"{m}_beagle"] = piv[m]["beagle"]
            out[f"{m}_delta"] = piv[m]["biunet"] - piv[m]["beagle"]
    return out.reset_index()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cells = load_cells()

    per_chr = per_chromosome_tables(cells)
    p1 = os.path.join(OUTDIR, f"genomewide_{DATASET}_per_chromosome.csv")
    per_chr.to_csv(p1, index=False)
    print(f"wrote {p1}  ({len(per_chr)} rows)")

    gw = genomewide_table(cells)
    p2 = os.path.join(OUTDIR, f"genomewide_{DATASET}_summary.csv")
    gw.to_csv(p2, index=False)
    print(f"wrote {p2}  ({len(gw)} rows)")

    h2h = head_to_head(gw)
    p3 = os.path.join(OUTDIR, f"genomewide_{DATASET}_biunet_minus_beagle.csv")
    h2h.to_csv(p3, index=False)
    print(f"wrote {p3}  ({len(h2h)} rows)")

    print("\n=== genome-wide head-to-head (BiUNet - Beagle) ===")
    print(h2h.to_string(index=False))


if __name__ == "__main__":
    main()
