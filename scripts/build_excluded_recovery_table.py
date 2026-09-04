"""Accuracy of the fine-tuned genome-wide model where a reference-based imputer
has nothing to say.

Usage: python scripts/build_excluded_recovery_table.py
Writes Results/excluded_markers_recovery.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_cells

ANALYSIS, OUTDIR = "analysis", "Results"
DATASET, POPULATION = "SGDP", "ALL"
RUN_ID, EPOCH = "SGDP_finetuned", "best"
SEG = "seg1024_overlap128"
RANDS, MISSING, CHRS = [0, 42, 1024], ["5%", "15%", "25%"], list(range(1, 23))
BIN_ORDER = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%",
             "30%~40%", "40%~50%"]


def cells():
    """Every excluded-marker cell, one row per chromosome, rate and bin."""
    rows, used, absent = [], [], 0
    for chrom in CHRS:
        for miss in MISSING:
            for rand in RANDS:
                path = analysis_cells.resolve(
                    ANALYSIS, RUN_ID, "phased", "biunet", "BAT_EX", rand,
                    DATASET, chrom, POPULATION, SEG, miss, epoch=EPOCH)
                if path is None:
                    absent += 1
                    continue
                used.append(path)
                df = pd.read_csv(path)
                problem = analysis_cells.check(path, df)
                if problem:
                    print(f"WARNING: {problem}")
                df["MAF_bin"] = df["MAF_bin"].astype(str).str.strip()
                for _, r in df.iterrows():
                    rows.append({"chr": chrom, "missing": miss, "rand": rand,
                                 "MAF_bin": r["MAF_bin"], "R2": r["Bin_R2"],
                                 "n": r["Num_SNPs"]})
    if absent:
        print(f"WARNING: {absent} cells absent; the table covers what was written")
    print("Cell provenance:", analysis_cells.describe(used))
    return pd.DataFrame(rows)


def weighted(group):
    w = group["n"].to_numpy(float)
    return np.average(group["R2"], weights=w) if w.sum() else np.nan


def main():
    d = cells()
    if d.empty:
        raise SystemExit("no excluded-marker cells found under analysis/")
    out = []
    for miss in MISSING:
        sub = d[d.missing == miss]
        row = {"missing": miss}
        for b in BIN_ORDER + ["Overall"]:
            s = sub[sub.MAF_bin == b]
            row[b] = weighted(s) if len(s) else np.nan
        # one seed's marker count per chromosome: the three seeds mask the same
        # markers, so summing all three would triple the total
        first_seed = sub[(sub.MAF_bin == "Overall") & (sub.rand == RANDS[0])]
        row["markers"] = int(first_seed.groupby("chr")["n"].first().sum())
        out.append(row)
    table = pd.DataFrame(out).set_index("missing")
    print(table.round(4).to_string())
    path = os.path.join(OUTDIR, "excluded_markers_recovery.csv")
    table.to_csv(path)
    print(f"wrote {path} ({len(table)} rows)")


if __name__ == "__main__":
    main()
