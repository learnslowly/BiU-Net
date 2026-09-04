"""Main-text accuracy table for the genome-wide revision.

Product (written with a _revision2 suffix, originals untouched):
  Results/rev2_{DS}_main_table_revision2.csv   methods x masking rate, R2 in
      Rare (<=1%) / Low (1-10%) / Common (>10%) / Overall, mean +/- SD over seeds

Usage: python scripts/build_rev2_main_tables.py [1KGP|SGDP]
"""
import os
import sys

import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_cells

DATASET = sys.argv[1] if len(sys.argv) > 1 else "1KGP"
ANALYSIS, OUTDIR = "analysis", "Results"
RANDS, MISSING, CHRS = [0, 42, 1024], ["5%", "15%", "25%"], list(range(1, 23))
SEG = "seg1024_overlap128"

# label -> (runId, benchmark method tag, seg tag, colour kept for the
# figure scripts that read this same mapping)
METHODS_BY_DATASET = {
    "1KGP": {
        "BiU-Net": ("v3low_ref_g22", "biunet", SEG, "#1f77b4"),
        "BiU-Net (chr fine-tuned)": ("v3low_ftchr", "biunet", SEG, "#17becf"),
        "Beagle": ("v3low_ref_g22", "beagle", SEG, "#d62728"),
        "SCDA": ("scda_fl_g22", "biunet", "seg-1_overlap0", "#2ca02c"),
    },
    # SGDP is read from the standalone runs. Beagle.s cells carry no epoch.
    "SGDP": {
        "BiU-Net": ("ft_sgdp_full", "biunet", SEG, "#1f77b4"),
        "BiU-Net (chr fine-tuned)": ("ft_sgdp_ftchr", "biunet", SEG, "#17becf"),
        "Beagle": ("ft_sgdp_aln", "beagle", SEG, "#d62728"),
        "SCDA": ("scda_sgdp_scratch", "biunet", "seg-1_overlap0", "#2ca02c"),
    },
}
METHODS = METHODS_BY_DATASET[DATASET]
# The checkpoint each method is reported at, matching build_rev2_tables.py so a
# figure and a table can never disagree about which model they describe.
EPOCHS_BY_DATASET = {
    "1KGP": {"BiU-Net": 70, "BiU-Net (chr fine-tuned)": 30, "SCDA": 286},
    "SGDP": {"BiU-Net": "best", "BiU-Net (chr fine-tuned)": "best",
             "SCDA": "best"},
}
EPOCHS = EPOCHS_BY_DATASET.get(DATASET, {})
# MAF collapse for the main table: the 7-bin detail stays in the supplement
COLLAPSE = {"0.1%~0.5%": "Rare (<=1%)", "0.5%~1%": "Rare (<=1%)",
            "1%~10%": "Low (1-10%)", "10%~20%": "Common (>10%)",
            "20%~30%": "Common (>10%)", "30%~40%": "Common (>10%)",
            "40%~50%": "Common (>10%)"}
GROUPS = ["Rare (<=1%)", "Low (1-10%)", "Common (>10%)"]


def load():
    rows, used, absent = [], [], 0
    for label, (run_id, tag, seg, _) in METHODS.items():
        for rs in RANDS:
            for c in CHRS:
                for m in MISSING:
                    p = analysis_cells.resolve(ANALYSIS, run_id, "phased", tag,
                                               "BAT", rs, DATASET, c, "ALL",
                                               seg, m, epoch=EPOCHS.get(label))
                    if p is None:
                        absent += 1
                        continue
                    used.append(p)
                    df = pd.read_csv(p)
                    df["method"], df["rand"], df["chr"], df["missing"] = label, rs, c, m
                    rows.append(df)
    if not rows:
        sys.exit(f"No benchmark cells found for {DATASET}.")
    if absent:
        print(f"NOTE: {absent} cells absent; products cover what exists.")
    have = {r["method"].iloc[0] for r in rows}
    expected = len(RANDS) * len(CHRS) * len(MISSING)
    for label in list(METHODS):
        n = sum(len(r) > 0 for r in rows if r["method"].iloc[0] == label)
        if label not in have:
            print(f"NOTE: {label} has no cells and is omitted from this build.")
            del METHODS[label]
        elif n < expected:
            print(f"NOTE: {label} covers {n}/{expected} cells; "
                  f"its rows are pooled over what exists.")
    print(f"Cell provenance: {analysis_cells.describe(used)}")
    cells = pd.concat(rows, ignore_index=True)
    cells["MAF_bin"] = cells["MAF_bin"].astype(str).str.strip()
    return cells


def wmean(values, weights):
    weights = np.asarray(weights, dtype=float)
    return np.average(values, weights=weights) if weights.sum() > 0 else np.nan


def main_table(cells):
    binned = cells[cells.MAF_bin.isin(COLLAPSE)].copy()
    binned["group"] = binned.MAF_bin.map(COLLAPSE)
    # per (method, missing, seed): weighted mean across chromosomes and bins
    per_seed = (binned.groupby(["method", "missing", "rand", "group"])
                .apply(lambda g: wmean(g.Bin_R2, g.Num_SNPs), include_groups=False)
                .rename("R2").reset_index())
    overall = (cells.groupby(["method", "missing", "rand", "chr"])
               .agg(R2=("Overall R2", "first"), N=("Num_SNPs", "sum")).reset_index()
               .groupby(["method", "missing", "rand"])
               .apply(lambda g: wmean(g.R2, g.N), include_groups=False)
               .rename("R2").reset_index())
    overall["group"] = "Overall"
    both = pd.concat([per_seed, overall], ignore_index=True)
    stats = (both.groupby(["method", "missing", "group"])
             .agg(mean=("R2", "mean"), sd=("R2", "std")).reset_index())
    stats["cell"] = stats.apply(
        lambda r: f"{r['mean']:.4f} ± {0.0 if pd.isna(r['sd']) else r['sd']:.4f}", axis=1)
    table = (stats.pivot(index=["missing", "method"], columns="group", values="cell")
             .reindex(GROUPS + ["Overall"], axis=1)
             .reindex(pd.MultiIndex.from_product([MISSING, list(METHODS)],
                                                 names=["missing", "method"])))
    path = os.path.join(OUTDIR, f"rev2_{DATASET}_main_table_revision2.csv")
    with open(path, "w") as f:
        f.write(f"# {DATASET}: phased 4-class R2, all positions (benchmarkAll), "
                f"pooled within chromosome then Num_SNPs-weighted across "
                f"chromosomes; mean +/- SD over masking seeds {RANDS}.\n")
        table.to_csv(f)
    print(f"wrote {path}")
    return table


cells = load()
os.makedirs(OUTDIR, exist_ok=True)
table = main_table(cells)
print(table.to_string())
