"""Table for the contribution of the merged reference panel.

Reads the benchmark cells of the genome-wide SGDP models trained with and
without the panel and writes Results/rev2_merge_effect_revision2.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_cells

ANALYSIS = os.environ.get("BIUNET_ANALYSIS", "analysis")
OUTDIR = "Results"
RANDS, MISSING = [0, 42, 1024], ["5%", "15%", "25%"]
SEG = "seg1024_overlap128"
CHRS = list(range(1, 23))
BIN_ORDER = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%",
             "30%~40%", "40%~50%"]

# label -> run identity in analysis_cells
MODELS = {
    "SGDP training split only": "SGDP_ablation_noreference",
    "reference merged on the SGDP axis": "SGDP_genomewide",
}


def cells(run_id, chrom, miss, rand):
    p = analysis_cells.resolve(ANALYSIS, run_id, "phased", "biunet", "BAT",
                               rand, "SGDP", chrom, "ALL", SEG, miss)
    if p is None:
        return None
    df = pd.read_csv(p)
    df["MAF_bin"] = df["MAF_bin"].astype(str).str.strip()
    return df


rows, missing_cells = [], []
for label, run_id in MODELS.items():
    for miss in MISSING:
        per_seed = {b: [] for b in BIN_ORDER + ["Overall"]}
        counts = {b: [] for b in BIN_ORDER + ["Overall"]}
        for rand in RANDS:
            frames = []
            for c in CHRS:
                df = cells(run_id, c, miss, rand)
                if df is None:
                    missing_cells.append(f"{run_id} chr{c} {miss} rand{rand}")
                    continue
                frames.append(df)
            if not frames:
                continue
            for b in BIN_ORDER:
                sub = pd.concat([f[f.MAF_bin == b] for f in frames])
                w = sub["Num_SNPs"].to_numpy(dtype=float)
                per_seed[b].append(np.average(sub["Bin_R2"], weights=w))
                counts[b].append(w.sum())
            # Overall is one value per chromosome, weighted by that
            # chromosome's marker count rather than repeated per bin.
            ov = [(f["Overall R2"].iloc[0], f[f.MAF_bin != "Overall"]["Num_SNPs"].sum())
                  for f in frames]
            vals = np.array([v for v, _ in ov], dtype=float)
            wts = np.array([n for _, n in ov], dtype=float)
            per_seed["Overall"].append(np.average(vals, weights=wts))
            counts["Overall"].append(wts.sum())
        for b in BIN_ORDER + ["Overall"]:
            if not per_seed[b]:
                continue
            rows.append({"model": label, "missing": miss, "MAF_bin": b,
                         "R2": float(np.mean(per_seed[b])),
                         "R2_sd": float(np.std(per_seed[b], ddof=1))
                                  if len(per_seed[b]) > 1 else 0.0,
                         "Num_SNPs": int(round(np.mean(counts[b])))})

out = pd.DataFrame(rows)
os.makedirs(OUTDIR, exist_ok=True)
path = os.path.join(OUTDIR, "rev2_merge_effect_revision2.csv")
out.to_csv(path, index=False)
print(f"wrote {path} ({len(out)} rows)")
if missing_cells:
    print(f"NOTE: {len(missing_cells)} cells absent, e.g. {missing_cells[:3]}")
