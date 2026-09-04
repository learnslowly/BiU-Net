"""Tables for the two supporting experiments: alignment ablation and
per-chromosome specialisation.

  alignment ablation  three SGDP finetunes that differ only in how the 1KGP
                      reference is merged: not at all, on the reference's own
                      variant axis, or projected onto the target axis.
  specialisation      the reported genome-wide model fine-tuned on a single
                      chromosome for 30 epochs, against both the genome-wide
                      model it started from and the manuscript's own
                      per-chromosome model for that chromosome.

Usage: python scripts/build_ablation_tables.py
Writes Results/reference_merge_ablation.csv
       Results/reference_merge_ablation_chr19.csv
       Results/region_models_by_frequency_bin.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_cells
import manuscript_supp

ANALYSIS, OUTDIR = "analysis", "Results"
RANDS, MISSING = [0, 42, 1024], ["5%", "15%", "25%"]
SEG = "seg1024_overlap128"
BIN_ORDER = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%",
             "30%~40%", "40%~50%"]

# label -> (runId, epoch, description of how the reference was merged)
ABLATION = {
    "no reference": ("SGDP_ablation_noreference", None,
                     "SGDP training split only"),
    "unaligned reference": ("SGDP_ablation_unaligned", None,
                            "1KGP samples merged on the 1KGP variant axis"),
    "aligned reference": ("SGDP_ablation_aligned", None,
                          "1KGP samples projected onto the SGDP variant axis"),
}
# label -> (dataset, chrom, runId, epoch)
SPECIALISATION = {
    "1KGP chr22 genome-wide": ("1KGP", 22, "1KGP_genomewide", 70),
    "1KGP chr22 specialised": ("1KGP", 22, "1KGP_finetuned_anchor", 30),
    "SGDP chr19 genome-wide": ("SGDP", 19, "SGDP_genomewide", "best"),
    "SGDP chr19 specialised": ("SGDP", 19, "SGDP_finetuned_anchor", "best"),
}
# manuscript supplementary table indices, when the DOCX can be read
MS_TABLES = {("1KGP", 22): {"5%": 0, "15%": 1, "25%": 2},
             ("SGDP", 19): {"5%": 12, "15%": 13, "25%": 14}}
MS_BINS = ["<=0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%", "30%~40%",
           "40%~50%", "Overall"]
BIN_ALIAS = {"0.1%~0.5%": "<=0.5%"}
SUPP = manuscript_supp.find()


# Marker set a row is scored on. Cells on the full target axis carry the _FA
# flag and exist only where a rescore was run.
DENOMINATORS = (("BAT", "Beagle intersection"), ("BAT_FA", "full target axis"))


def cells(run_id, epoch, dataset, chrom, seg, miss, flags="BAT"):
    frames, used = [], []
    for rs in RANDS:
        p = analysis_cells.resolve(ANALYSIS, run_id, "phased", "biunet", flags,
                                   rs, dataset, chrom, "ALL", seg, miss,
                                   epoch=epoch)
        if p is None:
            continue
        used.append(p)
        frames.append(pd.read_csv(p))
    if not frames:
        return None, []
    df = pd.concat(frames)
    df["MAF_bin"] = df["MAF_bin"].astype(str).str.strip()
    return df, used


def per_bin(df):
    g = df.groupby("MAF_bin").agg(R2=("Bin_R2", "mean"), Acc=("Bin_Acc", "mean"),
                                  N=("Num_SNPs", "mean"))
    out = {b: (g.loc[b, "R2"], g.loc[b, "Acc"], g.loc[b, "N"]) for b in g.index}
    out["Overall"] = (df["Overall R2"].mean(), df["Overall Acc"].mean(),
                      df[df.MAF_bin == "Overall"]["Num_SNPs"].mean())
    return out


def _supp_table(dataset, chrom, miss):
    key = (dataset, chrom)
    if key not in MS_TABLES or not SUPP:
        return None
    try:
        import docx
    except ImportError:
        return None
    return docx.Document(SUPP).tables[MS_TABLES[key][miss]]


def manuscript(dataset, chrom, miss):
    t = _supp_table(dataset, chrom, miss)
    if t is None:
        return None
    for row in t.rows:
        c = [x.text.strip() for x in row.cells]
        if c[0] == "R2" and c[1] == "BiU-Net":
            return {b: float(v) for b, v in zip(MS_BINS, c[2:10])}
    return None


def manuscript_nsnps(dataset, chrom, miss):
    """Marker count behind each published bin, used to check like-for-like.

    The published rarest bin is labelled <=0.5% while ours starts at 0.1%, so the
    labels alone cannot say whether the two cover the same markers; the counts
    can, and on 1KGP chr22 they agree exactly.
    """
    t = _supp_table(dataset, chrom, miss)
    if t is None:
        return None
    for row in t.rows:
        c = [x.text.strip() for x in row.cells]
        if c[0] == "#SNPs" and c[1] == "#SNPs":
            return {b: int(v.replace(",", "")) for b, v in zip(MS_BINS, c[2:10])}
    return None


os.makedirs(OUTDIR, exist_ok=True)
prov = []

# ---- alignment ablation (genome-wide, weighted across chromosomes) --------
rows = []
for label, (run_id, epoch, how) in ABLATION.items():
    for miss in MISSING:
        frames, used = [], []
        for c in range(1, 23):
            df, u = cells(run_id, epoch, "SGDP", c, SEG, miss)
            if df is not None:
                frames.append(df)
                used.extend(u)
        if not frames:
            print(f"NOTE: no cells for {label} {miss}")
            continue
        prov.extend(used)
        allc = pd.concat(frames)
        for b in BIN_ORDER + ["Overall"]:
            if b == "Overall":
                val = allc["Overall R2"].mean()
                n = allc["Num_SNPs"].sum()
            else:
                sub = allc[allc.MAF_bin == b]
                if sub.empty:
                    continue
                w = sub["Num_SNPs"].to_numpy(dtype=float)
                val = np.average(sub["Bin_R2"], weights=w) if w.sum() else np.nan
                n = w.sum()
            rows.append({"arm": label, "reference_handling": how, "missing": miss,
                         "MAF_bin": b, "R2": val, "Num_SNPs": n})
abl = pd.DataFrame(rows)
p1 = os.path.join(OUTDIR, "reference_merge_ablation.csv")
abl.to_csv(p1, index=False)
print(f"wrote {p1} ({len(abl)} rows)")

# ---- alignment ablation at chr19, on both denominators ----
rows = []
for label, (run_id, epoch, how) in ABLATION.items():
    for flags, denom in DENOMINATORS:
        for miss in MISSING:
            df, used = cells(run_id, epoch, "SGDP", 19, SEG, miss, flags=flags)
            if df is None:
                continue
            prov.extend(used)
            ms = manuscript("SGDP", 19, miss)
            for b, (r2, acc, n) in per_bin(df).items():
                ms_b = BIN_ALIAS.get(b, b)
                rows.append({"arm": label, "reference_handling": how,
                             "denominator": denom, "missing": miss, "MAF_bin": b,
                             "R2": r2, "Num_SNPs": n,
                             "R2_manuscript": ms.get(ms_b) if ms else None})
abl19 = pd.DataFrame(rows)
p3 = os.path.join(OUTDIR, "reference_merge_ablation_chr19.csv")
abl19.to_csv(p3, index=False)
print(f"wrote {p3} ({len(abl19)} rows)")

# ---- specialisation probes ----------------------------------------------
rows = []
for label, (dataset, chrom, run_id, epoch) in SPECIALISATION.items():
    for flags, denom in DENOMINATORS:
        for miss in MISSING:
            df, used = cells(run_id, epoch, dataset, chrom, SEG, miss, flags=flags)
            if df is None:
                if flags == "BAT":
                    print(f"NOTE: no cells for {label} {miss}")
                continue
            prov.extend(used)
            vals = per_bin(df)
            ms = manuscript(dataset, chrom, miss)
            ns = manuscript_nsnps(dataset, chrom, miss)
            for b, (r2, acc, n) in vals.items():
                ms_b = BIN_ALIAS.get(b, b)
                ms_n = ns.get(ms_b) if ns else None
                rows.append({
                    "model": label, "dataset": dataset, "chr": chrom,
                    "denominator": denom, "missing": miss, "MAF_bin": b,
                    "R2": r2, "Acc": acc, "Num_SNPs": n,
                    "Num_SNPs_manuscript": ms_n,
                    "R2_manuscript": ms.get(ms_b) if ms else None,
                    "R2_minus_manuscript": (r2 - ms[ms_b]) if ms and ms_b in ms else None,
                    "same_variant_set": (None if ms_n is None
                                         else int(round(n)) == int(ms_n)),
                })
spec = pd.DataFrame(rows)
p2 = os.path.join(OUTDIR, "region_models_by_frequency_bin.csv")
spec.to_csv(p2, index=False)
print(f"wrote {p2} ({len(spec)} rows)")
print(f"Cell provenance: {analysis_cells.describe(prov)}")
