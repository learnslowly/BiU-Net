#!/usr/bin/env python3
"""Emit SGDP chr19 comparison stats as a tab-separated printout per missing rate,
in the manuscript table row order (paste straight into the Word form):

    row 1: #SNPs (total / full per bin)
    row 2: #SNPs, Beagle (kept/overlapped per bin)
    row 3: MAF header
    rows 4-18: Acc / R2 / Prc / Rec / F1, each x {Beagle, SCDA, BiU-Net}
"""
import csv
import os

ANALYSIS = os.path.join(os.path.dirname(__file__), "analysis")
STATES = [0, 42, 1024]
RATES = [5, 15, 25]
METHODS = [
    ("Beagle", "v3_low_seg128_phased_beagle_BAT_rand{s}_SGDP_chr19_ALL_seg128_overlap16_missing{r}%.csv"),
    ("SCDA",   "scda_chr19_phased_biunet_BAT_rand{s}_SGDP_chr19_ALL_seg-1_overlap0_missing{r}%.csv"),
    ("BiU-Net","v3_low_seg128_phased_biunet_BAT_rand{s}_SGDP_chr19_ALL_seg128_overlap16_missing{r}%.csv"),
]
BINS = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%", "30%~40%", "40%~50%", "Overall"]
HDR  = ["<=0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%", "30%~40%", "40%~50%", "Overall"]
METRICS = [("Acc", "Bin_Acc"), ("R2", "Bin_R2"), ("Prc", "Bin_Precision"),
           ("Rec", "Bin_Recall"), ("F1", "Bin_F1")]


def read_csv(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["MAF_bin"].strip()] = row
    return out


def avg(method_tmpl, rate, col):
    """Mean of `col` per bin over the available state files. Returns {bin: float|None}."""
    acc = {}
    for s in STATES:
        p = os.path.join(ANALYSIS, method_tmpl.format(s=s, r=rate))
        if not os.path.exists(p):
            continue
        data = read_csv(p)
        for b, row in data.items():
            v = row.get(col)
            if v in (None, "", "None"):
                continue
            acc.setdefault(b, []).append(float(v))
    return {b: (sum(vs) / len(vs)) for b, vs in acc.items()}


def fmt_int_row(d):
    return "\t".join(str(int(round(d[b]))) if d.get(b) is not None else "NA" for b in BINS)


def fmt_metric_row(d):
    return "\t".join(f"{d[b]:.4f}" if d.get(b) is not None else "NA" for b in BINS)


def main():
    bt = {label: tmpl for label, tmpl in METHODS}
    for rate in RATES:
        print(f"\n========== {rate}% Missing ==========")
        # row 1: total (full) -- Num_SNPs_total lives in the Beagle CSV (after re-run)
        print(fmt_int_row(avg(bt["Beagle"], rate, "Num_SNPs_total")))
        # row 2: Beagle kept (overlapped)
        print(fmt_int_row(avg(bt["Beagle"], rate, "Num_SNPs")))
        # row 3: MAF header
        print("\t".join(HDR))
        # rows 4-18: metrics x methods
        for _mlabel, mcol in METRICS:
            for method in ("Beagle", "SCDA", "BiU-Net"):
                print(fmt_metric_row(avg(bt[method], rate, mcol)))


if __name__ == "__main__":
    main()
