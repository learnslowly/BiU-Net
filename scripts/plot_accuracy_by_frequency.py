"""One figure for the whole genome-wide comparison, resolved by allele frequency.

Usage: python scripts/plot_accuracy_by_frequency.py
Writes Results/figures/accuracy_by_frequency.pdf and .png plus its caption.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_cells

# Embed text as TrueType rather than matplotlib.s default Type 3.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


ANALYSIS, OUTDIR = "analysis", "Results"
FIGDIR = os.path.join(OUTDIR, "figures")
RANDS, MISSING, CHRS = [0, 42, 1024], ["5%", "15%", "25%"], list(range(1, 23))
SEG = "seg1024_overlap128"
BINS = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%", "30%~40%",
        "40%~50%"]
BIN_LABEL = ["0.1–0.5", "0.5–1", "1–10", "10–20", "20–30", "30–40", "40–50"]
DATASETS = ["1KGP", "SGDP"]
SUSPECT = []  # cells dropped for the constant-prediction signature

# label -> (runId, method tag, seg tag, epoch, colour, linewidth, z)
# Ordered so the two BiU-Net series draw last and sit above the baselines.
METHODS = {
    "1KGP": [
        ("Beagle", "1KGP_genomewide", "beagle", SEG, None, "#C44E52", 1.6, 2),
        ("SCDA", "1KGP_scda_genomewide", "biunet", "seg-1_overlap0", 286, "#55A868", 1.6, 2),
        ("BiU-Net, genome-wide", "1KGP_genomewide", "biunet", SEG, 70, "#8AB8D8", 1.8, 3),
        ("BiU-Net, chromosome fine-tuned", "1KGP_finetuned", "biunet", SEG, 30, "#1F4E79", 2.2, 4),
    ],
    "SGDP": [
        ("Beagle", "SGDP_ablation_aligned", "beagle", SEG, None, "#C44E52", 1.6, 2),
        ("SCDA", "SGDP_scda_genomewide", "biunet", "seg-1_overlap0", "best", "#55A868", 1.6, 2),
        ("BiU-Net, genome-wide", "SGDP_genomewide", "biunet", SEG, "best", "#8AB8D8", 1.8, 3),
        ("BiU-Net, chromosome fine-tuned", "SGDP_finetuned", "biunet", SEG, "best", "#1F4E79", 2.2, 4),
    ],
}


def series(dataset, run_id, tag, seg, epoch, miss):
    """Per-chromosome R2 for one method and masking rate, seeds averaged.

    Returns a (n_chromosomes, n_bins) array with NaN where a cell is absent, so
    a partially finished campaign narrows the band rather than moving the line.
    """
    out = np.full((len(CHRS), len(BINS)), np.nan)
    for i, c in enumerate(CHRS):
        frames = []
        for rs in RANDS:
            p = analysis_cells.resolve(ANALYSIS, run_id, "phased", tag, "BAT",
                                       rs, dataset, c, "ALL", seg, miss,
                                       epoch=epoch)
            if p is not None:
                frames.append(pd.read_csv(p))
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df["MAF_bin"] = df["MAF_bin"].astype(str).str.strip()
        bad = analysis_cells.degenerate_rows(df)
        if len(bad):
            SUSPECT.append(f"{dataset} {run_id} chr{c} {miss}: "
                           + ", ".join(sorted(set(bad["MAF_bin"]))))
            df = df.drop(bad.index)
            if df.empty:
                continue
        g = df.groupby("MAF_bin")["Bin_R2"].mean()
        for j, b in enumerate(BINS):
            if b in g.index:
                out[i, j] = g[b]
    return out


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    x = np.arange(len(BINS))
    fig, axes = plt.subplots(len(DATASETS), len(MISSING), figsize=(13.5, 7.2),
                             sharex=True)
    plt.rcParams.update({"font.size": 10})
    coverage = {}

    for r, ds in enumerate(DATASETS):
        row_min, row_max = 1.0, 0.0
        for c, miss in enumerate(MISSING):
            ax = axes[r, c]
            for label, run_id, tag, seg, epoch, colour, lw, z in METHODS[ds]:
                mat = series(ds, run_id, tag, seg, epoch, miss)
                n = int(np.isfinite(mat).any(axis=1).sum())
                coverage[(ds, label, miss)] = n
                if n == 0:
                    continue
                lo = np.nanmin(mat, axis=0)
                hi = np.nanmax(mat, axis=0)
                mid = np.nanmedian(mat, axis=0)
                ax.fill_between(x, lo, hi, color=colour, alpha=0.18, lw=0,
                                zorder=z)
                ax.plot(x, mid, color=colour, lw=lw, marker="o", ms=3.2,
                        zorder=z + 10, label=label if (r == 0 and c == 0) else None)
                row_min = min(row_min, np.nanmin(lo))
                row_max = max(row_max, np.nanmax(hi))
            ax.grid(axis="y", alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            ax.set_title(f"({'abcdef'[r * 3 + c]}) {miss} masked", loc="left",
                         fontsize=10.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        pad = 0.04 * (row_max - row_min) if row_max > row_min else 0.02
        for c in range(len(MISSING)):
            axes[r, c].set_ylim(row_min - pad, row_max + pad)
        axes[r, 0].set_ylabel(f"{ds}\nphased $R^2$")

    for c in range(len(MISSING)):
        axes[-1, c].set_xticks(x)
        axes[-1, c].set_xticklabels(BIN_LABEL, rotation=30, ha="right")
        axes[-1, c].set_xlabel("Minor allele frequency (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.5", alpha=0.35, lw=6))
    labels.append("range over the 22 autosomes")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    path = os.path.join(FIGDIR, "accuracy_by_frequency.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    if SUSPECT:
        print(f"NOTE: {len(SUSPECT)} cell(s) dropped as degenerate:")
        for line in SUSPECT:
            print(f"  {line}")
    partial = {k: v for k, v in coverage.items() if 0 < v < len(CHRS)}
    absent = [k for k, v in coverage.items() if v == 0]
    if partial:
        print("NOTE: series drawn from fewer than 22 chromosomes:")
        for (ds, label, miss), n in sorted(partial.items()):
            print(f"  {ds} {label} {miss}: {n}/22")
    if absent:
        print(f"NOTE: {len(absent)} series had no cells and were omitted.")

    with open(path.replace(".pdf", "_caption.txt"), "w") as f:
        f.write(
            "Imputation accuracy across the allele-frequency spectrum. Rows are "
            "datasets, columns are the fraction of genotypes masked. Each line "
            "is the median over the 22 autosomes of the phased four-class R2 "
            "pooled within a chromosome, averaged over three masking seeds, and "
            "the band around it spans the full range over those chromosomes. "
            "The two BiU-Net series differ only in whether the genome-wide "
            "model was fine-tuned on one chromosome for 30 epochs, so the "
            "vertical distance between them is the effect of that fine-tuning.\n\n")
        f.write("*Source: benchmark cells in analysis/. "
                "Code: scripts/plot_accuracy_by_frequency.py*\n")


if __name__ == "__main__":
    main()
