"""Chromosome-by-chromosome comparison of the delivered model against baselines.

Usage: python scripts/build_rev2_per_chromosome.py
Writes Results/figures/rev2_per_chromosome_revision2.pdf and .png plus caption.
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

ANALYSIS, OUTDIR = "analysis", "Results"
FIGDIR = os.path.join(OUTDIR, "figures")
RANDS, MISSING, CHRS = [0, 42, 1024], ["5%", "15%", "25%"], list(range(1, 23))
SEG = "seg1024_overlap128"
DATASETS = ["1KGP", "SGDP"]

C_MODEL, C_BEAGLE, C_SCDA = "#1F4E79", "#C44E52", "#55A868"
C_AHEAD, C_BEHIND = "#1F4E79", "#C44E52"

# dataset -> {label: (runId, method tag, seg tag, epoch)}
METHODS = {
    "1KGP": {
        "BiU-Net": ("v3low_ftchr", "biunet", SEG, 30),
        "Beagle": ("v3low_ref_g22", "beagle", SEG, None),
        "SCDA": ("scda_fl_g22", "biunet", "seg-1_overlap0", 286),
    },
    "SGDP": {
        "BiU-Net": ("ft_sgdp_ftchr", "biunet", SEG, "best"),
        "Beagle": ("ft_sgdp_aln", "beagle", SEG, None),
        "SCDA": ("scda_sgdp_scratch", "biunet", "seg-1_overlap0", "best"),
    },
}
SUSPECT = []


def overall(dataset, run_id, tag, seg, epoch, chrom, miss):
    """Overall R2 for one cell, averaged over seeds, or NaN when absent."""
    vals = []
    for rs in RANDS:
        p = analysis_cells.resolve(ANALYSIS, run_id, "phased", tag, "BAT", rs,
                                   dataset, chrom, "ALL", seg, miss, epoch=epoch)
        if p is None:
            continue
        df = pd.read_csv(p)
        df["MAF_bin"] = df["MAF_bin"].astype(str).str.strip()
        if len(analysis_cells.degenerate_rows(df)):
            SUSPECT.append(f"{dataset} {run_id} chr{chrom} {miss}")
            continue
        vals.append(df["Overall R2"].mean())
    return float(np.mean(vals)) if vals else np.nan


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(len(DATASETS), len(MISSING), figsize=(13.0, 10.5),
                             sharey=True)
    y = np.arange(len(CHRS))

    for r, ds in enumerate(DATASETS):
        data = {}
        for label, (run_id, tag, seg, epoch) in METHODS[ds].items():
            data[label] = {
                m: np.array([overall(ds, run_id, tag, seg, epoch, c, m)
                             for c in CHRS])
                for m in MISSING
            }
        for c, miss in enumerate(MISSING):
            ax = axes[r, c]
            ours, beagle, scda = (data["BiU-Net"][miss], data["Beagle"][miss],
                                  data["SCDA"][miss])
            for i in y:
                if np.isnan(ours[i]) or np.isnan(beagle[i]):
                    continue
                colour = C_AHEAD if ours[i] >= beagle[i] else C_BEHIND
                ax.plot([beagle[i], ours[i]], [i, i], color=colour, lw=2.4,
                        alpha=0.55, solid_capstyle="round", zorder=2)
            ax.scatter(scda, y, s=26, color=C_SCDA, zorder=3, label="SCDA")
            ax.scatter(beagle, y, s=30, color=C_BEAGLE, zorder=4, label="Beagle")
            ax.scatter(ours, y, s=34, color=C_MODEL, zorder=5,
                       label="BiU-Net, chromosome fine-tuned")
            ax.set_title(f"({'abcdef'[r * 3 + c]}) {miss} masked", loc="left",
                         fontsize=10.5)
            ax.grid(axis="x", alpha=0.25, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if c == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([str(x) for x in CHRS], fontsize=8)
                ax.set_ylabel(f"{ds}\nchromosome")
            ax.invert_yaxis()
            if r == len(DATASETS) - 1:
                ax.set_xlabel("phased $R^2$")

    handles = [
        Line2D([], [], marker="o", ls="", color=C_MODEL, ms=7,
               label="BiU-Net, chromosome fine-tuned"),
        Line2D([], [], marker="o", ls="", color=C_BEAGLE, ms=7, label="Beagle"),
        Line2D([], [], marker="o", ls="", color=C_SCDA, ms=6, label="SCDA"),
        Line2D([], [], color=C_AHEAD, lw=3, alpha=0.55,
               label="BiU-Net ahead of Beagle"),
        Line2D([], [], color=C_BEHIND, lw=3, alpha=0.55,
               label="Beagle ahead of BiU-Net"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.015), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    path = os.path.join(FIGDIR, "rev2_per_chromosome_revision2.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    if SUSPECT:
        print(f"NOTE: {len(SUSPECT)} degenerate cell(s) excluded: "
              + "; ".join(sorted(set(SUSPECT))))

    with open(path.replace(".pdf", "_caption.txt"), "w") as f:
        f.write(
            "Per-chromosome accuracy of the delivered model against the two "
            "baselines. Rows are datasets, columns are the fraction of "
            "genotypes masked, and each row within a panel is one autosome. "
            "Points are the phased four-class R2 pooled over all positions of "
            "that chromosome and averaged over three masking seeds. The "
            "connector joins Beagle to BiU-Net and is coloured by which of the "
            "two is higher. BiU-Net here is the genome-wide merged-data model "
            "after 30 epochs of fine-tuning on the chromosome being scored; the "
            "genome-wide checkpoint it starts from is a stage in that procedure "
            "and is reported in Table T4 rather than shown as a competitor.\n\n")
        f.write("*Source: benchmark cells in analysis/. "
                "Code: scripts/build_rev2_per_chromosome.py*\n")


if __name__ == "__main__":
    main()
