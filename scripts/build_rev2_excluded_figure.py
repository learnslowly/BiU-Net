"""Where the markers Beagle cannot call sit on the frequency spectrum.

Usage: python scripts/build_rev2_excluded_figure.py
Writes Results/figures/rev2_excluded_markers_revision2.pdf and .png plus caption.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "Results"
FIGDIR = os.path.join(OUTDIR, "figures")
SRC = os.path.join(OUTDIR, "beagle_excluded_markers_SGDP_revision2.csv")
BINS = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%", "30%~40%",
        "40%~50%"]
LABEL = ["0.1-0.5", "0.5-1", "1-10", "10-20", "20-30", "30-40", "40-50"]
# One ramp, dark for rare through light for common, so the reader sees at a
# glance that the bulk of the bar is not the rare end.
COLOURS = ["#08306B", "#28569E", "#4A81C0", "#7BA7D7", "#A8C6E4", "#CBDCEF",
           "#E4EEF8"]


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC}; run scripts/beagle_excluded_markers.py first")
    d = pd.read_csv(SRC)
    d["dropped"] = d["n_target"] - d["matched"]
    chrs = sorted(d["chr"].unique())

    counts = (d.pivot_table(index="chr", columns="MAF_bin", values="dropped",
                            aggfunc="sum")
              .reindex(chrs).reindex(BINS, axis=1).fillna(0))
    totals = (d.pivot_table(index="MAF_bin", values=["n_target", "dropped"],
                            aggfunc="sum").reindex(BINS))
    totals["rate"] = 100 * totals["dropped"] / totals["n_target"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6),
                             gridspec_kw={"width_ratios": [2.4, 1]})
    plt.rcParams.update({"font.size": 10})

    ax = axes[0]
    bottom = np.zeros(len(chrs))
    x = np.arange(len(chrs))
    for b, lab, col in zip(BINS, LABEL, COLOURS):
        v = counts[b].to_numpy(dtype=float)
        ax.bar(x, v, bottom=bottom, color=col, label=lab, width=0.78,
               edgecolor="white", linewidth=0.4)
        bottom += v
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in chrs], fontsize=8)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("Excluded markers")
    ax.set_title("(a)", loc="left", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(title="minor allele frequency (%)", ncol=4, fontsize=8.5,
              title_fontsize=8.5, frameon=False, loc="upper right")

    ax = axes[1]
    ax.bar(np.arange(len(BINS)), totals["rate"].to_numpy(), color=COLOURS,
           width=0.72, edgecolor="white", linewidth=0.4)
    ax.set_xticks(np.arange(len(BINS)))
    ax.set_xticklabels(LABEL, rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Minor allele frequency (%)")
    ax.set_ylabel("Excluded / band total (%)")
    ax.set_title("(b)", loc="left", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, "rev2_excluded_markers_revision2.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    tot = int(totals["dropped"].sum())
    rare = int(totals.loc[["0.1%~0.5%", "0.5%~1%"], "dropped"].sum())
    common = int(totals.loc[["10%~20%", "20%~30%", "30%~40%", "40%~50%"],
                            "dropped"].sum())
    print(f"excluded markers: {tot:,}; rare (<=1%) {rare:,} ({100*rare/tot:.1f}%); "
          f"common (>10%) {common:,} ({100*common/tot:.1f}%)")

    with open(path.replace(".pdf", "_caption.txt"), "w") as f:
        f.write(
            "Target markers with no reference match, SGDP, 22 autosomes. A "
            "marker is retained only when the panel holds a record with the "
            "same chromosome, position and allele list. (a) count of excluded "
            "markers per chromosome, segmented by minor allele frequency. "
            "(b) excluded markers in a frequency band over all target markers "
            "in that band, pooled across the 22 autosomes: 17,659 / 331,887 at "
            "0.1 to 0.5%, 112,485 / 3,213,493 at 1 to 10%.\n\n")
        f.write("*Source: Results/beagle_excluded_markers_SGDP_revision2.csv. "
                "Code: scripts/build_rev2_excluded_figure.py*\n")


if __name__ == "__main__":
    main()
