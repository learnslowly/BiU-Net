"""Render the merged-reference figure from rev2_merge_effect_revision2.csv.
"""
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    # TrueType, not matplotlib's default Type 3: journals reject Type 3 and
    # vector converters drop its glyphs.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.linestyle': '-',
    'grid.alpha': 0.4,
    'grid.color': 'gray',
    'grid.linewidth': 0.5,
})

BINS = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%",
        "30%~40%", "40%~50%"]
LABELS = ["0.1–0.5%", "0.5–1%", "1–10%", "10–20%", "20–30%", "30–40%", "40–50%"]
MISSING = ["5%", "15%", "25%"]
NO_MERGE = "SGDP training split only"
MERGED = "reference merged on the SGDP axis"
C_NO, C_YES = "#8C8C8C", "#1F4E79"
# Solid fill rather than an alpha channel: the figure is embedded as EMF, which
# has no transparency, and a converter silently flattens alpha to black.
C_BAND = "#DCE4EE"

df = pd.read_csv("rev2_merge_effect_revision2.csv")
# every panel carries its own y axis: a reader comparing two panels should not
# have to trace a gridline back to the leftmost one to read a value
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)

for ax, miss in zip(axes, MISSING):
    d = df[df.missing == miss].set_index(["model", "MAF_bin"])["R2"]
    lo = [d[(NO_MERGE, b)] for b in BINS]
    hi = [d[(MERGED, b)] for b in BINS]
    x = range(len(BINS))
    # The shaded interval is the gain itself, so the reader reads the size of
    # the effect off the same axis as the two values that produce it.
    ax.fill_between(x, lo, hi, color=C_BAND, linewidth=0, zorder=0)
    ax.plot(x, lo, marker='o', markersize=5, linewidth=1.6, color=C_NO,
            label="SGDP training set only")
    ax.plot(x, hi, marker='o', markersize=5, linewidth=1.6, color=C_YES,
            label="+ 1KGP panel reindexed onto the SGDP axis")
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS, rotation=45, ha='right', fontsize=10)
    ax.set_title(f"{miss} of genotypes masked", fontsize=12, pad=8)
    ax.set_xlabel("Minor allele frequency", fontsize=11)
    ax.tick_params(axis='y', labelleft=True, labelsize=10)
    ax.set_ylabel(r"$\mathrm{R}^2$", fontsize=12)
    ov_lo = d[(NO_MERGE, "Overall")]
    ov_hi = d[(MERGED, "Overall")]
    ax.text(0.03, 0.06,
            f"Overall  {ov_lo:.4f} $\\rightarrow$ {ov_hi:.4f}  "
            f"({ov_hi - ov_lo:+.4f})",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.8"))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("figures/rev2_merge_effect_revision2.pdf", bbox_inches='tight')
fig.savefig("figures/rev2_merge_effect_revision2.png", dpi=300, bbox_inches='tight')
fig.savefig("figures/rev2_merge_effect_revision2.svg", bbox_inches='tight')
print("wrote Results/figures/rev2_merge_effect_revision2.pdf and .png")
