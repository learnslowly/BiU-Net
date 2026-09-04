#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

# A metric's column name and the way it is written on the figure are two
# different things: R2 keys the data, R with a superscript 2 is how the axis
# says it. \mathrm keeps the letter upright, matching the way the text sets
# these names; mathtext would otherwise italicise it.
def metric_label(name):
    stem = name.replace("Overall_", "").replace("Bin_", "")
    shown = {"R2": r"$\mathrm{R}^2$", "F1": r"$\mathrm{F}_1$",
             "Acc": "Accuracy"}.get(stem, stem)
    return shown


plt.rcParams.update({
    # Embed text as TrueType rather than matplotlib.s default Type 3.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
plt.rcParams.update({
    'figure.dpi': 300,  # Increase DPI for high-resolution plots
    'savefig.dpi': 300,  # High resolution for saved figures
    'font.size': 12,
    'legend.fontsize': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'figure.dpi': 100,  # Ensure consistent resolution
})
# Load the CSV file
file_path = 'S1.csv'
data = pd.read_csv(file_path)

# Filter the relevant metrics for plotting
metrics = ['Overall_Acc', 'Overall_R2', 'Overall_IQS', 'Overall_Precision', 
           'Overall_Recall', 'Overall_F1']
missingness_levels = data['Missingness'].unique()
segment_lengths = sorted(data['Segment_Length'].unique())

# Prepare subplots with a single legend at the bottom
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False, dpi=300)
axes = axes.flatten()

# Convert segment lengths to strings for labeling
segment_labels = [str(length) for length in segment_lengths]

# Plot each missingness level in a subplot
for i, missingness in enumerate(sorted(missingness_levels)):
    ax = axes[i]
    subset = data[data['Missingness'] == missingness]
    
    for metric in metrics:
        mean_values = subset.groupby('Segment_Length')[metric].mean()
        ax.plot(segment_labels, mean_values, marker='o', label=metric_label(metric))
    
    ax.set_title(f"Missingness: {int(missingness * 100)}%")
    ax.set_xlabel("Segment Length")
    ax.set_ylabel("Metric Value")
    ax.grid(True)

# Add a single legend at the bottom
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(metrics), fontsize=12, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
# The legend sits below the axes, outside the figure canvas, so a plain savefig
# crops it away and the six series are left unlabelled. bbox_inches="tight"
# grows the saved area until it contains the legend. The figure title is gone
# with it: the title of a figure belongs in its caption, not drawn inside it.
for ext in ("pdf", "svg", "png"):
    plt.savefig(f"../Figures/Figure_S1.{ext}", bbox_inches="tight")
plt.show()
