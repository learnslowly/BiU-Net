import pandas as pd
import matplotlib.pyplot as plt
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
        ax.plot(segment_labels, mean_values, marker='o', label=metric)
    
    ax.set_title(f"Missingness: {int(missingness * 100)}%")
    ax.set_xlabel("Segment Length")
    ax.set_ylabel("Metric Value")
    ax.grid(True)

# Add a single legend at the bottom
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(metrics), fontsize=12, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.suptitle("Metrics Across Segment Lengths and Missingness Levels", fontsize=14, y=1.02)
plt.savefig("Figure_S1.pdf")
plt.show()