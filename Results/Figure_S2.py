import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Read the CSV file
df = pd.read_csv('S1.csv')

# Remove rows with NaN in 'Missingness' if necessary
df = df.dropna(subset=['Missingness'])

# Set MAF_Bin as a categorical variable with the correct order
maf_bin_order = sorted(df['MAF_Bin'].unique())
df['MAF_Bin'] = pd.Categorical(df['MAF_Bin'], categories=maf_bin_order, ordered=True)

# Get unique values for Missingness and Segment_Length
missingness_levels = sorted(df['Missingness'].unique())
segment_lengths = sorted(df['Segment_Length'].unique())

# Use GridSpec to create space for missingness titles
nrows = len(missingness_levels) * 2  # Double the rows to include title rows
ncols = len(segment_lengths)

# Adjust the figure size as needed
fig = plt.figure(figsize=(ncols * 4, len(missingness_levels) * 3.5))  # Increased height
plt.rcParams.update({
        'figure.dpi': 400,  # Increase DPI for high-resolution plots
        'savefig.dpi': 400,  # High resolution for saved figures     
        'font.size': 22,          # Default text size
        'axes.titlesize': 22,     # Axes title font size
        'axes.labelsize': 22,     # Axes labels font size
        'xtick.labelsize': 18,    # X tick labels font size
        'ytick.labelsize': 18,    # Y tick labels font size
        'legend.fontsize': 25,    # Legend font size
        'figure.titlesize': 25,   # Figure title font size
        'axes.grid': True,        # Enable grid
        'grid.linestyle': '-',    # Set grid line style
        'grid.alpha': 0.5,        # Set grid transparency
        'grid.color': 'gray',     # Set grid color
        'grid.linewidth': 0.5,    # Set grid line width
})

# Define height ratios: increased height for title rows
height_ratios = np.tile([0.2, 1], len(missingness_levels))  # Increased title row height to 20%

# Add more space between rows by increasing the height_ratios gap
gs = GridSpec(nrows, ncols, figure=fig, height_ratios=height_ratios, hspace=0.3)  # Increased hspace

# Initialize a list to hold axes
axes = []

# Loop over missingness levels and segment lengths to create subplots
for i, missingness in enumerate(missingness_levels):
    # Create a row for the missingness title
    title_row = i * 2  # Title rows are at even indices
    title_ax = fig.add_subplot(gs[title_row, :])
    title_ax.axis('off')  # Hide the axes
    # Convert missingness to percentage and format the title
    missingness_pct = f'{missingness * 100:.0f}%'
    title_ax.text(0.5, 0.5, f'Missingness: {missingness_pct}', ha='center', va='center', fontsize=14)
    
    row_axes = []
    for j, segment_length in enumerate(segment_lengths):
        subplot_row = title_row + 1  # Subplot rows are after title rows
        ax = fig.add_subplot(gs[subplot_row, j])
        row_axes.append(ax)
    axes.append(row_axes)

# Define the metrics to plot and colors
metrics = ['Bin_Acc', 'Bin_R2', 'Bin_IQS', 'Bin_Precision', 'Bin_Recall', 'Bin_F1']
colors = sns.color_palette("tab10", n_colors=len(metrics))

# Create a mapping from metrics to colors
metric_color_map = dict(zip(metrics, colors))

# Collect handles and labels for the legend
handles = []
labels = []

# First pass: plot data and collect y-axis limits for each row
row_ylims = []
for mi, missingness in enumerate(missingness_levels):
    row_min = float('inf')
    row_max = float('-inf')
    
    # Find min and max values for all plots in the row
    for si, segment_length in enumerate(segment_lengths):
        subset = df[(df['Missingness'] == missingness) & (df['Segment_Length'] == segment_length)]
        if not subset.empty:
            for metric in metrics:
                row_min = min(row_min, subset[metric].min())
                row_max = max(row_max, subset[metric].max())
    
    # Add some padding to the limits
    padding = (row_max - row_min) * 0.05
    row_ylims.append((row_min - padding, row_max + padding))

# Second pass: plot data with unified y-axis limits per row
for mi, missingness in enumerate(missingness_levels):
    for si, segment_length in enumerate(segment_lengths):
        ax = axes[mi][si]
        # Filter data for the current Missingness and Segment_Length
        subset = df[(df['Missingness'] == missingness) & (df['Segment_Length'] == segment_length)]
        if subset.empty:
            # If no data is available for this combination, turn off the axis
            ax.axis('off')
            continue
            
        # Ensure the data is sorted by MAF_Bin
        subset = subset.sort_values('MAF_Bin')
        
        # Set up secondary y-axis
        ax2 = ax.twinx()
        
        # Plot each metric over MAF_Bin on the primary y-axis
        for k, metric in enumerate(metrics):
            line, = ax.plot(
                subset['MAF_Bin'], subset[metric], label=metric, 
                marker='o', markersize=4, color=colors[k], linewidth=0.8
            )
            # Collect handles and labels for the legend only once
            if mi == 0 and si == 0:
                handles.append(line)
                labels.append(metric)
                
        # Set unified y-axis limits for the row
        ax.set_ylim(row_ylims[mi])
        
        # Plot the bar chart of Num_SNPs on the secondary y-axis
        ax2.bar(subset['MAF_Bin'], subset['Num_SNPs'], color='gray', alpha=0.3, label='Num_SNPs')
        
        # Set left y-axis (metrics) labels only for leftmost subplot
        if si == 0:  # Leftmost subplot
            ax.set_ylabel('Metrics Values', fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # Set right y-axis (Num_SNPs) labels only for rightmost subplot
        if si == len(segment_lengths) - 1:  # Rightmost subplot
            #ax2.set_ylabel('Num_SNPs', color='gray', fontsize=10)
            #ax2.tick_params(axis='y', labelcolor='gray', labelsize=10)
            ax2.set_ylabel('Num_SNPs', fontsize=10)
            ax2.tick_params(axis='y', labelsize=10)
        else:
            ax2.set_ylabel('')
            ax2.set_yticklabels([])
        
        # Set segment length titles for all rows
        ax.set_title(f'Segment Length: {segment_length}', fontsize=12, pad=12)
        
        # Set x-axis labels only for the last row
        ax.set_xticks(range(len(maf_bin_order)))
        if mi == len(missingness_levels) - 1:  # Last row
            ax.set_xticklabels(subset['MAF_Bin'], rotation=45, ha='right', fontsize=12)
            ax.set_xlabel('MAF Bin', fontsize=9)
        else:
            ax.set_xticklabels([])  # Hide x-axis labels for all other rows
            ax.set_xlabel('')  # Remove x-axis label for all other rows

# Adjust layout to prevent overlapping
#plt.tight_layout()

# Add more space at the bottom for the legend
plt.subplots_adjust(bottom=0.1)

# Add a single legend at the bottom of the subplots
fig.legend(handles, labels, loc='lower center', ncol=len(metrics), fontsize=12, bbox_to_anchor=(0.5, -0.02))
plt.savefig("../Figures/Figure_S2.pdf")
# Show the plot
plt.show()