import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({
        'figure.dpi': 400,
        'savefig.dpi': 400,
        'font.size': 22,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 25,
        'figure.titlesize': 25,
        'axes.grid': True,
        'grid.linestyle': '-',
        'grid.alpha': 0.5,
        'grid.color': 'gray',
        'grid.linewidth': 0.5,
})

def plot_metrics_with_segmentation(file_path, output_pdf, align_y=False, model_names=None, metrics_to_plot=None):
    # Load the data
    data = pd.read_csv(file_path)

    # Get unique model IDs and create model mapping
    unique_models = sorted(data['Model'].unique())
    if model_names is None:
        model_names = {model_id: str(model_id) for model_id in unique_models}
    
    # Map Model numbers/strings to names
    data['Model'] = data['Model'].map(model_names)

    # Extract unique MAF bins and sort them naturally
    maf_bins = sorted(data['MAF_Bin'].unique(), key=lambda x: float(x.split('~')[0].strip('%')) if '~' in x else float(x.strip('>=').strip('%')))
    data['MAF_bin'] = pd.Categorical(data['MAF_Bin'], categories=maf_bins, ordered=True)

    # Rename columns
    data.rename(columns={
        'Missingness': 'Missing Ratio',
        'Num_SNPs': '#snps',
        'Bin_Acc': 'Accuracy',
        'Bin_R2': 'R2',
        'Bin_IQS': 'IQS',
        'Bin_Precision': 'Precision',
        'Bin_Recall': 'Recall',
        'Bin_F1': 'F1'
    }, inplace=True)

    # Convert Missing Ratio to percentage string
    data['Missing Ratio'] = (data['Missing Ratio'] * 100).astype(int).astype(str) + '% Missing'

    # Prepare the list of metrics
    all_available_metrics = []
    for metric in ['Accuracy', 'R2', 'IQS', 'Precision', 'Recall', 'F1']:
        bin_column = f"Bin_{metric}" if metric != "Accuracy" and "Bin_" + metric in data.columns else metric
        if bin_column in data.columns:
            all_available_metrics.append(metric)
    
    # Use specified metrics if provided, otherwise use all available metrics
    if metrics_to_plot is not None:
        invalid_metrics = [m for m in metrics_to_plot if m not in all_available_metrics]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics requested: {invalid_metrics}. Valid options are: {all_available_metrics}")
        metrics = metrics_to_plot
    else:
        metrics = all_available_metrics

    # Extract overall metrics
    available_overall_metrics = []
    overall_column_mapping = {}
    
    for metric in metrics:
        overall_column = f"Overall_{metric}" if metric != "Accuracy" else "Overall_Acc"
        if overall_column in data.columns:
            available_overall_metrics.append(overall_column)
            overall_column_mapping[overall_column] = metric
    
    select_columns = ['Model', 'Missing Ratio'] + available_overall_metrics
    existing_columns = [col for col in select_columns if col in data.columns]
    
    overall_metrics = data[existing_columns].drop_duplicates()

    if len(available_overall_metrics) > 0:
        overall_metrics = pd.melt(overall_metrics, id_vars=['Model', 'Missing Ratio'],
                                value_vars=available_overall_metrics,
                                var_name='Metric', value_name='Overall Value')
        overall_metrics['Metric'] = overall_metrics['Metric'].map(overall_column_mapping)

    # Get unique missing ratios
    missing_ratios = sorted(data['Missing Ratio'].unique(), key=lambda x: float(x.split('%')[0]))

    # Generate color cycle and marker cycle for models
    color_list = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)[:10]
    marker_list = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', '<', '>']
    
    # Create a style dictionary for each model
    model_styles = {}
    for i, model_name in enumerate(model_names.values()):
        model_styles[model_name] = {
            'color': color_list[i % len(color_list)],
            'marker': marker_list[i % len(marker_list)]
        }

    # Initialize PDF
    pdf = PdfPages(output_pdf)
    
    # Create a single combined figure with all missing ratios as rows
    num_rows = len(missing_ratios)
    num_cols = len(metrics)
    
    # Calculate figure height (adjust based on number of rows)
    figure_height = min(num_rows * 5, 30)  # Cap at 30 inches for very large plots
    
    # Create a large figure with subplots arranged in a grid
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, 
                            figsize=(5 * num_cols, figure_height), 
                            sharex='col')
    
    # If there's only one row or one column, ensure axs is a 2D array
    if num_rows == 1 and num_cols == 1:
        axs = np.array([[axs]])
    elif num_rows == 1:
        axs = np.array([axs])
    elif num_cols == 1:
        axs = np.array([[ax] for ax in axs])
        
    # Process each missing ratio (row)
    for row_idx, missing_ratio in enumerate(missing_ratios):
        df_plot = data[data['Missing Ratio'] == missing_ratio]
        
        # Initialize min and max y-axis values for the current row
        y_min, y_max = float('inf'), float('-inf') if align_y else (None, None)
        
        # Process each metric (column)
        for col_idx, metric in enumerate(metrics):
            ax = axs[row_idx, col_idx]
            ax2 = ax.twinx()
            
            # Bar chart for '#snps' with potential differences between models
            df_snps = df_plot[['MAF_bin', 'Model', '#snps']].drop_duplicates().reset_index(drop=True)
            
            # Get unique MAF bins for this missing ratio
            maf_bins_list = sorted(df_snps['MAF_bin'].unique())
            
            # Set up the bar positions
            bar_positions = np.arange(len(maf_bins_list))
            
            # Convert to wide format for easier processing
            snp_counts_by_bin_model = {}
            for _, row in df_snps.iterrows():
                maf_bin = row['MAF_bin']
                model = row['Model']
                count = row['#snps']
                
                if maf_bin not in snp_counts_by_bin_model:
                    snp_counts_by_bin_model[maf_bin] = {}
                
                snp_counts_by_bin_model[maf_bin][model] = count
            
            # Are there differences in SNP counts between models?
            has_different_snp_counts = False
            for bin_counts in snp_counts_by_bin_model.values():
                counts = list(bin_counts.values())
                if len(counts) > 1 and not all(c == counts[0] for c in counts):
                    has_different_snp_counts = True
                    break
            
            # If all models have the same SNP counts, just show one set of bars
            if not has_different_snp_counts:
                # Get the first available counts
                snp_counts = []
                for maf_bin in maf_bins_list:
                    if maf_bin in snp_counts_by_bin_model and len(snp_counts_by_bin_model[maf_bin]) > 0:
                        snp_counts.append(next(iter(snp_counts_by_bin_model[maf_bin].values())))
                    else:
                        snp_counts.append(0)
                
                ax2.bar(bar_positions, snp_counts, alpha=0.3, color=model_styles[list(model_names.values())[0]]['color'], label='#SNPs')
                max_snp_count = max(snp_counts) if snp_counts else 0
            else:
                # Find the model with minimum counts
                min_model = None
                min_total_snps = float('inf')
                
                # Find which model consistently has the minimum SNP counts
                for model in model_names.values():
                    if model in df_plot['Model'].unique():
                        total_snps = 0
                        for maf_bin in maf_bins_list:
                            if maf_bin in snp_counts_by_bin_model and model in snp_counts_by_bin_model[maf_bin]:
                                total_snps += snp_counts_by_bin_model[maf_bin][model]
                        if total_snps < min_total_snps:
                            min_total_snps = total_snps
                            min_model = model
                
                # Get the base SNP counts from the model with minimum counts
                base_snp_counts = []
                for maf_bin in maf_bins_list:
                    if maf_bin in snp_counts_by_bin_model and min_model in snp_counts_by_bin_model[maf_bin]:
                        base_snp_counts.append(snp_counts_by_bin_model[maf_bin][min_model])
                    else:
                        base_snp_counts.append(0)
                
                # Plot the base SNPs with model-specific color
                ax2.bar(bar_positions, base_snp_counts, alpha=0.3, color=model_styles[min_model]['color'], 
                       label=f'SNPs ({min_model})')
                
                # Calculate maximum SNP count for y-axis scaling
                max_snp_count = 0
                
                # Use a single neutral color for all extra SNPs (light gray)
                extra_snps_color = 'gray'
                all_models_have_extra_snps = False
                
                # First, check if any model has extra SNPs
                for model in model_names.values():
                    if model == min_model or model not in df_plot['Model'].unique():
                        continue
                    
                    model_has_extra = False
                    for i, maf_bin in enumerate(maf_bins_list):
                        if maf_bin in snp_counts_by_bin_model and model in snp_counts_by_bin_model[maf_bin]:
                            model_count = snp_counts_by_bin_model[maf_bin][model]
                            base_count = base_snp_counts[i]
                            if model_count > base_count:
                                model_has_extra = True
                                all_models_have_extra_snps = True
                                break
                    
                    if model_has_extra:
                        break
                
                # If any model has extra SNPs, show them as a single extra SNPs category
                if all_models_have_extra_snps:
                    # Calculate the maximum extra SNPs across all models for each bin
                    max_extra_by_bin = []
                    for i, maf_bin in enumerate(maf_bins_list):
                        base_count = base_snp_counts[i]
                        max_extra = 0
                        
                        for model in model_names.values():
                            if model == min_model or model not in df_plot['Model'].unique():
                                continue
                                
                            if maf_bin in snp_counts_by_bin_model and model in snp_counts_by_bin_model[maf_bin]:
                                model_count = snp_counts_by_bin_model[maf_bin][model]
                                extra = max(0, model_count - base_count)
                                max_extra = max(max_extra, extra)
                                max_snp_count = max(max_snp_count, base_count + max_extra)
                        
                        max_extra_by_bin.append(max_extra)
                    
                    # Plot the maximum extra SNPs with a neutral color
                    if any(extra > 0 for extra in max_extra_by_bin):
                        ax2.bar(bar_positions, max_extra_by_bin, bottom=base_snp_counts, 
                               alpha=0.15, color=extra_snps_color, 
                               label='Extra SNPs (other models)')
            
            # Adjust the y-axis scale of #snps to make it less dominant
            ax2.set_ylim(0, max_snp_count * 1.1 if max_snp_count > 0 else 1)
            
            # Plot each model
            for model in model_names.values():
                df_model = df_plot[df_plot['Model'] == model]
                
                # Skip if this model doesn't have data for this missing ratio
                if df_model.empty:
                    continue
                    
                style = model_styles[model]
                ax.plot(bar_positions, df_model[metric], marker=style['marker'], 
                        markersize=6, color=style['color'], label=model)
                
                # Get overall metric value for this model, if available
                overall_value = None
                if 'Overall Value' in overall_metrics.columns:
                    overall_val_df = overall_metrics.query(
                        f"Model == '{model}' and `Missing Ratio` == '{missing_ratio}' and Metric == '{metric}'"
                    )
                    if not overall_val_df.empty:
                        overall_value = overall_val_df['Overall Value'].values[0]
                
                # Update y-axis limits if align_y is True
                if align_y:
                    y_min = min(y_min, df_model[metric].min())
                    y_max = max(y_max, df_model[metric].max())
                    if overall_value is not None:
                        y_min = min(y_min, overall_value)
                        y_max = max(y_max, overall_value)
            
            # Add overall metrics as semi-transparent text, if available
            y_pos = 0.5  # Vertical position in axes coordinates (middle)
            
            for model in model_names.values():
                # Skip if this model doesn't have data for this missing ratio
                if model not in df_plot['Model'].unique():
                    continue
                
                # Get style for this model
                style = model_styles[model]
                
                # Get overall metric value if available
                overall_value = None
                if 'Overall Value' in overall_metrics.columns:
                    overall_val_df = overall_metrics.query(
                        f"Model == '{model}' and `Missing Ratio` == '{missing_ratio}' and Metric == '{metric}'"
                    )
                    
                    if not overall_val_df.empty:
                        overall_value = overall_val_df['Overall Value'].values[0]
                        # Add text at a position based on the model index
                        model_idx = list(model_names.values()).index(model)
                        x_pos = 0.5  # Horizontal position in axes coordinates (middle)
                        
                        # Split the text into model name and value parts for alignment
                        model_text = f"{model}:"
                        value_text = f"{overall_value:.4f}"
                        
                        # Add text with model color but semi-transparent, aligned at colon
                        ax.text(x_pos - 0.02, y_pos - 0.15 * model_idx, 
                                model_text, 
                                transform=ax.transAxes,
                                ha='right', va='center',
                                color=style['color'],
                                alpha=0.8,
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1),
                                fontsize=20,
                                weight='normal')
                        
                        # Add the value text separately, left-aligned after the colon
                        ax.text(x_pos + 0.02, y_pos - 0.15 * model_idx, 
                                value_text, 
                                transform=ax.transAxes,
                                ha='left', va='center',
                                color=style['color'],
                                alpha=0.8,
                                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1),
                                fontsize=20,
                                weight='normal')
            
            ax.set_xticks(bar_positions)
            
            # Get labels for each position
            x_labels = [maf_bins_list[i] for i in range(len(bar_positions))]
            
            # Only add x-tick labels for the bottom row
            if row_idx == num_rows - 1:
                ax.set_xticklabels(x_labels, rotation=45, ha='right')
            else:
                ax.set_xticklabels([])
            
            # Add metric title only to the top row
            if row_idx == 0:
                ax.set_title(metric)
            
            # Add missing ratio label to leftmost column
            if col_idx == 0:
                ax.text(-0.2, 0.5, missing_ratio, 
                       transform=ax.transAxes, 
                       ha='right', va='center', 
                       fontsize=22, 
                       rotation='vertical')
            
            ax.grid(alpha=0.3)
            
            # Explicitly set the twin axis label position to right
            ax2.yaxis.set_label_position("right")
            
            # Hide the right y-axis labels to reduce clutter
            if col_idx < num_cols - 1:
                ax2.tick_params(axis='y', labelright=False)
            else:
                # For the rightmost plot, show the right y-axis ticks and labels
                ax2.tick_params(axis='y', labelcolor='gray', labelsize=14)
                # Set the ylabel for #SNPs only on the rightmost plot and ensure it's on the right
                ax2.set_ylabel('#SNPs', color='gray', fontsize=18)
        
        # Set unified y-axis limits for all subplots in this row if align_y is True
        if align_y and y_min < y_max:  # Check that we have valid limits
            for col_idx in range(num_cols):
                axs[row_idx, col_idx].set_ylim(max(0, y_min * 0.9), y_max * 1.1)
                
        # Handle y-axis labels based on alignment choice
        if align_y:
            # When y-axes are aligned, only show labels on leftmost plot
            for col_idx in range(num_cols):
                if col_idx > 0:  # Not the leftmost
                    axs[row_idx, col_idx].tick_params(axis='y', labelleft=False)
                
                # Get the twin axis
                ax2 = axs[row_idx, col_idx].get_shared_y_axes().get_siblings(axs[row_idx, col_idx])[0]
                
                # Hide all right y-axis labels except for the rightmost plot
                if col_idx < num_cols - 1:  # Not the rightmost
                    ax2.tick_params(axis='y', labelright=False)
        else:
            # When y-axes are not aligned, show left labels on all plots
            # but still only show right label on rightmost plot
            for col_idx in range(num_cols):
                # Get the twin axis
                ax2 = axs[row_idx, col_idx].get_shared_y_axes().get_siblings(axs[row_idx, col_idx])[0]
                
                # Hide all right y-axis labels except for the rightmost plot
                if col_idx < num_cols - 1:  # Not the rightmost
                    ax2.tick_params(axis='y', labelright=False)
    
    # Create legend elements, separating models and SNP counts
    model_legend_elements = []
    snp_legend_elements = []
    
    # First add all models to the model legend
    for model_name, style in model_styles.items():
        # Only include models that have data for any missing ratio
        if model_name in data['Model'].unique():
            model_legend_elements.append(
                Line2D([0], [0], marker=style['marker'], color=style['color'], 
                       label=model_name, linestyle='-', markersize=8)
            )
    
    # Check if models have different SNP counts
    has_different_snp_counts = False
    min_model = None
    min_total_snps = float('inf')
    
    # Find which model consistently has the minimum SNP counts
    for model in model_names.values():
        if model in data['Model'].unique():
            total_snps = data[data['Model'] == model]['#snps'].sum()
            if total_snps < min_total_snps:
                min_total_snps = total_snps
                min_model = model
    
    # Check if any model has a different number of SNPs than min_model
    if min_model:
        for model in model_names.values():
            if model != min_model and model in data['Model'].unique():
                maf_bins_to_check = data['MAF_bin'].unique()
                for maf_bin in maf_bins_to_check:
                    min_snps = data[(data['Model'] == min_model) & (data['MAF_bin'] == maf_bin)]['#snps'].values
                    model_snps = data[(data['Model'] == model) & (data['MAF_bin'] == maf_bin)]['#snps'].values
                    
                    if len(min_snps) > 0 and len(model_snps) > 0 and min_snps[0] != model_snps[0]:
                        has_different_snp_counts = True
                        break
                if has_different_snp_counts:
                    break
    
    # Add SNP elements to the SNP legend
    if has_different_snp_counts and min_model:
        # Use a Rectangle instead of Line2D for the base SNPs
        snp_legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color=model_styles[min_model]['color'], 
                         alpha=0.3, label=f'SNPs ({min_model})')
        )
        
        # Add a single entry for extra SNPs
        snp_legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color='gray', 
                         alpha=0.15, label='Extra SNPs (other models)')
        )
    else:
        # If no differences, just add a single SNP element
        snp_legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.3, label='#SNPs')
        )
    
    # Create two separate legends
    # Models legend on top
    if model_legend_elements:
        model_legend = fig.legend(handles=model_legend_elements, loc='lower center', 
                bbox_to_anchor=(0.5, -0.02), ncol=min(len(model_legend_elements), 3))
    
    # SNP legend below models legend
    if snp_legend_elements:
        if model_legend_elements:
            # Position the SNP legend below the model legend
            fig.legend(handles=snp_legend_elements, loc='lower center', 
                    bbox_to_anchor=(0.5, -0.1), ncol=min(len(snp_legend_elements), 3))
    
    # Add an overall title
    fig.suptitle('Performance Metrics by Missing Ratio', y=0.995, fontsize=30)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.99])  # Increased bottom margin for legend
    plt.subplots_adjust(hspace=0.3, wspace=0.2)  # Adjust spacing between subplots
    
    # Save the combined figure to PDF
    pdf.savefig(fig, bbox_inches='tight')
    
    # Close the PDF file
    pdf.close()
    print(f"All figures saved to {output_pdf}")
    
    # Display the figure
    #plt.show()
    
    return fig

plot_metrics_with_segmentation(
    'exp1_1KGP.csv', 
    '../Figures/Figure_1.pdf', 
    align_y=False
)

plot_metrics_with_segmentation(
    'exp1_LOS.csv', 
    '../Figures/Figure_S5.pdf', 
    align_y=False
)

plot_metrics_with_segmentation(
    'exp1_HLA.csv', 
    '../Figures/Figure_S6.pdf', 
    align_y=False
)

plot_metrics_with_segmentation(
    'exp1_SGDP.csv', 
    '../Figures/Figure_S7.pdf', 
    align_y=False
)

plot_metrics_with_segmentation(
    'exp2_LOS_AA.csv', 
    '../Figures/Figure_S8.pdf', 
    align_y=False
)

plot_metrics_with_segmentation(
    'exp2_LOS_CA.csv', 
    '../Figures/Figure_S9.pdf', 
    align_y=False
)