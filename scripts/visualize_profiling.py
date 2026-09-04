#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization script for training profiling data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from matplotlib.cm import get_cmap
import argparse


def visualize_profiling_data(profile_dir):
    """
    Visualize the profiling data collected during training.
    
    Args:
        profile_dir: Directory containing profiling logs
        
    Returns:
        bool: True if visualization was successful, False otherwise
    """
    # Create output directory for plots
    plots_dir = os.path.join(profile_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load the profiling data
    timing_file = os.path.join(profile_dir, 'custom_timing.txt')
    preproc_file = os.path.join(profile_dir, 'preprocessing_timing.txt')
    allreduce_file = os.path.join(profile_dir, 'allreduce_timing.txt')
    
    plot_count = 0
    
    # Process custom timing file
    if os.path.exists(timing_file):
        try:
            print(f"Processing timing data from {timing_file}")
            timing_df = pd.read_csv(timing_file)
            
            # 1. Overall operation timing
            plt.figure(figsize=(12, 8))
            if 'Dataset' in timing_df.columns and 'Batch' in timing_df.columns:
                # Group by operation and calculate average duration
                op_timing = timing_df.groupby('Operation')['Duration(ms)'].mean().sort_values(ascending=False)
            else:
                # Fall back to simpler grouping if Dataset and Batch columns don't exist
                op_timing = timing_df.groupby('Operation')['Duration(ms)'].mean().sort_values(ascending=False)
            
            op_timing.plot(kind='bar', color='skyblue')
            plt.title('Average Duration of Operations')
            plt.ylabel('Duration (ms)')
            plt.xlabel('Operation')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, 'operation_timing.png')
            plt.savefig(plot_path)
            print(f"Saved plot to {plot_path}")
            plot_count += 1
            
            # 2. Operation timing by batch
            if 'Dataset' in timing_df.columns and 'Batch' in timing_df.columns:
                plt.figure(figsize=(14, 8))
                pivot_data = timing_df.pivot_table(
                    index=['Dataset', 'Batch'],
                    columns='Operation',
                    values='Duration(ms)'
                ).reset_index()
                
                # Get top 5 operations by average duration
                top_ops = op_timing.head(5).index.tolist()
                
                # For each top operation, plot its duration over batches
                for op in top_ops:
                    if op in pivot_data.columns:
                        plt.plot(range(len(pivot_data)), pivot_data[op], marker='o', label=op)
                
                plt.title('Top Operation Duration by Batch')
                plt.xlabel('Batch Index')
                plt.ylabel('Duration (ms)')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, 'operation_timing_by_batch.png')
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")
                plot_count += 1
                
            # 3. Stacked bar chart of operation timing percentages
            if 'Dataset' in timing_df.columns and 'Batch' in timing_df.columns:
                plt.figure(figsize=(14, 8))
                
                # Group by batch and operation
                batch_op_timing = timing_df.pivot_table(
                    index=['Dataset', 'Batch'], 
                    columns='Operation', 
                    values='Duration(ms)',
                    aggfunc='mean'
                ).fillna(0)
                
                # Calculate percentage of total time for each batch
                batch_totals = batch_op_timing.sum(axis=1)
                batch_op_percent = batch_op_timing.div(batch_totals, axis=0) * 100
                
                # Get a color map
                cmap = get_cmap('tab20')
                colors = [cmap(i/20) for i in range(min(20, len(batch_op_timing.columns)))]
                
                # Plot the stacked bar chart (first 10 batches for readability)
                if len(batch_op_percent) > 10:
                    batch_op_percent_subset = batch_op_percent.iloc[:10]
                else:
                    batch_op_percent_subset = batch_op_percent
                    
                batch_op_percent_subset.plot(kind='bar', stacked=True, figsize=(14, 8), color=colors)
                plt.title('Operation Time Percentage by Batch')
                plt.ylabel('Percentage of Total Time')
                plt.xlabel('(Dataset, Batch)')
                plt.legend(title='Operation', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, 'operation_percentage.png')
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")
                plot_count += 1
                
        except Exception as e:
            print(f"Error processing timing data: {e}")
    
    # Process preprocessing file
    if os.path.exists(preproc_file):
        try:
            print(f"Processing preprocessing data from {preproc_file}")
            preproc_df = pd.read_csv(preproc_file)
            
            # Detailed preprocessing breakdown
            plt.figure(figsize=(12, 8))
            if 'Dataset' in preproc_df.columns and 'Batch' in preproc_df.columns:
                # Group by step and calculate average duration
                preproc_timing = preproc_df.groupby('Step')['Duration(ms)'].mean().sort_values(ascending=False)
            else:
                # Fall back to simpler grouping
                preproc_timing = preproc_df.groupby('Step')['Duration(ms)'].mean().sort_values(ascending=False)
            
            preproc_timing.plot(kind='bar', color='lightgreen')
            plt.title('Average Duration of Preprocessing Steps')
            plt.ylabel('Duration (ms)')
            plt.xlabel('Preprocessing Step')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, 'preprocessing_timing.png')
            plt.savefig(plot_path)
            print(f"Saved plot to {plot_path}")
            plot_count += 1
            
            # Preprocessing timing by batch (if available)
            if 'Dataset' in preproc_df.columns and 'Batch' in preproc_df.columns:
                plt.figure(figsize=(14, 8))
                pivot_data = preproc_df.pivot_table(
                    index=['Dataset', 'Batch'],
                    columns='Step',
                    values='Duration(ms)'
                ).reset_index()
                
                # Get top steps by average duration
                top_steps = preproc_timing.head(5).index.tolist()
                
                # For each top step, plot its duration over batches
                for step in top_steps:
                    if step in pivot_data.columns:
                        plt.plot(range(len(pivot_data)), pivot_data[step], marker='o', label=step)
                
                plt.title('Preprocessing Step Duration by Batch')
                plt.xlabel('Batch Index')
                plt.ylabel('Duration (ms)')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, 'preprocessing_timing_by_batch.png')
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")
                plot_count += 1
                
        except Exception as e:
            print(f"Error processing preprocessing data: {e}")
    
    # Process AllReduce file
    if os.path.exists(allreduce_file):
        try:
            print(f"Processing AllReduce data from {allreduce_file}")
            allreduce_df = pd.read_csv(allreduce_file)
            
            # AllReduce operations by name
            plt.figure(figsize=(12, 8))
            if 'AllReduce' in allreduce_df.columns:
                allreduce_timing = allreduce_df.groupby('AllReduce')['Duration(ms)'].mean().sort_values(ascending=False)
                allreduce_timing.plot(kind='bar', color='coral')
                plt.title('Average Duration of AllReduce Operations')
                plt.ylabel('Duration (ms)')
                plt.xlabel('AllReduce Operation')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, 'allreduce_timing.png')
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")
                plot_count += 1
            
            # AllReduce size vs duration scatter plot
            if 'Size(bytes)' in allreduce_df.columns:
                plt.figure(figsize=(12, 8))
                # Create a scatter plot of size vs duration
                plt.scatter(
                    allreduce_df['Size(bytes)'] / 1024 / 1024,  # Convert to MB
                    allreduce_df['Duration(ms)'],
                    alpha=0.7, 
                    c=allreduce_df.index,
                    cmap='viridis'
                )
                plt.title('AllReduce Duration vs Data Size')
                plt.xlabel('Data Size (MB)')
                plt.ylabel('Duration (ms)')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add a colorbar
                cbar = plt.colorbar()
                cbar.set_label('AllReduce Operation Index')
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, 'allreduce_size_duration.png')
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")
                plot_count += 1
            
            # AllReduce operations comparisons by name and size
            if 'AllReduce' in allreduce_df.columns and 'Size(bytes)' in allreduce_df.columns:
                plt.figure(figsize=(14, 8))
                
                # Group by operation name and get average size and duration
                grouped = allreduce_df.groupby('AllReduce').agg({
                    'Size(bytes)': 'mean',
                    'Duration(ms)': 'mean'
                })
                
                # Create bubble chart where size is proportional to data size
                sizes = (grouped['Size(bytes)'] / 1024 / 1024)  # Convert to MB
                max_size = 1000  # Maximum bubble size
                min_size = 50    # Minimum bubble size
                
                if sizes.max() > sizes.min():
                    normalized_sizes = ((sizes - sizes.min()) / (sizes.max() - sizes.min()) 
                                        * (max_size - min_size) + min_size)
                else:
                    normalized_sizes = [min_size] * len(sizes)
                
                plt.scatter(
                    range(len(grouped)),
                    grouped['Duration(ms)'],
                    s=normalized_sizes,
                    alpha=0.6,
                    c=range(len(grouped)),
                    cmap='plasma'
                )
                
                plt.title('AllReduce Operations: Duration vs Size')
                plt.xlabel('AllReduce Operation')
                plt.ylabel('Average Duration (ms)')
                plt.xticks(range(len(grouped)), grouped.index, rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Add size annotation
                for i, (idx, row) in enumerate(grouped.iterrows()):
                    plt.annotate(
                        f"{row['Size(bytes)']/1024/1024:.2f} MB",
                        (i, row['Duration(ms)']),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8
                    )
                
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, 'allreduce_comparison.png')
                plt.savefig(plot_path)
                print(f"Saved plot to {plot_path}")
                plot_count += 1
                
        except Exception as e:
            print(f"Error processing AllReduce data: {e}")
    
    # Overall summary
    print(f"\nGenerated {plot_count} visualization plots in {plots_dir}")
    print("To view the plots, open the PNG files in your preferred image viewer")
    
    # Print summary of analysis
    print("\nSummary of profiling analysis:")
    
    if os.path.exists(timing_file):
        try:
            timing_df = pd.read_csv(timing_file)
            op_timing = timing_df.groupby('Operation')['Duration(ms)'].agg(['mean', 'sum']).sort_values('mean', ascending=False)
            total_time = op_timing['sum'].sum()
            
            print("\nOperation Timing Summary (sorted by average duration):")
            print(f"{'Operation':<25} {'Avg (ms)':<10} {'Total (ms)':<10} {'% of Total':<10}")
            print("-" * 60)
            
            for op, (avg, total) in op_timing.iterrows():
                percentage = total / total_time * 100 if total_time > 0 else 0
                print(f"{op:<25} {avg:<10.2f} {total:<10.2f} {percentage:<10.2f}%")
            
            print(f"\nTotal recorded time: {total_time:.2f} ms")
        except Exception as e:
            print(f"Error generating timing summary: {e}")
    
    if os.path.exists(preproc_file):
        try:
            preproc_df = pd.read_csv(preproc_file)
            preproc_timing = preproc_df.groupby('Step')['Duration(ms)'].agg(['mean', 'sum']).sort_values('mean', ascending=False)
            
            print("\nPreprocessing Steps Summary (sorted by average duration):")
            print(f"{'Step':<25} {'Avg (ms)':<10} {'Total (ms)':<10}")
            print("-" * 50)
            
            for step, (avg, total) in preproc_timing.iterrows():
                print(f"{step:<25} {avg:<10.2f} {total:<10.2f}")
                
        except Exception as e:
            print(f"Error generating preprocessing summary: {e}")
    
    if os.path.exists(allreduce_file):
        try:
            allreduce_df = pd.read_csv(allreduce_file)
            if 'AllReduce' in allreduce_df.columns and 'Size(bytes)' in allreduce_df.columns:
                allreduce_timing = allreduce_df.groupby('AllReduce').agg({
                    'Duration(ms)': ['mean', 'sum'],
                    'Size(bytes)': 'mean'
                }).sort_values(('Duration(ms)', 'mean'), ascending=False)
                
                print("\nAllReduce Operations Summary (sorted by average duration):")
                print(f"{'Operation':<25} {'Avg Size (KB)':<15} {'Avg (ms)':<10} {'Total (ms)':<10}")
                print("-" * 65)
                
                for op, row in allreduce_timing.iterrows():
                    avg_size_kb = row[('Size(bytes)', 'mean')] / 1024
                    avg_duration = row[('Duration(ms)', 'mean')]
                    total_duration = row[('Duration(ms)', 'sum')]
                    print(f"{op:<25} {avg_size_kb:<15.2f} {avg_duration:<10.2f} {total_duration:<10.2f}")
                    
        except Exception as e:
            print(f"Error generating AllReduce summary: {e}")
    
    # Return success
    return True


def main():
    """Main function to parse arguments and run the visualization."""
    parser = argparse.ArgumentParser(description='Visualize training profiling data')
    parser.add_argument('profile_dir', type=str, help='Directory containing profiling logs')
    parser.add_argument('--save_format', type=str, default='png', choices=['png', 'pdf', 'svg'], 
                        help='Format to save the plots (default: png)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.profile_dir):
        print(f"Error: Profile directory '{args.profile_dir}' does not exist")
        return 1
    
    print(f"Visualizing profiling data from: {args.profile_dir}")
    success = visualize_profiling_data(args.profile_dir)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
