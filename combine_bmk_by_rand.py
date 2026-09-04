#!/usr/bin/env python3
"""
Script to combine benchmark results CSV files from multiple missing levels.
Run this script after all your benchmark jobs have completed.

Usage:
    python combine_benchmark_results.py --configFile CONFIG_FILE [--randStates RAND_STATES]

Example:
    python combine_benchmark_results.py --configFile archive/exp1/1KGP_chr22_ALL_seg128_overlap16/LONI_Feb25_2304.yaml --randStates 0 42 1024
"""

import os
import glob
import re
import pandas as pd
import argparse
from config.modelconfig import ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Combine benchmark CSV results")
    parser.add_argument("--configFile", type=str, required=True, help="Path to the config file")
    parser.add_argument("--randStates", type=int, nargs='+', default=[0, 42, 1024], 
                        help="Random states to process (default: 0 42 1024)")
    return parser.parse_args()


def combine_csv_files(config, combined_filename, rand_state):
    """
    Combines CSV files for a specific random state into a single CSV file.
    
    Args:
        config: ModelConfig object
        combined_filename: Path to save the combined CSV
        rand_state: Random state value
    """
    # Identify all the CSV files matching the pattern
    pattern = f"{config.analysisDir}/{config.runId}_rand{rand_state}_*.csv"
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No CSV files found for random state {rand_state} with pattern: {pattern}")
        return False
    
    # Sort by missing percentage for consistent ordering
    csv_files_sorted = sorted(csv_files, key=lambda x: float(re.search(r'.*?missing(\d+)%.*?', x).group(1)))
    
    print(f"Found {len(csv_files_sorted)} CSV files for random state {rand_state}:")
    for csv_file in csv_files_sorted:
        print(f"  - {os.path.basename(csv_file)}")
    
    # Combine all CSV files
    dfs = []
    for csv_file in csv_files_sorted:
        try:
            df = pd.read_csv(csv_file)
            # Add missing level information if not already present
            missing_pct = re.search(r'.*?missing(\d+)%.*?', csv_file).group(1)
            if 'Missing_Level' not in df.columns:
                df['Missing_Level'] = f"{missing_pct}%"
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if not dfs:
        print(f"No valid CSV files could be read for random state {rand_state}")
        return False
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save combined dataframe
    combined_df.to_csv(combined_filename, index=False)
    print(f"Combined CSV saved to {combined_filename}")
    return True


def main():
    args = parse_args()
    config = ModelConfig.from_yaml(args.configFile)
    print(f"Processing benchmark results for config: {args.configFile}")
    
    success_count = 0
    for rand_state in args.randStates:
        output_file = os.path.join(config.analysisDir, f"{config.runId}_{rand_state}.csv")
        print(f"\nProcessing random state {rand_state}...")
        if combine_csv_files(config, output_file, rand_state):
            success_count += 1
    
    print(f"\nSummary: Successfully combined {success_count}/{len(args.randStates)} random states.")


if __name__ == "__main__":
    main()
