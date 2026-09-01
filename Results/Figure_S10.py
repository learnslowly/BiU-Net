#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 2, reproduced in the supplement as Figure S10.

    python Results/Figure_S10.py

Writes Figures/Figure_2.pdf and Figures/Figure_S10_identical_to_Figure_2.pdf,
and prints each panel's statistics so a rebuild can be compared with the
published figure.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config.modelconfig import ModelConfig  # noqa: E402

# Set visual style
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.2)

# Embed text as TrueType rather than matplotlib.s default Type 3.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

CONFIGS = {
    "AA": "configs/test_cohort_LOS_chr22_AA.yaml",
    "CA": "configs/test_cohort_LOS_chr22_CA.yaml",
    "ALL": "configs/test_cohort_LOS_chr22_ALL.yaml",
}
COHORTS = ["AA", "CA", "ALL"]
RANDOM_STATES = [0, 42, 1024]
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for 5%, 15%, 25%
OUTPUTS = ["Figures/Figure_2.pdf",
           "Figures/Figure_S10_identical_to_Figure_2.pdf"]


def resolve(path):
    """Config paths are written relative to the project root."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_configs():
    cfgs = {c: ModelConfig.from_yaml(resolve(p)) for c, p in CONFIGS.items()}
    rates = cfgs["ALL"].missing_percent_strs
    for cohort, cfg in cfgs.items():
        if cfg.missing_percent_strs != rates:
            raise SystemExit(
                f"{CONFIGS[cohort]} masks at {cfg.missing_percent_strs}, "
                f"but the admixed model was scored at {rates}; the panels would "
                "not compare like with like")
    return cfgs, rates


def calculate_all_sample_r2(test_df, imputed_df):
    """Calculate R² for all samples in one go"""
    r2_values = {}

    # Both dataframes have the same samples in the same order
    for sample in test_df.columns:
        # Get true and predicted values
        true_values = test_df[sample].values
        pred_values = imputed_df[sample].values

        # Remove NaN values
        mask = ~np.isnan(true_values) & ~np.isnan(pred_values)

        # Calculate R² if enough data points
        if np.sum(mask) > 10:
            r2 = r2_score(true_values[mask], pred_values[mask])
            r2_values[sample] = r2

    return r2_values


def main():
    print("Starting imputation performance analysis...")
    cfgs, missing_levels = load_configs()

    # Dictionary to store average R² values
    avg_r2 = {cohort: {level: {} for level in missing_levels} for cohort in COHORTS}

    # Calculate R² values for each cohort/level combination. One cohort's test
    # split and one imputed matrix are in memory at a time: the admixed split is
    # 754 samples by 257k markers, and holding all three at once needs several
    # gigabytes more than the comparison itself does.
    for cohort in COHORTS:
        cfg = cfgs[cohort]
        test_path = resolve(cfg.test_csv_gz)
        test_df = pd.read_csv(test_path, compression='gzip', index_col=0)
        print(f"Loaded test data for {cohort}: {test_df.shape} "
              f"from {os.path.relpath(test_path, ROOT)}")
        for idx, level in enumerate(missing_levels):
            # Dictionary to accumulate R² values across random states
            sample_r2_sums = {}
            sample_r2_counts = {}

            # Process each random state
            for rs in RANDOM_STATES:
                # Load imputed data
                imputed_path = resolve(cfg.imputed_csv_gzs(rs)[idx])
                imputed_df = pd.read_csv(imputed_path, compression='gzip', index_col=0)

                # Calculate R² for all samples
                r2_values = calculate_all_sample_r2(test_df, imputed_df)
                del imputed_df

                # Accumulate R² values
                for sample, r2 in r2_values.items():
                    if sample not in sample_r2_sums:
                        sample_r2_sums[sample] = 0
                        sample_r2_counts[sample] = 0
                    sample_r2_sums[sample] += r2
                    sample_r2_counts[sample] += 1

            # Calculate averages
            for sample in sample_r2_sums:
                if sample_r2_counts[sample] == len(RANDOM_STATES):
                    avg_r2[cohort][level][sample] = sample_r2_sums[sample] / sample_r2_counts[sample]

            print(f"Calculated average R² for {cohort}, level {level}: {len(avg_r2[cohort][level])} samples")
        del test_df

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=False, sharey=False)
    fig.suptitle('Imputation Performance: Cohort-Specific vs. Admixed Population (ALL)', fontsize=20)

    print("\ncohort  missing   ALL R²   cohort R²        delta      N   better")
    # Plot each comparison
    for row, cohort in enumerate(["AA", "CA"]):
        for col, level in enumerate(missing_levels):
            ax = axes[row, col]

            # Find common samples
            common_samples = set(avg_r2[cohort][level].keys()) & set(avg_r2["ALL"][level].keys())
            sorted_samples = sorted(common_samples)

            # Get R² values for plotting
            cohort_r2_values = [avg_r2[cohort][level][s] for s in sorted_samples]
            all_r2_values = [avg_r2["ALL"][level][s] for s in sorted_samples]

            # Calculate subplot-specific range with padding
            if cohort_r2_values and all_r2_values:
                min_x = max(0, min(all_r2_values) - 0.05)
                max_x = min(1, max(all_r2_values) + 0.05)
                min_y = max(0, min(cohort_r2_values) - 0.05)
                max_y = min(1, max(cohort_r2_values) + 0.05)

                # Expand range a bit if too narrow
                if max_x - min_x < 0.1:
                    pad = (0.1 - (max_x - min_x)) / 2
                    min_x = max(0, min_x - pad)
                    max_x = min(1, max_x + pad)
                if max_y - min_y < 0.1:
                    pad = (0.1 - (max_y - min_y)) / 2
                    min_y = max(0, min_y - pad)
                    max_y = min(1, max_y + pad)

                # Make sure both axes have the same limits for fair comparison
                min_val = min(min_x, min_y)
                max_val = max(max_x, max_y)
            else:
                min_val, max_val = 0, 1

            # Plot scatter points
            ax.scatter(all_r2_values, cohort_r2_values, alpha=0.6, s=30, color=COLORS[col], edgecolor='k', linewidth=0.5)

            # Plot y=x line
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

            # Calculate statistics
            cohort_mean = np.mean(cohort_r2_values)
            all_mean = np.mean(all_r2_values)
            diff = cohort_mean - all_mean
            above_diagonal = sum(1 for i in range(len(all_r2_values)) if cohort_r2_values[i] > all_r2_values[i])
            above_pct = above_diagonal / len(all_r2_values) * 100
            print(f"{cohort:>6}  {level:>7}   {all_mean:.4f}     {cohort_mean:.4f}  "
                  f"{diff:+.4f}  {len(common_samples):>5}  {above_pct:5.1f}%")

            # Add text with statistics
            ax.text(0.05, 0.95,
                    f"ALL R²: {all_mean:.3f}\n{cohort} R²: {cohort_mean:.3f}\nΔ: {diff:.3f}\n"
                    f"N={len(common_samples)}\n"
                    f"Better: {above_pct:.1f}%",
                    transform=ax.transAxes, fontsize=16, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Set title and limits
            ax.set_title(f"{cohort} vs ALL (Missing: {level})")
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)

            # Add labels
            if col == 0:
                ax.set_ylabel(f"{cohort} R²")
            if row == 1:
                ax.set_xlabel("ALL R²")

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for out in OUTPUTS:
        path = os.path.join(ROOT, out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, format='pdf', bbox_inches='tight')
        print(f"\nVisualization saved as '{out}'")


if __name__ == "__main__":
    main()
