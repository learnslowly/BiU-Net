
import matplotlib.pyplot as plt
import re
import seaborn as sns
import os
from typing import Optional
from pdb import set_trace
from config.modelconfig import ModelConfig
from data.metrics import calculate_maf, calculate_accuracy, calculate_r2, calculate_iqs
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import multiprocessing
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib


# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_args():
    parser = argparse.ArgumentParser(description="Testing SCDA on Genotype Data Imputation")
    parser.add_argument("--configFile", type=str, required=True, help="Path to the config file")
    parser.add_argument("--randState", type=int, default=42, help="Random state")
    parser.add_argument("--impMethod", type=str, default=None, required=False, help="Imputation method")
    # Add new argument for missing level index
    parser.add_argument("--missingLevelIdx", type=int, required=True, help="Index of the missing level to process")
    args = parser.parse_args()
    return args

def proportion_variants_in_masked_positions(df_orig, mask):
    masked_values_orig = df_orig.values[mask]
    non_ones_count = np.count_nonzero(masked_values_orig != 1)
    total_masked = np.count_nonzero(mask)
    return non_ones_count, total_masked

def format_maf_bin_label(lower, upper):
    if upper == 1.0:
        return ">=50%"
    lower_pct = lower * 100
    upper_pct = upper * 100
    def format_pct(value):
        if value < 1:
            return f"{value:.3f}".rstrip('0').rstrip('.')
        else:
            return f"{value:.1f}".rstrip('0').rstrip('.')
    return f"{format_pct(lower_pct)}%~{format_pct(upper_pct)}%"

def benchmark(config: ModelConfig, missing_index: int, rand_state: int, imp_method: Optional[str] = None, overlapped_only: bool = False):
    orig = pd.read_csv(config.test_csv_gz, compression='gzip', index_col=0)
    orig.columns = orig.columns.astype(int)
    masked = pd.read_csv(config.masked_csv_gzs(rand_state)[missing_index], compression='gzip', index_col=0)
    masked.columns = masked.columns.astype(int)
    total_snps = orig.shape[0]
    mask = (masked == config.missingId)
    num_masked_genotypes = mask.sum().sum()
    masked_percentage = (num_masked_genotypes / (total_snps * orig.shape[1]))
    variants_in_masked, total_masked = proportion_variants_in_masked_positions(orig, mask)
    variants_percentage = (variants_in_masked / total_masked) if total_masked > 0 else 0
    print("="*100)
    print(f"Total snps/samples: {total_snps}/{orig.shape[1]}, masked genotypes: {num_masked_genotypes} ({masked_percentage:.2%}), variants in masked genotypes {variants_in_masked} ({variants_percentage:.2%})")

    # Variables to store MAF-related data for later use with result_by_bin
    ignored_maf_counts = None
    total_bin_counts = None
    # Calculate MAF from training set
    train_df = pd.read_csv(config.test_csv_gz.replace('test', 'train'), compression='gzip', index_col=0)
    train_df.columns = train_df.columns.astype(int)
    maf = calculate_maf(train_df)
    bin_labels = [format_maf_bin_label(config.bins[i], config.bins[i + 1]) for i in range(len(config.bins) - 1)]
    maf_bins = pd.cut(maf, bins=config.bins,
                           labels=bin_labels,
                           include_lowest=True, right=True)
    overlapped_loci_file = os.path.join(
        config.analysisDir,
        f"Beagle_overlapped_loci_rand{rand_state}_{config.dataset}_chr{config.chromosome}_missing{config.missing_percent_strs[missing_index]}.txt"
    )

    if imp_method is not None:
        imputed_csv = f"../{imp_method}/impute/{rand_state}/{config.missing_percent_strs[missing_index]}/{config.dataset}_chr{config.chromosome}_{config.population}_missing{config.missing_percent_strs[missing_index]}_imputed.csv.gz"
        imputed = pd.read_csv(imputed_csv, compression='gzip', index_col=0)
        # Find ignored loci (SNPs present in original but not in Beagle output)
        imputed_set = set(imputed.index.tolist())
        overlapped_loci = orig[orig.index.isin(imputed_set)].index.tolist()
        ignored_loci = orig[~orig.index.isin(imputed_set)].index.tolist()
        num_overlapped = len(overlapped_loci)
        num_ignored = len(ignored_loci)

        if num_ignored > 0:
            print(f"#Ignored loci / #Reserved loci: {num_ignored} / {num_overlapped})")

            # Analyze MAF distribution of ignored loci
            #set_trace()
            ignored_maf_counts = maf_bins[ignored_loci].value_counts().sort_index()
            total_bin_counts = maf_bins.value_counts().sort_index()

            print("\nMAF Distribution of Ignored Loci:")
            print("-" * 100)
            print(f"{'MAF Bin':<15} {'Ignored Count':<15} {'Total in Bin':<15} {'Proportion':<15}")

            for bin_label in total_bin_counts.index:
                ignored_count = ignored_maf_counts.get(bin_label, 0)
                total_count = total_bin_counts[bin_label]
                proportion = ignored_count / total_count if total_count > 0 else 0
                print(f"{str(bin_label):<15} {str(ignored_count):<15} {str(total_count):<15} {proportion:.2%}")
            print(". " * 50)
            #set_trace()
            # Save overlapped loci with their MAF values
            overlapped_loci_df = pd.DataFrame({
                'locus': overlapped_loci,
                'maf': maf[overlapped_loci].values,
                'maf_bin': maf_bins[overlapped_loci].values
            })
            #overlapped_loci_df.sort_values(['locus'], ascending=[True], inplace=True)

            # Save overlapped loci to file
            overlapped_loci_df.to_csv(overlapped_loci_file, index=False, sep='\t')
            print(f"Overlapped loci list saved to: {overlapped_loci_file}")
            #set_trace()
            # Benchmark only against overlapped loci when Beagle/Minimac4 exclude some SNPs
            orig = orig.loc[overlapped_loci]
            masked = masked.loc[overlapped_loci]
            imputed = imputed.loc[overlapped_loci]
            maf = maf.loc[overlapped_loci]
            maf_bins = maf_bins.loc[overlapped_loci]
        else:
            if imputed.shape[0] > num_overlapped:
                print(f"Imputed dataset contains {imputed.shape[0]-num_overlapped} extra SNPs, will be excluded when benchmarking.")
                imputed = imputed.loc[(overlapped_loci)]

    else: # Reference-free method
        imputed = pd.read_csv(config.imputed_csv_gzs(rand_state)[missing_index], compression='gzip', index_col=0)
        num_ignored = 0
        ignored_loci = set()
        if os.path.exists(overlapped_loci_file) and overlapped_only:
            print("This ref-free model will be benchmarked against only overlapped SNPs.")
            overlapped_loci = pd.read_csv(overlapped_loci_file, sep='\t')['locus'].tolist()
            orig = orig.loc[overlapped_loci]
            masked = masked.loc[overlapped_loci]
            imputed = imputed.loc[overlapped_loci]
            maf = maf.loc[overlapped_loci]
            maf_bins = maf_bins.loc[overlapped_loci]

    imputed.columns = imputed.columns.astype(int)
    grouped_indices = {label: maf_bins[maf_bins == label].index.tolist() for label in bin_labels}
    result_by_bin = []
    # Recalculate the mask again since masked might be modified when the loci of imputed and orig don't match
    mask = (masked == config.missingId)

    #set_trace()
    for bin_range in grouped_indices.keys():
        orig_bin = orig.loc[grouped_indices[bin_range]].values.ravel(order="C")
        imputed_bin = imputed.loc[grouped_indices[bin_range]].values.ravel(order="C")

        if not config.benchmarkAll:

            mask_bin = mask.loc[grouped_indices[bin_range]].values.ravel(order="C")
            orig_bin = orig_bin[mask_bin]
            imputed_bin = imputed_bin[mask_bin]

        bin_acc = calculate_accuracy(orig_bin, imputed_bin)

        bin_r2 = calculate_r2(orig_bin, imputed_bin)
        bin_iqs = calculate_iqs(orig_bin, imputed_bin)
        bin_precision = precision_score(orig_bin, imputed_bin, average='macro', zero_division=0)
        bin_recall = recall_score(orig_bin, imputed_bin, average='macro', zero_division=0)
        bin_f1 = f1_score(orig_bin, imputed_bin, average='macro', zero_division=0)

        result = {
            'MAF_bin': bin_range,
            'Num_SNPs': len(grouped_indices[bin_range]),
            'Bin_Acc': bin_acc,
            'Bin_R2': bin_r2,
            'Bin_IQS': bin_iqs,
            'Bin_Precision': bin_precision,
            'Bin_Recall': bin_recall,
            'Bin_F1': bin_f1
        }
        result_by_bin.append(result)

    # Display result by MAF bins
    result_by_bin = pd.DataFrame(result_by_bin)
    print('-'*100)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(result_by_bin.to_string(index=False))
    print('. '*50)

    # Calculate overall result
    orig_all = orig.values.ravel(order="C")
    imputed_all = imputed.values.ravel(order="C")

    if not config.benchmarkAll:
        mask_snps = mask.values.ravel(order="C")
        orig_all = orig_all[mask_snps]
        imputed_all = imputed_all[mask_snps]

    all_acc = calculate_accuracy(orig_all, imputed_all)
    all_r2 = calculate_r2(orig_all, imputed_all)
    all_iqs = calculate_iqs(orig_all, imputed_all)
    all_precision = precision_score(orig_all, imputed_all, average='macro', zero_division=0)
    all_recall = recall_score(orig_all, imputed_all, average='macro', zero_division=0)
    all_f1 = f1_score(orig_all, imputed_all, average='macro', zero_division=0)

    overall_result = pd.DataFrame({
        'Total_SNPs': [orig.shape[0]],
        'Overall_Acc': [all_acc],
        'Overall_R2': [all_r2],
        'Overall_IQS': [all_iqs],
        'Overall_Precision': [all_precision],
        'Overall_Recall': [all_recall],
        'Overall_F1': [all_f1]
    })

    print(f"Overall Average Metrics / Testing on {config.missing[missing_index]} Missing / {config.dataset}: Chromosome {config.chromosome}")
    print(overall_result.to_string(index=False))
    print("+"*100)

    # Broadcast overall results to result by bin
    for col in overall_result.columns[1:]:
        result_by_bin[col] = overall_result[col].iloc[0]

    # Save to CSV
    os.makedirs(config.analysisDir, exist_ok=True)
    result_csv = os.path.join(
        config.analysisDir,
        f"{config.runId}_rand{rand_state}_{config.dataset}_chr{config.chromosome}_{config.population}_seg{config.segLen}_overlap{config.overlap}_missing{config.missing_percent_strs[missing_index]}.csv"
    )

    result_by_bin.to_csv(result_csv, index=False, sep=',')
    print(f"Results saved to {result_csv}")

    # Compute confusion matrix
    labels = [0, 1, 2, 3]
    conf_mat = confusion_matrix(orig_all-1, imputed_all-1, labels=labels)

    # Plot confusion matrix
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'axes.grid': True,
        'grid.linestyle': '-',
        'grid.alpha': 0.5,
        'grid.color': 'gray',
        'grid.linewidth': 0.5,
    })
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted Genotype')
    plt.ylabel('True Genotype')
    plt.title('Confusion Matrix')

    confusion_matrix_plot_path = result_csv.replace(".csv", "_confusion_matrix.pdf")
    os.makedirs(os.path.dirname(confusion_matrix_plot_path), exist_ok=True)
    plt.savefig(confusion_matrix_plot_path)
    plt.close()
    print(f"Confusion matrix plot saved at: {confusion_matrix_plot_path}")

def main(config: ModelConfig, random_state: int, missing_level_idx: int, imp_method: Optional[str] = None):
    # Process just one specific missing level instead of looping
    benchmark(config, missing_level_idx, random_state, imp_method, overlapped_only=True)

if __name__ == "__main__":

    args = parse_args()
    config = ModelConfig.from_yaml(args.configFile)

    print(f"Testing the imputation results from: {args.impMethod}" if args.impMethod is not None else "Testing the imputation results")
    hostname = os.getenv('HOSTNAME', 'unknown')
    print(f"Running on host {hostname} with random state {args.randState}")
    print(f"Using checkpoint: {config.checkpoint}")
    print(f"Processing missing level: {config.missing_percent_strs[args.missingLevelIdx]}")

    # Pass the missing level index to the main function
    main(config, random_state=args.randState, imp_method=args.impMethod, missing_level_idx=args.missingLevelIdx)
