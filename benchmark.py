import matplotlib.pyplot as plt
import re
import seaborn as sns
import os
from typing import Optional
from config.modelconfig import ModelConfig
from data.metrics import (
    calculate_maf, calculate_accuracy, calculate_r2, calculate_r2_per_variant,
)
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import multiprocessing
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib


def analysis_basename(config: ModelConfig, kind: str, method_tag: str, flags: str,
                      rand_state: int, miss_str: str) -> str:
    """Filename for one benchmark cell written under analysis/.

    Our own model's results carry the checkpoint epoch, the way test.py has
    always tagged its impute/ outputs: scoring two checkpoints of one run then
    lands in two files instead of the later silently overwriting the earlier.
    External imputers (Beagle, Minimac) have no epoch, so their cells keep the
    plain name. Readers that predate this tag are handled by
    scripts/analysis_cells.py, which resolves either spelling.
    """
    epoch_tag = "" if method_tag not in ("biunet",) else f"_epoch{config.epoch}"
    return (f"{config.runId}{epoch_tag}_{kind}_{method_tag}_{flags}"
            f"_rand{rand_state}_{config.dataset}_chr{config.chromosome}"
            f"_{config.population}_seg{config.segLen}_overlap{config.overlap}"
            f"_missing{miss_str}.csv")


def parse_args():
    """Which config to run, whose output to read, and whether to score or report.

    The config describes one evaluation: its paths, its checkpoint, its missing
    rates, its random states and its marker set. What stays on the command line
    is what selects between runs of that same evaluation rather than describing
    it: whose imputed output to read, and whether to score cells or lay the
    already-scored cells out as a table.
    """
    parser = argparse.ArgumentParser(description="Score imputed genotypes against the truth.")
    parser.add_argument("--configFile", type=str, required=True,
                        help="Path to the evaluation config YAML")
    parser.add_argument("--impMethod", type=str, default=None,
                        help="Imputer whose output to score ('beagle', 'scda'); "
                             "omit to score this model's own output")
    parser.add_argument("--report", action="store_true",
                        help="Aggregate the cells already written into a comparison "
                             "table instead of scoring")
    parser.add_argument("--reportScda", type=str, default=None,
                        help="Optional SCDA eval config; adds an SCDA column in --report mode")
    return parser.parse_args()


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

VALID_GENOTYPES = (1, 2, 3, 4)


def drop_corrupt_loci(orig, masked, imputed, maf, maf_bins):
    """Remove markers whose predictions hold values outside the genotype codes.

    A handful of out-of-range entries leaves accuracy untouched, because they are
    a vanishing fraction of a matrix that is mostly homozygous reference, while
    driving R2 to zero, because the variance of the predictions is then set by an
    outlier of order 1e11 rather than by the genotypes. Scoring such a cell
    produces a number that looks like a model collapse and is not one, so the
    affected markers are dropped and counted instead.
    """
    values = imputed.to_numpy()
    bad_cells = ~np.isin(values, VALID_GENOTYPES)
    n_bad = int(bad_cells.sum())
    if n_bad == 0:
        return orig, masked, imputed, maf, maf_bins, 0, 0
    bad_row_mask = bad_cells.any(axis=1)
    bad_loci = set(imputed.index[bad_row_mask])
    bad_values = values[bad_cells]
    # NaN marks a marker the method declined; an out-of-range value marks a
    # wrong call. The two are counted apart.
    n_missing = int(pd.isna(bad_values).sum())
    finite = bad_values[~pd.isna(bad_values)]
    sample = sorted({int(v) for v in finite[:8]})
    print(f"WARNING: {n_bad} prediction(s) outside {VALID_GENOTYPES} on "
          f"{len(bad_loci)} marker(s), of which {n_missing} are absent "
          f"(no call); those markers are excluded from scoring. "
          f"Example out-of-range values: {sample}", flush=True)
    # Index each frame by its own index: maf comes from the training split and
    # orig from the test split, and their lengths differ where a duplicate was dropped.
    return (orig[~orig.index.isin(bad_loci)], masked[~masked.index.isin(bad_loci)],
            imputed[~imputed.index.isin(bad_loci)], maf[~maf.index.isin(bad_loci)],
            maf_bins[~maf_bins.index.isin(bad_loci)], n_bad, len(bad_loci))


def _maf_from_source(train_path):
    """Parse the training split and reduce it to a per-marker MAF series."""
    train_df = pd.read_csv(train_path, compression='gzip', index_col=0)
    train_df.columns = train_df.columns.astype(int)
    # Keep the first of any duplicated position in the training CSV.
    if train_df.index.duplicated().any():
        n_dup = int(train_df.index.duplicated().sum())
        print(f"WARNING: {n_dup} duplicated position(s) in train CSV; keeping first occurrence for MAF")
        train_df = train_df[~train_df.index.duplicated(keep='first')]
    return calculate_maf(train_df)


def cached_maf(config):
    """Per-marker MAF from the training split, computed once per chromosome.

    MAF depends only on the training genotypes, not on the masking seed or the
    missing rate, yet every benchmark cell recomputed it. For 1KGP chromosome 1
    that meant parsing a 189 MB gzip CSV nine times per chromosome, which
    dominated the wall time of the scoring phase. The cache is keyed on the
    source file's size and modification time, so an updated split invalidates it
    instead of being silently reused.

    Every cache operation falls back to reading the source, so a cache that is
    unwritable, truncated, or holds an index this cannot round-trip costs time
    and never correctness.
    """
    train_path = config.test_csv_gz.replace('test', 'train')
    cache_path = None
    try:
        st = os.stat(train_path)
        cache_dir = os.path.join(config.analysisDir, '.maf_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir,
            f"{os.path.basename(train_path)}_{st.st_size}_{int(st.st_mtime)}.npz")
        if os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=False)
            maf = pd.Series(z['values'], index=z['index'])
            print(f"MAF read from cache ({len(maf)} markers)")
            return maf
    except Exception as exc:
        print(f"MAF cache unavailable ({exc}); reading the training split")
        cache_path = None

    maf = _maf_from_source(train_path)

    if cache_path is not None and np.issubdtype(maf.index.dtype, np.number):
        try:
            # np.savez appends .npz, so the temporary name has to end in it already.
            tmp = f"{cache_path}.{os.getpid()}.tmp.npz"
            np.savez(tmp, index=maf.index.to_numpy(), values=maf.to_numpy())
            os.replace(tmp, cache_path)
            print(f"MAF cached to {cache_path}")
        except Exception as exc:
            print(f"MAF cache not written ({exc}); continuing")
    return maf


def benchmark(config: ModelConfig, missing_index: int, rand_state: int, imp_method: Optional[str] = None,
              overlapped_only: bool = False, excluded_only: bool = False):
    orig = pd.read_csv(config.test_csv_gz, compression='gzip', index_col=0)
    orig.columns = orig.columns.astype(int)
    masked_path = config.masked_csv_gzs(rand_state)[missing_index]
    total_snps = orig.shape[0]
    print("="*100)
    if os.path.exists(masked_path):
        masked = pd.read_csv(masked_path, compression='gzip', index_col=0)
        masked.columns = masked.columns.astype(int)
        mask = (masked == config.missingId)
        num_masked_genotypes = mask.sum().sum()
        masked_percentage = (num_masked_genotypes / (total_snps * orig.shape[1]))
        variants_in_masked, total_masked = proportion_variants_in_masked_positions(orig, mask)
        variants_percentage = (variants_in_masked / total_masked) if total_masked > 0 else 0
        print(f"Total snps/samples: {total_snps}/{orig.shape[1]}, masked genotypes: {num_masked_genotypes} ({masked_percentage:.2%}), variants in masked genotypes {variants_in_masked} ({variants_percentage:.2%})")
    elif config.benchmarkAll:
        # With benchmarkAll the truth stands in for the masked matrix, keeping the
        # frames index-aligned.
        masked = orig
        print(f"Total snps/samples: {total_snps}/{orig.shape[1]}; {masked_path} is absent, "
              "so the masked-position summary is omitted")
    else:
        raise SystemExit(f"{masked_path} is needed to score masked positions only; "
                         "set benchmarkAll to score every genotype instead")

    # Variables to store MAF-related data for later use with result_by_bin
    ignored_maf_counts = None
    total_bin_counts = None
    # Calculate MAF from training set
    maf = cached_maf(config)
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
        if excluded_only:
            # The complement of the retained set: markers with no reference haplotype,
            # for which the copying model produced no call.
            if not os.path.exists(overlapped_loci_file):
                raise SystemExit(f"--excludedOnly needs {overlapped_loci_file}, which is written "
                                 f"when the reference-based imputer is scored for this cell")
            kept = set(pd.read_csv(overlapped_loci_file, sep='\t')['locus'].tolist())
            n_excluded = int((~orig.index.isin(kept)).sum())
            print(f"Scoring the {n_excluded} marker(s) the reference-based imputer excluded "
                  f"({n_excluded / len(orig.index):.2%} of the target)")
            if n_excluded == 0:
                raise SystemExit("no excluded markers for this cell; nothing to score")
            # Index each frame by its own index; the two differ in length wherever a
            # duplicated position was removed.
            orig = orig[~orig.index.isin(kept)]
            masked = masked[~masked.index.isin(kept)]
            imputed = imputed[~imputed.index.isin(kept)]
            maf = maf[~maf.index.isin(kept)]
            maf_bins = maf_bins[~maf_bins.index.isin(kept)]
        elif os.path.exists(overlapped_loci_file) and overlapped_only:
            print("This ref-free model will be benchmarked against only overlapped SNPs.")
            overlapped_loci = pd.read_csv(overlapped_loci_file, sep='\t')['locus'].tolist()
            orig = orig.loc[overlapped_loci]
            masked = masked.loc[overlapped_loci]
            imputed = imputed.loc[overlapped_loci]
            maf = maf.loc[overlapped_loci]
            maf_bins = maf_bins.loc[overlapped_loci]

    imputed.columns = imputed.columns.astype(int)
    orig, masked, imputed, maf, maf_bins, _, _ = drop_corrupt_loci(
        orig, masked, imputed, maf, maf_bins)
    grouped_indices = {label: maf_bins[maf_bins == label].index.tolist() for label in bin_labels}
    result_by_bin = []
    # Recomputed here because dropping corrupt loci above rebuilds `masked`.
    mask = (masked == config.missingId)

    for bin_range in grouped_indices.keys():
        loci = grouped_indices[bin_range]
        orig_2d = orig.loc[loci].values
        imputed_2d = imputed.loc[loci].values
        mask_2d = mask.loc[loci].values if not config.benchmarkAll else None

        if config.perVariantR2:
            bin_r2 = calculate_r2_per_variant(
                orig_2d, imputed_2d, mask_2d=mask_2d,
                variants_only=config.benchmarkVariantsOnly,
            )
            # For per-variant mode, also compute acc/precision/recall/f1 on the pooled
            # filtered samples (so summary stats still reflect the same scope).
            orig_bin = orig_2d.ravel(order="C")
            imputed_bin = imputed_2d.ravel(order="C")
            if mask_2d is not None:
                m1d = mask_2d.ravel(order="C")
                orig_bin = orig_bin[m1d]
                imputed_bin = imputed_bin[m1d]
            if config.benchmarkVariantsOnly:
                vm = (orig_bin != 1)
                orig_bin = orig_bin[vm]
                imputed_bin = imputed_bin[vm]
        else:
            orig_bin = orig_2d.ravel(order="C")
            imputed_bin = imputed_2d.ravel(order="C")
            if mask_2d is not None:
                m1d = mask_2d.ravel(order="C")
                orig_bin = orig_bin[m1d]
                imputed_bin = imputed_bin[m1d]
            # Variants-only filter: drop positions where TRUTH is homo-ref (class 1 = 0|0).
            # Phased vocab: 1=0|0, 2=0|1, 3=1|0, 4=1|1.
            if config.benchmarkVariantsOnly:
                variant_mask = (orig_bin != 1)
                orig_bin = orig_bin[variant_mask]
                imputed_bin = imputed_bin[variant_mask]
            bin_r2 = calculate_r2(orig_bin, imputed_bin)

        bin_acc = calculate_accuracy(orig_bin, imputed_bin)
        bin_precision = precision_score(orig_bin, imputed_bin, average='macro', zero_division=0)
        bin_recall = recall_score(orig_bin, imputed_bin, average='macro', zero_division=0)
        bin_f1 = f1_score(orig_bin, imputed_bin, average='macro', zero_division=0)

        result = {
            'MAF_bin': bin_range,
            'Num_SNPs': len(grouped_indices[bin_range]),
            # Full bin count against the number the reference-based method dropped.
            # Reference-free runs fall back to the kept count and zero.
            'Num_SNPs_total': int(total_bin_counts.get(bin_range, 0)) if total_bin_counts is not None else len(grouped_indices[bin_range]),
            'Num_SNPs_removed': int(ignored_maf_counts.get(bin_range, 0)) if ignored_maf_counts is not None else 0,
            'Bin_Acc': bin_acc,
            'Bin_R2': bin_r2,
            'Bin_Precision': bin_precision,
            'Bin_Recall': bin_recall,
            'Bin_F1': bin_f1
        }
        result_by_bin.append(result)

    # Display result by MAF bins
    result_by_bin = pd.DataFrame(result_by_bin)
    # Add a leading space to all MAF_bin values so pasting into excel will not be a pain
    result_by_bin['MAF_bin'] = result_by_bin['MAF_bin'].apply(lambda x: ' ' + str(x))
    print('-'*100)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # Calculate overall result
    if config.perVariantR2:
        # Per-variant: r² averaged over ALL variants (across all bins)
        orig_2d_all = orig.values
        imputed_2d_all = imputed.values
        mask_2d_all = mask.values if not config.benchmarkAll else None
        all_r2 = calculate_r2_per_variant(
            orig_2d_all, imputed_2d_all, mask_2d=mask_2d_all,
            variants_only=config.benchmarkVariantsOnly,
        )
        # Other summary stats use the same pooled filtered samples
        orig_all = orig_2d_all.ravel(order="C")
        imputed_all = imputed_2d_all.ravel(order="C")
        if mask_2d_all is not None:
            m1d = mask_2d_all.ravel(order="C")
            orig_all = orig_all[m1d]
            imputed_all = imputed_all[m1d]
        if config.benchmarkVariantsOnly:
            vm = (orig_all != 1)
            orig_all = orig_all[vm]
            imputed_all = imputed_all[vm]
    else:
        orig_all = orig.values.ravel(order="C")
        imputed_all = imputed.values.ravel(order="C")
        if not config.benchmarkAll:
            mask_snps = mask.values.ravel(order="C")
            orig_all = orig_all[mask_snps]
            imputed_all = imputed_all[mask_snps]
        if config.benchmarkVariantsOnly:
            variant_mask_all = (orig_all != 1)
            orig_all = orig_all[variant_mask_all]
            imputed_all = imputed_all[variant_mask_all]
        all_r2 = calculate_r2(orig_all, imputed_all)

    all_acc = calculate_accuracy(orig_all, imputed_all)
    all_precision = precision_score(orig_all, imputed_all, average='macro', zero_division=0)
    all_recall = recall_score(orig_all, imputed_all, average='macro', zero_division=0)
    all_f1 = f1_score(orig_all, imputed_all, average='macro', zero_division=0)

    # Create overall result for saving to CSV later
    overall_result = pd.DataFrame({
        'Total_SNPs': [orig.shape[0]],
        'Overall_Acc': [all_acc],
        'Overall_R2': [all_r2],
        'Overall_Precision': [all_precision],
        'Overall_Recall': [all_recall],
        'Overall_F1': [all_f1]
    })

    # Add the overall row to the result_by_bin DataFrame
    overall_row = {
        'MAF_bin': ' Overall',
        'Num_SNPs': orig.shape[0],
        'Num_SNPs_total': int(total_bin_counts.sum()) if total_bin_counts is not None else orig.shape[0],
        'Num_SNPs_removed': int(ignored_maf_counts.sum()) if ignored_maf_counts is not None else 0,
        'Bin_Acc': all_acc,
        'Bin_R2': all_r2,
        'Bin_Precision': all_precision,
        'Bin_Recall': all_recall,
        'Bin_F1': all_f1
    }

    # Add the overall row to the result_by_bin DataFrame
    result_by_bin = pd.concat([result_by_bin, pd.DataFrame([overall_row])], ignore_index=True)
    print(result_by_bin.to_string(index=False))
    print("+"*100)

    # Save to CSV
    os.makedirs(config.analysisDir, exist_ok=True)
    method_tag = imp_method if imp_method is not None else 'biunet'
    ba_tag = 'BAT' if config.benchmarkAll else 'BAF'
    vo_tag = '_VO' if config.benchmarkVariantsOnly else ''
    pv_tag = '_PV' if config.perVariantR2 else ''
    # Only our own model can be scored on the full target axis, so the flag
    # distinguishes those cells from the intersection cells of the same run.
    if excluded_only:
        set_tag = '_EX'
    elif imp_method is None and not overlapped_only:
        set_tag = '_FA'
    else:
        set_tag = ''
    result_csv = os.path.join(
        config.analysisDir,
        analysis_basename(config, "phased", method_tag, f"{ba_tag}{vo_tag}{pv_tag}{set_tag}",
                          rand_state, config.missing_percent_strs[missing_index])
    )

    # Need to keep the overall columns for CSV saving
    for col in overall_result.columns[1:]:
        result_by_bin[col.replace('Overall_', 'Overall ')] = overall_result[col].iloc[0]

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

# Pivot the per-bin CSVs into (metric x method) rows by MAF-bin columns,
# one block per missing rate, averaged over random states.
def _report_csv_path(cfg: ModelConfig, method_tag: str, rand_state: int, missing_str: str) -> str:
    """Locate the cell benchmark() wrote for one (method, rate, state).

    Our own model's cells carry the checkpoint epoch, so a run scored at two
    checkpoints lands in two files rather than one overwriting the other.
    External imputers have no epoch and keep the plain name. Both spellings are
    resolved here, newest checkpoint first, so a report reads whichever exists.
    """
    ba_tag = 'BAT' if cfg.benchmarkAll else 'BAF'
    vo_tag = '_VO' if cfg.benchmarkVariantsOnly else ''
    pv_tag = '_PV' if cfg.perVariantR2 else ''
    # Rebuild the marker-set tag on the rule the scorer used, or the cells are
    # searched for under the wrong name.
    if getattr(cfg, 'excludedOnly', False):
        set_tag = '_EX'
    elif method_tag == 'biunet' and not getattr(cfg, 'overlappedOnly', True):
        set_tag = '_FA'
    else:
        set_tag = ''
    tail = (f"_phased_{method_tag}_{ba_tag}{vo_tag}{pv_tag}{set_tag}_rand{rand_state}"
            f"_{cfg.dataset}_chr{cfg.chromosome}_{cfg.population}"
            f"_seg{cfg.segLen}_overlap{cfg.overlap}_missing{missing_str}.csv")
    tagged = sorted(glob.glob(os.path.join(cfg.analysisDir, f"{cfg.runId}_epoch*{tail}")))
    if tagged:
        def epoch_of(path):
            m = re.search(r"_epoch(\d+|best)_phased_", os.path.basename(path))
            if not m:
                return -1
            return float("inf") if m.group(1) == "best" else int(m.group(1))
        return max(tagged, key=epoch_of)
    return os.path.join(cfg.analysisDir, f"{cfg.runId}{tail}")


def _avg_over_states(cfg: ModelConfig, method_tag: str, missing_str: str, states, col: str) -> dict:
    """Mean of `col` per MAF bin across the available per-state CSVs."""
    acc: dict = {}
    for s in states:
        p = _report_csv_path(cfg, method_tag, s, missing_str)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if col not in df.columns:
            continue
        for _, row in df.iterrows():
            b = str(row['MAF_bin']).strip()
            v = row[col]
            if pd.isna(v):
                continue
            acc.setdefault(b, []).append(float(v))
    return {b: sum(vs) / len(vs) for b, vs in acc.items()}


def report_table(config: ModelConfig, scda_config: Optional[ModelConfig] = None):
    """Tab-separated comparison table per missing rate (paste straight into a doc).

    Rows: #SNPs (total), #SNPs (Beagle/overlapped), MAF-bin header, then
    {Acc, R2, Prc, Rec, F1} x {Beagle[, SCDA], BiU-Net}. Averaged over
    config.testRandStates. Pure aggregation of CSVs benchmark() already wrote.
    """
    states = config.testRandStates or [42]
    # (label, config, method_tag): Beagle & BiU-Net share `config` (same runId/seg);
    # SCDA lives under its own runId/seg, so it brings its own config.
    methods = [("Beagle", config, "beagle")]
    if scda_config is not None:
        methods.append(("SCDA", scda_config, "biunet"))
    methods.append(("BiU-Net", config, "biunet"))
    metrics = [("Acc", "Bin_Acc"), ("R2", "Bin_R2"), ("Prc", "Bin_Precision"),
               ("Rec", "Bin_Recall"), ("F1", "Bin_F1")]

    for m in config.missing:
        mstr = f"{m * 100:.0f}%"
        print(f"\n========== {mstr} Missing ==========")
        # bin order from Beagle's CSV (Overall last); fall back to BiU-Net's
        order = (_avg_over_states(config, "beagle", mstr, states, "Bin_Acc")
                 or _avg_over_states(config, "biunet", mstr, states, "Bin_Acc"))
        bins = [b for b in order if b != "Overall"] + (["Overall"] if "Overall" in order else [])

        def int_row(d):
            return "\t".join(str(int(round(d[b]))) if d.get(b) is not None else "NA" for b in bins)

        def met_row(d):
            return "\t".join(f"{d[b]:.4f}" if d.get(b) is not None else "NA" for b in bins)

        # #SNPs rows from Beagle (total / overlapped-kept)
        print(int_row(_avg_over_states(config, "beagle", mstr, states, "Num_SNPs_total")))
        print(int_row(_avg_over_states(config, "beagle", mstr, states, "Num_SNPs")))
        print("\t".join(bins))
        for _mlabel, mcol in metrics:
            for _label, cfg, tag in methods:
                print(met_row(_avg_over_states(cfg, tag, mstr, states, mcol)))


def main(config: ModelConfig, imp_method: Optional[str] = None):
    """Score every missing rate and random state this evaluation is defined over.

    The rates and the states are the repeats the reported number is averaged
    over, so they belong to the config rather than to the call site.
    """
    overlapped_only = bool(getattr(config, "overlappedOnly", True))
    excluded_only = bool(getattr(config, "excludedOnly", False))
    states = config.testRandStates or [42]
    idx = getattr(config, "missingLevelIdx", None)
    levels = range(len(config.missing_percent_strs)) if idx is None else [idx]

    marker_set = ("the markers the reference-based imputer could not match"
                  if excluded_only else
                  "the markers every method retains" if overlapped_only else
                  "every marker in the target")
    print(f"Scoring {imp_method or 'BiU-Net'} on host {os.getenv('HOSTNAME', 'unknown')}")
    print(f"Checkpoint {config.checkpoint}; marker set: {marker_set}")

    for level in levels:
        for state in states:
            print(f"\n===== missing {config.missing_percent_strs[level]}, "
                  f"random state {state} =====", flush=True)
            benchmark(config, level, state, imp_method,
                      overlapped_only=overlapped_only, excluded_only=excluded_only)


if __name__ == "__main__":
    args = parse_args()
    config = ModelConfig.from_yaml(args.configFile)
    if args.report:
        scda = ModelConfig.from_yaml(args.reportScda) if args.reportScda else None
        report_table(config, scda)
    else:
        main(config, imp_method=args.impMethod)
