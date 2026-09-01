"""Chip-array masking: only variants on a (synthetic) chip are observed.
"""
import argparse
import os
import gzip
import numpy as np
import pandas as pd


def compute_maf_from_codes(df):
    """codes: 1=0|0, 2=0|1, 3=1|0, 4=1|1, 0=missing. Returns MAF per variant."""
    data = df.values
    # Allele counts: each genotype contributes 2 alleles. 0|0 -> 0; 0|1/1|0 -> 1; 1|1 -> 2
    alt_count = np.zeros(data.shape[0], dtype=np.int64)
    total_count = np.zeros(data.shape[0], dtype=np.int64)
    for code, alt in [(1, 0), (2, 1), (3, 1), (4, 2)]:
        m = (data == code)
        alt_count += m.sum(axis=1) * alt
        total_count += m.sum(axis=1) * 2
    af = np.where(total_count > 0, alt_count / np.maximum(total_count, 1), 0.0)
    maf = np.minimum(af, 1 - af)
    return maf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='1KGP')
    p.add_argument('--chromosome', type=int, default=22)
    p.add_argument('--population', default='ALL')
    p.add_argument('--random_state', type=int, default=8,
                   help='Fresh rs slot for chip-masked outputs')
    p.add_argument('--target_chip_count', type=int, default=12000,
                   help='Approximate number of chip "observed" positions on chr22')
    p.add_argument('--maf_min', type=float, default=0.05)
    args = p.parse_args()

    test_csv = f'./split/{args.dataset}_chr{args.chromosome}_{args.population}_test.csv.gz'
    print(f'Loading: {test_csv}')
    df = pd.read_csv(test_csv, compression='gzip', index_col=0)
    n_var, n_sam = df.shape
    print(f'  shape: {n_var} variants x {n_sam} samples')

    # Compute MAF per variant
    print('Computing MAF...')
    maf = compute_maf_from_codes(df)
    print(f'  MAF range: [{maf.min():.4f}, {maf.max():.4f}]')

    # Filter to common variants
    common_mask = (maf >= args.maf_min) & (maf <= (1 - args.maf_min))
    common_idx = np.where(common_mask)[0]
    print(f'  Common variants (MAF>={args.maf_min}): {len(common_idx)}')

    # Subsample to target chip count, evenly along position
    if len(common_idx) <= args.target_chip_count:
        chip_idx = common_idx
    else:
        step = max(1, len(common_idx) // args.target_chip_count)
        chip_idx = common_idx[::step][:args.target_chip_count]
    print(f'  Chip positions selected: {len(chip_idx)}')

    chip_positions = df.index.values[chip_idx]
    chip_set = set(chip_positions.tolist())

    # Compute missing rate
    pct_obs = len(chip_idx) / n_var
    print(f'  Observed fraction: {pct_obs:.4f}  (missing fraction: {1 - pct_obs:.4f})')

    # Build masked matrix: keep values at chip positions; set rest to 0
    print('Masking...')
    masked = df.copy()
    not_chip_mask = ~df.index.isin(chip_set)
    masked.loc[not_chip_mask, :] = 0
    actual_missing = (masked.values == 0).sum() / masked.values.size
    print(f'  actual missing fraction: {actual_missing:.4f}')

    # Save outputs
    missing_pct = int(round((1 - pct_obs) * 100))
    out_dir = f'./masked/{args.random_state}/{missing_pct}%'
    os.makedirs(out_dir, exist_ok=True)
    out_csv = f'{out_dir}/{args.dataset}_chr{args.chromosome}_{args.population}_missing{missing_pct}%_masked.csv.gz'
    masked.to_csv(out_csv, compression='gzip', index=True)
    print(f'  wrote {out_csv}')

    # Save chip position list (for record / reproducibility)
    chip_list_path = f'{out_dir}/chip_positions.tsv'
    with open(chip_list_path, 'w') as f:
        f.write('chr\tpos\n')
        for pos in chip_positions:
            f.write(f'{args.chromosome}\t{pos}\n')
    print(f'  wrote chip positions to {chip_list_path}')


if __name__ == '__main__':
    main()
