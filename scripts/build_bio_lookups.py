"""Build per-SNP biological-info lookup tables for chr22:
  (1) cM coordinate (from plink genetic map; interpolated to all SNP positions)
  (2) MAF (computed from training set)

Saves: data/bio/chr22_bio.npz with arrays:
  - pos:     int64 [N] sorted bp positions
  - cM_abs:  float32 [N] absolute cM coordinate
  - cM_gap:  float32 [N] cM gap to previous SNP (0 for first)
  - cM_norm: float32 [N] cM normalized to chromosome length
  - maf:     float32 [N] minor allele frequency from train data
"""
import os
import gzip
import numpy as np
import pandas as pd


PLINK_MAP = '$GENOTYPE_DATA_DIR/1KGP/orig/GRCh38/genetic_maps/plink.chr22.GRCh38.map'
TEST_CSV  = '$GENOTYPE_DATA_DIR/1KGP/split/1KGP_chr22_ALL_test.csv.gz'
TRAIN_CSV = '$GENOTYPE_DATA_DIR/1KGP/split/1KGP_chr22_ALL_train.csv.gz'
OUT_DIR   = '$GENOTYPE_DATA_DIR/bio'
OUT_FILE  = os.path.join(OUT_DIR, 'chr22_bio.npz')


def load_plink_map(path):
    """plink format: chr, snpId, cM, bp."""
    df = pd.read_csv(path, sep=r'\s+', header=None, names=['chr', 'rs', 'cM', 'bp'])
    df = df.sort_values('bp').reset_index(drop=True)
    return df['bp'].values.astype(np.int64), df['cM'].values.astype(np.float64)


def interpolate_cM(snp_bp, map_bp, map_cM):
    """Linear interpolation; clamp out-of-range positions to map endpoints."""
    return np.interp(snp_bp.astype(np.float64), map_bp.astype(np.float64), map_cM).astype(np.float32)


def compute_maf_from_train(train_csv):
    """MAF per SNP from training-set genotype matrix.
    Vocab: 1=0|0, 2=0|1, 3=1|0, 4=1|1. Alt allele count: 2=>1, 3=>1, 4=>2; 1=>0.
    MAF = min(p, 1-p) where p is alt allele frequency.
    """
    df = pd.read_csv(train_csv, compression='gzip', index_col=0)
    # rows = SNPs (POS as index), cols = sample IDs
    arr = df.values.astype(np.int8)
    alt_count = np.zeros_like(arr, dtype=np.int8)
    alt_count[arr == 2] = 1
    alt_count[arr == 3] = 1
    alt_count[arr == 4] = 2
    # Per-SNP sum of alt allele counts; total alleles = 2 * n_samples
    total_alleles = 2 * arr.shape[1]
    af = alt_count.sum(axis=1).astype(np.float64) / total_alleles
    maf = np.minimum(af, 1 - af).astype(np.float32)
    return df.index.values.astype(np.int64), maf


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading plink map: {PLINK_MAP}")
    map_bp, map_cM = load_plink_map(PLINK_MAP)
    print(f"  map: {len(map_bp)} entries, cM range [{map_cM.min():.3f}, {map_cM.max():.3f}]")

    print(f"Computing MAF from training set: {TRAIN_CSV}")
    train_pos, train_maf = compute_maf_from_train(TRAIN_CSV)
    print(f"  train: {len(train_pos)} SNPs, MAF range [{train_maf.min():.5f}, {train_maf.max():.5f}]")

    print(f"Reading test SNP positions: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV, compression='gzip', index_col=0)
    test_pos = test_df.index.values.astype(np.int64)
    print(f"  test: {len(test_pos)} SNPs")

    # Assume train and test share the same SNP set (standard panel split)
    assert np.array_equal(np.sort(train_pos), np.sort(test_pos)), \
        f"train/test SNP sets differ: {len(train_pos)} vs {len(test_pos)}"
    pos = np.sort(test_pos)

    # Reorder MAF to match sorted pos
    train_pos_sorted_idx = np.argsort(train_pos)
    maf_sorted = train_maf[train_pos_sorted_idx]

    # cM interpolation
    cM_abs = interpolate_cM(pos, map_bp, map_cM)
    cM_total = float(cM_abs.max())
    cM_norm = (cM_abs / cM_total).astype(np.float32)
    cM_gap = np.zeros_like(cM_abs)
    cM_gap[1:] = cM_abs[1:] - cM_abs[:-1]

    print(f"cM_abs range: [{cM_abs.min():.3f}, {cM_abs.max():.3f}]")
    print(f"cM_gap range: [{cM_gap.min():.5f}, {cM_gap.max():.5f}]  median: {np.median(cM_gap):.5f}")
    print(f"maf range:    [{maf_sorted.min():.5f}, {maf_sorted.max():.5f}]")

    np.savez(OUT_FILE,
             pos=pos,
             cM_abs=cM_abs,
             cM_gap=cM_gap.astype(np.float32),
             cM_norm=cM_norm,
             maf=maf_sorted)
    print(f"Saved: {OUT_FILE}")


if __name__ == '__main__':
    main()
