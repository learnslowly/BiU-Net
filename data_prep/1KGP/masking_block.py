"""Block-contiguous masking for chr22 test set.
"""
import argparse
import os
import numpy as np
import pandas as pd


def add_block_missingness(df, missing_perc, random_state, block_size=2000):
    """
    Block-contiguous masking. For each sample (column), pick non-overlapping
    contiguous runs of variants (rows) and set them to 0 (missing).

    df: DataFrame, rows = variants, columns = samples, values in {1,2,3,4}
    Returns a DataFrame with same shape, masked positions set to 0.
    """
    rng = np.random.RandomState(random_state)
    n_variants, n_samples = df.shape

    n_to_mask = int(round(n_variants * missing_perc))
    if n_to_mask < block_size:
        block_size = max(1, n_to_mask // 2)

    n_blocks = max(1, n_to_mask // block_size)
    actual_block_size = n_to_mask // n_blocks
    last_block_extra = n_to_mask - actual_block_size * n_blocks

    data = df.values.copy()

    for s in range(n_samples):
        # n_blocks non-overlapping block starts; reject-sample for simplicity
        starts = []
        tries = 0
        max_tries = n_blocks * 20
        while len(starts) < n_blocks and tries < max_tries:
            cand = rng.randint(0, n_variants - actual_block_size + 1)
            ok = all(abs(cand - s0) >= actual_block_size for s0 in starts)
            if ok:
                starts.append(cand)
            tries += 1
        # Fallback if rejection failed: just pick non-overlapping evenly-spaced
        if len(starts) < n_blocks:
            stride = n_variants // n_blocks
            starts = [rng.randint(0, max(1, stride - actual_block_size)) + i * stride
                      for i in range(n_blocks)]
            starts = [min(s0, n_variants - actual_block_size) for s0 in starts]
        starts.sort()
        for bi, s0 in enumerate(starts):
            length = actual_block_size + (last_block_extra if bi == n_blocks - 1 else 0)
            length = min(length, n_variants - s0)
            data[s0:s0 + length, s] = 0

    return pd.DataFrame(data, index=df.index, columns=df.columns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='1KGP')
    p.add_argument('--chromosome', type=int, default=22)
    p.add_argument('--population', default='ALL')
    p.add_argument('--missingRatios', nargs='+', type=float, default=[0.95])
    p.add_argument('--random_state', type=int, default=7,
                   help='Use a fresh rand_state so block-masked outputs sit at masked/{rs}/...')
    p.add_argument('--block_size', type=int, default=2000)
    args = p.parse_args()

    test_csv_gz = f'./split/{args.dataset}_chr{args.chromosome}_{args.population}_test.csv.gz'
    print(f'Loading: {test_csv_gz}')
    df = pd.read_csv(test_csv_gz, compression='gzip', index_col=0)
    print(f'  shape: {df.shape}  (variants x samples)')

    for r in args.missingRatios:
        out_dir = f'./masked/{args.random_state}/{r * 100:.0f}%'
        os.makedirs(out_dir, exist_ok=True)
        out_csv = f'{out_dir}/{args.dataset}_chr{args.chromosome}_{args.population}_missing{r * 100:.0f}%_masked.csv.gz'
        print(f'Block masking @ rate={r}, block_size={args.block_size}, rs={args.random_state}')
        masked = add_block_missingness(df, r, args.random_state, block_size=args.block_size)
        actual_missing = (masked.values == 0).sum() / masked.values.size
        print(f'  actual missing fraction: {actual_missing:.4f}')
        masked.to_csv(out_csv, compression='gzip', index=True)
        print(f'  wrote {out_csv}')


if __name__ == '__main__':
    main()
