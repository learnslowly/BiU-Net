"""VCF → CSV converter for the LOS pipeline.
"""
import gzip
import os
import re
import argparse
import multiprocessing as mp
from typing import List, Tuple

import numpy as np
import pandas as pd


# Module-level so workers can pickle / re-create cheaply
GT_TO_INT = {
    b'0|0': 1,
    b'0|1': 2,
    b'1|0': 3,
    b'1|1': 4,
}


def _convert_lines(args) -> Tuple[List[str], np.ndarray]:
    """Worker: convert a list of VCF data lines into (positions, rows_uint8).

    Receives args=(idx, lines, num_samples) so callers can recover the original
    batch order from `idx` after pool.imap_unordered() returns.
    """
    idx, lines, num_samples = args
    positions: List[str] = []
    rows = np.zeros((len(lines), num_samples), dtype=np.uint8)
    for i, line in enumerate(lines):
        fields = line.strip().split(b'\t')
        positions.append(fields[1].decode('utf-8'))
        genotypes = fields[9:]
        for j, gt in enumerate(genotypes):
            # GT can be 'A|B:DP:...' — only the field before the first colon matters
            gt_only = gt.split(b':', 1)[0] if b':' in gt else gt
            rows[i, j] = GT_TO_INT.get(gt_only, 0)  # 0 for unknown/missing
    return idx, positions, rows


def parse_vcf_parallel(in_file: str, out_file: str, num_workers: int = 1,
                       chunk_size: int = 10000) -> Tuple[int, int]:
    """Parallel VCF→CSV. `num_workers=1` reproduces the original sequential path
    (with the same bit-identical output)."""
    # First pass: collect header + count variants + slurp data lines.
    # gzip decompression dominates the read so we do it once.
    print(f"First pass: reading header + variants ({num_workers} workers)...")
    header = None
    data_lines: List[bytes] = []
    with gzip.open(in_file, 'rb') as f:
        for line in f:
            if line.startswith(b'##'):
                continue
            if line.startswith(b'#CHROM'):
                header = line.decode('utf-8').strip().split('\t')
                continue
            data_lines.append(line)

    if header is None:
        raise ValueError("No header found in VCF file")

    sample_names = header[9:]
    num_samples = len(sample_names)
    num_variants = len(data_lines)

    pattern = re.compile(r'\d+$')
    sample_names_clean = [
        pattern.search(name).group() if pattern.search(name) else name
        for name in sample_names
    ]
    print(f"Found {num_variants} variants and {num_samples} samples")

    # Split into per-worker batches. ~4 batches per worker for load balancing.
    n_batches = max(1, num_workers * 4)
    batch_size = max(1, (num_variants + n_batches - 1) // n_batches)
    batches = [
        (i, data_lines[i * batch_size:(i + 1) * batch_size], num_samples)
        for i in range(n_batches)
        if i * batch_size < num_variants
    ]

    print(f"Second pass: converting in {len(batches)} batches across {num_workers} workers...")

    if num_workers == 1:
        # Single-threaded for compatibility / smoke tests.
        results = [_convert_lines(b) for b in batches]
    else:
        with mp.Pool(num_workers) as pool:
            # imap_unordered streams results as workers finish; the (idx, ...)
            # tuple lets us re-order before writing.
            results = list(pool.imap_unordered(_convert_lines, batches))

    # Re-order results by original batch index, then concat.
    results.sort(key=lambda r: r[0])
    all_positions: List[str] = []
    row_arrays: List[np.ndarray] = []
    for _, positions, rows in results:
        all_positions.extend(positions)
        row_arrays.append(rows)
    full_rows = np.concatenate(row_arrays, axis=0)

    # Write to gzipped CSV. pandas handles the gzip; index_label='POS' keeps the
    # column header bit-identical to the streaming version.
    print(f"Writing {len(all_positions)} variants to {out_file}...")
    df = pd.DataFrame(full_rows, index=all_positions, columns=sample_names_clean)
    df.index.name = 'POS'
    df.to_csv(out_file, compression='gzip')

    print(f"Conversion complete: {num_variants} variants, {num_samples} samples")
    return num_samples, num_variants


def _default_num_workers() -> int:
    env = os.environ.get('SLURM_CPUS_PER_TASK')
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def main(chromosome, population, subset, hla=False, num_workers=None,
         chunk_size=10000):
    dataset = 'LOS'
    suffix = "_HLA" if hla else ""
    vcf_file = f"./split/{dataset}_chr{chromosome}_{population}_{subset}{suffix}.vcf.gz"
    csv_gz_file = f"./split/{dataset}_chr{chromosome}_{population}_{subset}{suffix}.csv.gz"

    if num_workers is None:
        num_workers = _default_num_workers()

    num_samples, num_snps = parse_vcf_parallel(
        vcf_file, csv_gz_file, num_workers=num_workers, chunk_size=chunk_size,
    )
    print(f"{csv_gz_file} contains {num_samples} samples and {num_snps} SNPs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a single VCF file for a specific chromosome, population, and subset')
    parser.add_argument('chromosome', type=int, help='Chromosome number to process')
    parser.add_argument('population', type=str, help='Population to process')
    parser.add_argument('subset', type=str, help='Subset to process (train, val, test)')
    parser.add_argument('--hla', action='store_true', help='Use HLA region suffix (_HLA) for chromosome 6')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Number of variants per chunk (default: 10000; kept for CLI parity)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Worker processes; default = SLURM_CPUS_PER_TASK or os.cpu_count()')

    args = parser.parse_args()
    main(args.chromosome, args.population, args.subset, args.hla,
         num_workers=args.num_workers, chunk_size=args.chunk_size)

# Regular chromosome
# python vcf2csv.py 22 ALL train
# -> ./split/LOS_chr22_ALL_train.vcf.gz -> ./split/LOS_chr22_ALL_train.csv.gz

# Chromosome 6 HLA only
# python vcf2csv.py 6 ALL train --hla
# -> ./split/LOS_chr6_ALL_train_HLA.vcf.gz -> ./split/LOS_chr6_ALL_train_HLA.csv.gz
