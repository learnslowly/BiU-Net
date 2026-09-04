"""Build a synthetic genotyping array and the target files that go with it.

Usage: python scripts/masking_chip_gw.py --dataset SGDP --chromosome 22 [...]
"""
import argparse
import gzip
import json
import os
import subprocess

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='1KGP / SGDP / LOS / HLA')
    p.add_argument('--chromosome', type=int, required=True)
    p.add_argument('--population', default='ALL')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--suffix', default='', help="e.g. '_HLA' for the HLA chr6 split")
    p.add_argument('--observed-fraction', type=float, default=0.10,
                   help='fraction of the axis the array carries (0.10 -> 90%% missing)')
    p.add_argument('--maf-min', type=float, default=0.05,
                   help='array variants are ascertained above this minor allele frequency')
    p.add_argument('--rand-state', type=int, default=2000,
                   help='label baked into the output path, keeping array masks '
                        'apart from the sporadic (0/42/1024) and colwise (1042) ones')
    p.add_argument('--sites', default=None,
                   help='optional TSV of chrom<TAB>pos to use as the array manifest '
                        'instead of building a synthetic one')
    return p.parse_args()


def scan_frequencies(csv_path):
    """One pass over a genotype CSV: positions, minor allele frequency per row.

    Codes are 1=0|0, 2=0|1, 3=1|0, 4=1|1, 0=missing, 5=pad, one digit per field,
    so counting the digit-with-comma pairs is enough and costs no field splitting.
    """
    positions, maf = [], []
    with gzip.open(csv_path, 'rt') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\n')
            pos = line[:line.index(',')]
            n_het = line.count(',2') + line.count(',3')
            n_hom = line.count(',4')
            n_ref = line.count(',1')
            n_called = n_ref + n_het + n_hom
            if n_called == 0:
                af = 0.0
            else:
                af = (n_het + 2 * n_hom) / (2.0 * n_called)
            positions.append(int(pos))
            maf.append(min(af, 1.0 - af))
    return np.asarray(positions, dtype=np.int64), np.asarray(maf, dtype=np.float64)


def pick_array_sites(maf, maf_min, observed_fraction):
    """Common variants, thinned evenly along the chromosome to the target count."""
    common = np.where(maf >= maf_min)[0]
    n_target = int(round(len(maf) * observed_fraction))
    if n_target >= len(common):
        return common
    # Even spacing in rank order, which is position order: the array covers the
    # chromosome uniformly rather than clustering where variants are dense.
    take = np.linspace(0, len(common) - 1, num=n_target).round().astype(np.int64)
    return common[np.unique(take)]


def write_masked_csv(input_csv, output_csv, observed_positions):
    """Stream the grid, replacing every non-array row with the missing code.

    Membership is decided on the genomic position rather than the row number:
    the site list is built from the training split, whose axis can differ from
    the test split's by a row where one of them dropped a duplicated position,
    and a row-number mask would then sit on the wrong markers from that point on.
    """
    n_kept = 0
    with gzip.open(input_csv, 'rt') as fin, gzip.open(output_csv, 'wt') as fout:
        header = fin.readline()
        fout.write(header)
        n_samples = len(header.rstrip('\n').split(',')) - 1
        zero_suffix = ',0' * n_samples + '\n'
        for line in fin:
            pos = line[:line.index(',')]
            if int(pos) in observed_positions:
                fout.write(line)
                n_kept += 1
            else:
                fout.write(pos + zero_suffix)
    return n_kept


def write_beagle_target_vcf(input_vcf, output_vcf, chromosome, positions):
    keep_tsv = output_vcf + '.keep.tsv'
    with open(keep_tsv, 'w') as f:
        for pos in positions:
            f.write(f"{chromosome}\t{pos}\n")
    try:
        subprocess.run(['bcftools', 'view', '-T', keep_tsv, '-Oz', '-o', output_vcf,
                        input_vcf], check=True)
        subprocess.run(['tabix', '-f', '-p', 'vcf', output_vcf], check=True)
    finally:
        os.remove(keep_tsv)


def main():
    args = parse_args()
    ds, c, pop, sfx = args.dataset, args.chromosome, args.population, args.suffix
    split = f"{args.data_dir}/{ds}/split"
    train_csv = f"{split}/{ds}_chr{c}_{pop}_train{sfx}.csv.gz"
    test_csv = f"{split}/{ds}_chr{c}_{pop}_test{sfx}.csv.gz"
    test_vcf = f"{split}/{ds}_chr{c}_{pop}_test{sfx}.vcf.gz"
    for path in (train_csv, test_csv, test_vcf):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    print(f"=== array masking: {ds} chr{c} {pop}{sfx}", flush=True)
    print(f"    frequencies from {train_csv}", flush=True)
    train_pos, maf = scan_frequencies(train_csv)

    if args.sites:
        # A real manifest: keep the axis positions the array carries.
        manifest = set()
        with open(args.sites) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                manifest.add(int(parts[1]))
        obs_idx = np.where(np.isin(train_pos, list(manifest)))[0]
        print(f"    manifest {args.sites}: {len(manifest)} sites, "
              f"{len(obs_idx)} on this axis", flush=True)
    else:
        obs_idx = pick_array_sites(maf, args.maf_min, args.observed_fraction)

    n_var = len(train_pos)
    obs_frac = len(obs_idx) / n_var
    missing_pct = int(round((1.0 - obs_frac) * 100))
    print(f"    axis {n_var} variants, array carries {len(obs_idx)} "
          f"({obs_frac:.4f}) -> missing {missing_pct}%", flush=True)
    print(f"    array MAF range [{maf[obs_idx].min():.4f}, {maf[obs_idx].max():.4f}], "
          f"median spacing {int(np.median(np.diff(train_pos[obs_idx])))} bp", flush=True)

    out_dir = f"{args.data_dir}/{ds}/masked/{args.rand_state}/{missing_pct}%"
    os.makedirs(out_dir, exist_ok=True)
    m_str = f"{missing_pct}%"

    sites_path = f"{out_dir}/array_sites_chr{c}_{pop}{sfx}.tsv"
    with open(sites_path, 'w') as f:
        f.write('chrom\tpos\tmaf\n')
        for i in obs_idx:
            f.write(f"{c}\t{train_pos[i]}\t{maf[i]:.6f}\n")
    print(f"    sites  -> {sites_path}", flush=True)

    idx_path = f"{out_dir}/array_observed_indices_chr{c}_{pop}{sfx}.npy"
    np.save(idx_path, obs_idx)

    out_csv = f"{out_dir}/{ds}_chr{c}_{pop}_missing{m_str}_masked{sfx}.csv.gz"
    print(f"    BiU-Net CSV -> {out_csv}", flush=True)
    n_kept = write_masked_csv(test_csv, out_csv, set(train_pos[obs_idx].tolist()))
    print(f"    {n_kept} of {len(obs_idx)} array positions found on the test axis",
          flush=True)

    out_vcf = f"{out_dir}/{ds}_chr{c}_{pop}_missing{m_str}_target{sfx}.vcf.gz"
    print(f"    Beagle VCF  -> {out_vcf}", flush=True)
    write_beagle_target_vcf(test_vcf, out_vcf, c, train_pos[obs_idx])

    summary = {
        'dataset': ds, 'chromosome': c, 'population': pop,
        'n_variants': int(n_var), 'n_array_sites': int(len(obs_idx)),
        'observed_fraction': float(obs_frac), 'missing_pct_label': m_str,
        'maf_min': args.maf_min, 'source': args.sites or 'synthetic',
        'maf_from': os.path.basename(train_csv),
    }
    with open(f"{out_dir}/array_summary_chr{c}_{pop}{sfx}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    done: {json.dumps(summary)}", flush=True)


if __name__ == '__main__':
    main()
