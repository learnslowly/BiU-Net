"""Convert a Beagle output VCF into the phased four-class CSV the benchmark reads.

Usage: python scripts/beagle_vcf_to_csv.py VCF OUT.csv.gz --axis AXIS.csv.gz
"""
import argparse
import gzip
import os

CONV = {'0|0': 1, '0|1': 2, '1|0': 3, '1|1': 4,
        '0/0': 1, '0/1': 2, '1/0': 3, '1/1': 4}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--chromosome', type=int, required=True)
    p.add_argument('--population', default='ALL')
    p.add_argument('--rand-state', type=int, default=2000)
    p.add_argument('--missing', default='90%')
    p.add_argument('--beagle-dir', default='$REMOTE_ROOT/beagle')
    p.add_argument('--data-dir', default='data')
    return p.parse_args()


def axis_positions(csv_path):
    positions = []
    with gzip.open(csv_path, 'rt') as f:
        f.readline()
        for line in f:
            positions.append(int(line[:line.index(',')]))
    return positions


def main():
    a = parse_args()
    ds, c, pop, m = a.dataset, a.chromosome, a.population, a.missing
    stem = f"{a.beagle_dir}/impute/{a.rand_state}/{m}/{ds}_chr{c}_{pop}_missing{m}_imputed"
    vcf, out_csv = f"{stem}.vcf.gz", f"{stem}.csv.gz"
    test_csv = f"{a.data_dir}/{ds}/split/{ds}_chr{c}_{pop}_test.csv.gz"
    for path in (vcf, test_csv):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    axis = axis_positions(test_csv)
    axis_set = set(axis)
    print(f"{ds} chr{c}: target axis {len(axis)} markers", flush=True)

    rows, seen, n_records, n_dup = {}, set(), 0, 0
    n_samples = None
    with gzip.open(vcf, 'rt') as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                n_samples = len(line.rstrip('\n').split('\t')) - 9
                continue
            n_records += 1
            fields = line.rstrip('\n').split('\t')
            pos = int(fields[1])
            if pos not in axis_set:
                continue
            if pos in seen:
                n_dup += 1
                continue
            seen.add(pos)
            # Take the genotype component; FORMAT may add DS, GP and others.
            rows[pos] = [CONV.get(s.split(':', 1)[0], '') for s in fields[9:]]

    n_present = len(rows)
    print(f"  Beagle returned {n_records} records, {n_present} of them on the axis; "
          f"{len(axis) - n_present} axis markers uncalled; {n_dup} duplicate "
          f"positions dropped; {n_samples} samples", flush=True)
    if n_present == 0:
        raise SystemExit("no Beagle record matched the target axis")

    with gzip.open(out_csv, 'wt') as f:
        f.write('POS,' + ','.join(str(i) for i in range(1, n_samples + 1)) + '\n')
        for pos in axis:
            if pos in rows:
                f.write(str(pos) + ',' + ','.join(str(v) for v in rows[pos]) + '\n')
    print(f"  wrote {out_csv}", flush=True)


if __name__ == '__main__':
    main()
