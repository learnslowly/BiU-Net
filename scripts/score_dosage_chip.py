"""Score alternative allele dosage against the truth on the array-masked positions.

Usage: python scripts/score_dosage_chip.py --configFile CONFIG [...]
"""
import argparse
import gzip
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.modelconfig import ModelConfig
from data.dataset import SNPsDataset_HDF5

GENO_CLASSES = (1, 2, 3, 4)          # 0|0, 0|1, 1|0, 1|1
CLASS_DOSAGE = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--configFile', required=True)
    p.add_argument('--beagleVcf', default=None,
                   help='Beagle output to score alongside, read from its DS field')
    p.add_argument('--out', required=True)
    p.add_argument('--batchSize', type=int, default=8)
    p.add_argument('--dosageCache', default=None,
                   help='npz to write the model dosage matrix to, and to read it '
                        'back from on a later run so scoring costs no inference')
    return p.parse_args()


def read_truth(csv_path):
    """Positions and the genotype-code matrix of the test split."""
    positions, rows = [], []
    with gzip.open(csv_path, 'rt') as f:
        header = f.readline().rstrip('\n').split(',')
        n_samples = len(header) - 1
        for line in f:
            parts = line.rstrip('\n').split(',')
            positions.append(int(parts[0]))
            rows.append(np.fromiter((int(v) for v in parts[1:]), dtype=np.int8,
                                    count=n_samples))
    return np.asarray(positions, dtype=np.int64), np.vstack(rows)


def maf_from_train(csv_path):
    """Minor allele frequency per position, keyed by position.

    The training and test splits of a dataset can differ by a row where one of
    them dropped a duplicated position, so the two are joined on the position
    rather than on the row number.
    """
    pos, maf = [], []
    with gzip.open(csv_path, 'rt') as f:
        f.readline()
        for line in f:
            line = line.rstrip('\n')
            pos.append(int(line[:line.index(',')]))
            n_het = line.count(',2') + line.count(',3')
            n_hom = line.count(',4')
            n_called = line.count(',1') + n_het + n_hom
            af = 0.0 if n_called == 0 else (n_het + 2 * n_hom) / (2.0 * n_called)
            maf.append(min(af, 1.0 - af))
    return np.asarray(pos, dtype=np.int64), np.asarray(maf, dtype=np.float64)


def bin_labels(edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        def fmt(v):
            v *= 100
            return f"{v:.3f}".rstrip('0').rstrip('.') if v < 1 else f"{v:.1f}".rstrip('0').rstrip('.')
        out.append(f"{fmt(lo)}%~{fmt(hi)}%")
    return out


def pooled_r2(truth, pred):
    """One correlation over every cell given."""
    if truth.size < 2:
        return np.nan
    t = truth.astype(np.float64).ravel()
    p = pred.astype(np.float64).ravel()
    tv, pv = t - t.mean(), p - p.mean()
    den = (tv * tv).sum() * (pv * pv).sum()
    return np.nan if den == 0 else float((tv * pv).sum() ** 2 / den)


def per_variant_r2(truth, pred):
    """Mean over loci of the squared correlation across samples at that locus.

    Loci whose truth or prediction is constant have no correlation to report and
    are counted rather than silently averaged in.
    """
    vals = []
    n_const = 0
    for i in range(truth.shape[0]):
        t = truth[i].astype(np.float64)
        p = pred[i].astype(np.float64)
        tv, pv = t - t.mean(), p - p.mean()
        den = (tv * tv).sum() * (pv * pv).sum()
        if den == 0:
            n_const += 1
            continue
        vals.append((tv * pv).sum() ** 2 / den)
    return (float(np.mean(vals)) if vals else np.nan), len(vals), n_const


def main():
    args = parse_args()
    config = ModelConfig.from_yaml(args.configFile)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds, chrom, pop = config.dataset, config.chromosome, config.population
    data_dir = config.dataDir
    test_csv = f"{data_dir}/{ds}/split/{ds}_chr{chrom}_{pop}_test.csv.gz"
    train_csv = f"{data_dir}/{ds}/split/{ds}_chr{chrom}_{pop}_train.csv.gz"
    rand_state = (config.testRandStates or [0])[0]
    m_str = config.missing_percent_strs[0]
    obs_npy = (f"{data_dir}/{ds}/masked/{rand_state}/{m_str}/"
               f"array_observed_indices_chr{chrom}_{pop}.npy")

    print("reading truth and frequencies", flush=True)
    positions, truth_codes = read_truth(test_csv)
    train_pos, train_maf = maf_from_train(train_csv)
    maf_of_pos = dict(zip(train_pos.tolist(), train_maf.tolist()))
    maf = np.array([maf_of_pos.get(int(p), np.nan) for p in positions], dtype=np.float64)

    # The array is defined by the positions it carries, so the observed set is
    # taken from the site list rather than from row numbers into another split.
    sites_tsv = os.path.join(os.path.dirname(obs_npy),
                             f"array_sites_chr{chrom}_{pop}.tsv")
    site_pos = set()
    with open(sites_tsv) as f:
        for line in f:
            if line.startswith('chrom') or not line.strip():
                continue
            site_pos.add(int(line.split()[1]))
    observed_by_pos = np.array([int(p) in site_pos for p in positions], dtype=bool)
    masked_rows = ~observed_by_pos

    # What the masking script actually applied to the test grid was a row number
    # taken from the training split, so where the two axes differ the mask can
    # sit on the wrong rows. Report the disagreement rather than let it pass.
    observed_by_row = np.zeros(len(positions), dtype=bool)
    idx = np.load(obs_npy)
    observed_by_row[idx[idx < len(positions)]] = True
    n_disagree = int((observed_by_row != observed_by_pos).sum())
    print(f"axis: test {len(positions)} rows, train {len(train_pos)} rows; "
          f"site list {len(site_pos)} positions, {int(observed_by_pos.sum())} on the "
          f"test axis; row-index and position-based masks disagree at "
          f"{n_disagree} rows", flush=True)
    if not np.isfinite(maf).all():
        print(f"  {int((~np.isfinite(maf)).sum())} test positions absent from the "
              f"training split; they are excluded from every bin", flush=True)
    print(f"axis {len(positions)}, array sites {int(observed_by_pos.sum())}, "
          f"imputed positions {int(masked_rows.sum())}, "
          f"samples {truth_codes.shape[1]}", flush=True)

    truth_dosage = np.select(
        [truth_codes == 1, (truth_codes == 2) | (truth_codes == 3), truth_codes == 4],
        [0.0, 1.0, 2.0], default=np.nan).astype(np.float32)

    row_of_pos = {int(p): i for i, p in enumerate(positions)}
    n_loci, n_samples = truth_codes.shape
    dose_sum = np.zeros((n_loci, n_samples), dtype=np.float32)
    dose_cnt = np.zeros((n_loci, n_samples), dtype=np.int16)
    samp_col = {}

    cache = args.dosageCache
    if cache and os.path.exists(cache):
        z = np.load(cache)
        pred_dosage = z['pred_dosage']
        print(f"read model dosages from {cache}", flush=True)
        return_early = True
    else:
        return_early = False

    if not return_early:
        print("running the model", flush=True)
        rand_state = (config.testRandStates or [0])[0]
        masked_file = config.masked_dataset_files(rand_state)[0]
        dataset = SNPsDataset_HDF5(masked_file)
        loader = DataLoader(dataset, batch_size=args.batchSize, shuffle=False, num_workers=2)

        from model.unet import BiUNet
        model = BiUNet(config).to(device)
        ckpt = torch.load(config.checkpoint, map_location=device, weights_only=False)
        state = OrderedDict((k[7:] if k.startswith('module.') else k, v)
                            for k, v in ckpt['state_dict'].items())
        model.load_state_dict(state)
        model.eval()
        print(f"loaded {config.checkpoint}", flush=True)

        dosage_tbl = torch.tensor(CLASS_DOSAGE, device=device)
        with torch.no_grad():
            for bi, (snps, idx) in enumerate(loader):
                snps = snps.to(device)
                idx = idx.to(device)
                one_hot = torch.nn.functional.one_hot(
                    snps.long(), num_classes=config.vocabSize).float()
                bio_arg = None
                if config.bioAware:
                    from data.bio_lookup import compute_bio_channels
                    bio = compute_bio_channels(
                        idx[:, :, 0],
                        encoding=getattr(config, 'bioEncoding', 'normPos'),
                        bio_file=getattr(config, 'bioFile', None),
                    )
                    if getattr(config, 'useFiLM', False):
                        bio_arg = bio
                    else:
                        one_hot = torch.cat([one_hot, bio], dim=-1)
                logits = model(one_hot, bio=bio_arg)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.softmax(logits[:, :, list(GENO_CLASSES)].float(), dim=2)
                dosage = (probs * dosage_tbl).sum(dim=2).cpu().numpy()

                loci = idx[:, :, 0].cpu().numpy()
                samps = idx[:, :, 1].cpu().numpy()
                for b in range(dosage.shape[0]):
                    s = int(samps[b, 0])
                    if s not in samp_col:
                        samp_col[s] = len(samp_col)
                    col = samp_col[s]
                    rows = np.fromiter((row_of_pos.get(int(l), -1) for l in loci[b]),
                                       dtype=np.int64, count=loci.shape[1])
                    keep = rows >= 0
                    np.add.at(dose_sum, (rows[keep], col), dosage[b][keep])
                    np.add.at(dose_cnt, (rows[keep], col), 1)
                if bi % 50 == 0:
                    print(f"  batch {bi}/{len(loader)}", flush=True)

        covered = dose_cnt > 0
        pred_dosage = np.full_like(dose_sum, np.nan)
        pred_dosage[covered] = dose_sum[covered] / dose_cnt[covered]
        print(f"model covered {covered.mean():.4f} of the grid", flush=True)
        if cache:
            np.savez_compressed(cache, pred_dosage=pred_dosage)
            print(f"cached model dosages to {cache}", flush=True)

    beagle_dosage = None
    if args.beagleVcf:
        print("reading Beagle dosages", flush=True)
        beagle_dosage = np.full((n_loci, n_samples), np.nan, dtype=np.float32)
        n_hit = 0
        with gzip.open(args.beagleVcf, 'rt') as f:
            for line in f:
                if line.startswith('##'):
                    continue
                if line.startswith('#CHROM'):
                    continue
                fields = line.rstrip('\n').split('\t')
                row = row_of_pos.get(int(fields[1]), -1)
                if row < 0:
                    continue
                fmt = fields[8].split(':')
                if 'DS' not in fmt:
                    continue
                k = fmt.index('DS')
                vals = fields[9:]
                if len(vals) != n_samples:
                    raise SystemExit(f"Beagle has {len(vals)} samples, truth has {n_samples}")
                beagle_dosage[row] = [float(v.split(':')[k]) for v in vals]
                n_hit += 1
        print(f"  Beagle returned {n_hit} of {n_loci} axis positions", flush=True)

    edges = list(config.bins)
    labels = bin_labels(edges)
    which_bin = np.digitize(maf, edges[1:-1], right=False)
    which_bin[~np.isfinite(maf)] = -1

    # Beagle leaves no call where its panel has no matching marker, so the two
    # methods do not cover the same rows on every dataset. Both views are
    # reported: every imputed position, each method on what it produced, and the
    # subset Beagle also returns, where the two are strictly comparable.
    beagle_rows = (np.isfinite(beagle_dosage).all(axis=1) if beagle_dosage is not None
                   else np.ones(n_loci, dtype=bool))
    print(f"Beagle covers {int((masked_rows & beagle_rows).sum())} of the "
          f"{int(masked_rows.sum())} imputed positions", flush=True)

    rows_out = []
    def score(name, pred, row_sel, view):
        for bi_, label in enumerate(labels):
            sel = row_sel & (which_bin == bi_)
            if not sel.any():
                continue
            t = truth_dosage[sel]
            p = pred[sel]
            ok = np.isfinite(t) & np.isfinite(p)
            keep_rows = ok.all(axis=1)
            t, p = t[keep_rows], p[keep_rows]
            pv, n_used, n_const = per_variant_r2(t, p)
            rows_out.append(dict(method=name, marker_set=view, MAF_bin=label,
                                 n_variants=int(sel.sum()), n_scored=int(t.shape[0]),
                                 n_constant=n_const,
                                 pooled_r2=pooled_r2(t, p), per_variant_r2=pv))
        sel = row_sel
        t, p = truth_dosage[sel], pred[sel]
        ok = np.isfinite(t) & np.isfinite(p)
        keep_rows = ok.all(axis=1)
        t, p = t[keep_rows], p[keep_rows]
        pv, n_used, n_const = per_variant_r2(t, p)
        rows_out.append(dict(method=name, marker_set=view, MAF_bin='Overall',
                             n_variants=int(sel.sum()), n_scored=int(t.shape[0]),
                             n_constant=n_const,
                             pooled_r2=pooled_r2(t, p), per_variant_r2=pv))

    bu = f"BiU-Net {config.runId} (dosage)"
    score(bu, pred_dosage, masked_rows, 'all imputed positions')
    if beagle_dosage is not None:
        score("Beagle (DS)", beagle_dosage, masked_rows, 'all imputed positions')
        overlap = masked_rows & beagle_rows
        score(bu, pred_dosage, overlap, 'Beagle-retained')
        score("Beagle (DS)", beagle_dosage, overlap, 'Beagle-retained')

    import csv
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {args.out}", flush=True)
    for r in rows_out:
        print(f"  {r['method']:38s} {r['marker_set']:>22s} {r['MAF_bin']:>12s} "
              f"pooled {r['pooled_r2']:.4f}  per-variant {r['per_variant_r2']:.4f} "
              f"(n={r['n_scored']}, const={r['n_constant']})", flush=True)


if __name__ == '__main__':
    main()
