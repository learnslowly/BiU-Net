"""Classify the target markers Beagle drops when a reference panel is supplied.

    absent    the position does not occur in the panel at all -- a variant
              private to the target cohort, the sites reference distance
              actually removes;
    alleles   the position occurs but the REF/ALT list differs -- a
              representation difference (extra ALT allele at a multiallelic
              site, opposite REF/ALT orientation, different indel spelling),
              not a biological absence.

Usage: python scripts/beagle_excluded_markers.py <dataset> <chrom> [<chrom> ...]
Writes Results/beagle_excluded_markers_{dataset}.csv
"""
import gzip
import os
import sys

import numpy as np
import pandas as pd

# See scripts/align_ref_to_sgdp.py for what GENOTYPE_DATA_DIR points at.
DATA = os.environ.get("GENOTYPE_DATA_DIR", "./data")
PANEL = f"{DATA}/1KGP/ref"
OUTDIR = "Results"
BINS = [0.0, 0.001, 0.005, 0.010, 0.100, 0.200, 0.300, 0.400, 0.500]
LABELS = ["<0.1%", "0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%",
          "30%~40%", "40%~50%"]


def vcf_markers(path):
    """{pos: set of allele-tuples} for every record, streamed."""
    by_pos = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 6)
            pos, ref, alt = int(p[1]), p[3], p[4]
            by_pos.setdefault(pos, set()).add((ref, alt))
    return by_pos


def target_maf(dataset, chrom):
    """MAF per position from the target training split (benchmark.py's source)."""
    path = f"{DATA}/{dataset}/split/{dataset}_chr{chrom}_ALL_train.csv.gz"
    df = pd.read_csv(path, compression="gzip", index_col=0)
    df = df[~df.index.duplicated(keep="first")]
    # genotype codes 1=0|0 2=0|1 3=1|0 4=1|1 -> alt dosage 0,1,1,2
    dose = df.replace({1: 0, 2: 1, 3: 1, 4: 2}).to_numpy(dtype=np.float32)
    freq = np.nanmean(dose, axis=1) / 2.0
    maf = np.minimum(freq, 1.0 - freq)
    return pd.Series(maf, index=df.index)


rows = []
dataset = sys.argv[1]
for chrom in sys.argv[2:]:
    # Any masked VCF for this chromosome carries the same marker list -- masking
    # blanks genotypes, it does not drop records -- so take whichever exists.
    import glob as _glob
    cands = sorted(_glob.glob(f"{DATA}/{dataset}/masked/*/*/"
                              f"{dataset}_chr{chrom}_ALL_missing*_*.vcf.gz"))
    if not cands:
        sys.exit(f"no masked/target VCF found for {dataset} chr{chrom}")
    targ_vcf = cands[0]
    print(f"chr{chrom}: target VCF {targ_vcf}", flush=True)
    ref_vcf = f"{PANEL}/1KGP_chr{chrom}_ALL_ref.vcf.gz"
    print(f"chr{chrom}: reading panel {ref_vcf}", flush=True)
    ref = vcf_markers(ref_vcf)
    n_ref = sum(len(v) for v in ref.values())
    print(f"chr{chrom}: panel has {n_ref:,} markers at {len(ref):,} positions",
          flush=True)

    maf = target_maf(dataset, chrom)
    bins = pd.cut(maf, bins=BINS, labels=LABELS, include_lowest=True, right=True)

    # Classify the axis benchmark.py scores, the split CSV index, rather than a
    # masked VCF, whose marker set may come from an older pipeline.
    axis = maf.index.to_numpy()
    status = {int(pos): ("matched" if int(pos) in ref else "absent") for pos in axis}
    n_allele_mismatch = 0
    opener = gzip.open if targ_vcf.endswith(".gz") else open
    with opener(targ_vcf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            q = line.split("\t", 6)
            pos, alleles = int(q[1]), (q[3], q[4])
            if pos in ref and alleles not in ref[pos] and pos in status:
                status[pos] = "alleles"
                n_allele_mismatch += 1
    print(f"chr{chrom}: allele mismatches seen in the available VCF: "
          f"{n_allele_mismatch:,}", flush=True)
    st = pd.Series(status)
    print(f"chr{chrom}: target markers {len(st):,} -> "
          f"{(st=='matched').sum():,} matched, "
          f"{(st=='absent').sum():,} absent from panel, "
          f"{(st=='alleles').sum():,} allele mismatch", flush=True)

    common = st.index.intersection(bins.index)
    tab = pd.crosstab(bins.loc[common], st.loc[common])
    for label in LABELS:
        if label not in tab.index:
            continue
        r = tab.loc[label]
        total = int(r.sum())
        rows.append({
            "dataset": dataset, "chr": int(chrom), "MAF_bin": label,
            "n_target": total,
            "matched": int(r.get("matched", 0)),
            "absent_from_panel": int(r.get("absent", 0)),
            "allele_mismatch": int(r.get("alleles", 0)),
            "pct_dropped": round(100 * (total - int(r.get("matched", 0))) / total, 2)
            if total else np.nan,
            "panel_markers": n_ref,
        })

os.makedirs(OUTDIR, exist_ok=True)
out = pd.DataFrame(rows)
path = os.path.join(OUTDIR, f"beagle_excluded_markers_{dataset}.csv")
out.to_csv(path, index=False)
print(f"\nwrote {path}")
print(out.to_string(index=False))
