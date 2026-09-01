"""The dataset summary of Table 1: split sizes and marker counts.

Usage: python scripts/build_dataset_table.py
Writes Results/dataset_summary.csv
"""
import glob
import gzip
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config.modelconfig import ModelConfig  # noqa: E402

OUTDIR = "Results"
# The evaluation config of each reported dataset; it carries the paths.
CONFIGS = [
    "configs/test_seg128_1KGP_chr22_ALL.yaml",
    "configs/test_seg128_SGDP_chr22_ALL.yaml",
    "configs/test_seg128_SGDP_chr19_ALL.yaml",
    "configs/test_seg128_HLA_chr6_ALL.yaml",
    "configs/test_seg128_LOS_chr22_ALL.yaml",
    "configs/test_seg128_LOS_chr22_AA.yaml",
    "configs/test_seg128_LOS_chr22_CA.yaml",
]
SPLITS = ("train", "val", "test")


def shape(path):
    """Samples and markers in one split, without loading the genotypes."""
    if not os.path.exists(path):
        return None, None
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").split(",")
        markers = sum(1 for _ in f)
    return len(header) - 1, markers


def split_paths(cfg, split):
    """Where a split lives: one file, or the chunks training reads instead."""
    base = cfg.test_csv_gz.rsplit("_test.csv.gz", 1)[0]
    whole = f"{base}_{split}.csv.gz"
    if os.path.exists(whole):
        return [whole]
    stem = os.path.basename(base)
    return sorted(glob.glob(os.path.join(
        cfg.dataDir, cfg.dataset, "chunked", f"{stem}_{split}_chunk*.csv.gz")))


def main():
    rows = []
    for conf in CONFIGS:
        cfg = ModelConfig.from_yaml(conf)
        row = {"dataset": cfg.dataset, "chromosome": cfg.chromosome,
               "population": cfg.population}
        markers = None
        for split in SPLITS:
            paths = split_paths(cfg, split)
            counts = [shape(p) for p in paths]
            samples = sum(c[0] for c in counts if c[0] is not None) or None
            row[f"n_{split}"] = samples
            if split == "test" and counts:
                markers = counts[0][1]
            if samples is None:
                print(f"  {conf}: {split} split absent")
        row["n_total"] = sum(row[f"n_{s}"] or 0 for s in SPLITS)
        row["n_markers"] = markers
        rows.append(row)
        print(f"  {cfg.dataset} chr{cfg.chromosome} {cfg.population}: "
              f"{row['n_train']}/{row['n_val']}/{row['n_test']} samples, "
              f"{markers} markers")
    frame = pd.DataFrame(rows)
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "dataset_summary.csv")
    frame.to_csv(path, index=False)
    print(f"wrote {path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
