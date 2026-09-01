"""Build Results/exp{1,2}_<dataset>_revision.csv from the per-bin benchmark
CSVs produced by benchmark.py. Output schema matches the original
Results/exp*_*.csv:

  Model, Dataset, Segment_Length, Overlap, Missingness, MAF_bin, Num_SNPs,
  Bin_Acc, Bin_R2, Bin_Precision, Bin_Recall, Bin_F1,
  Overall_Acc, Overall_R2, Overall_Precision, Overall_Recall, Overall_F1
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO = Path(".")
ANALYSIS = REPO / "analysis"
RESULTS = REPO / "Results"

# Original column order from Results/exp*.csv
OUT_COLS = [
    "Model", "Dataset", "Segment_Length", "Overlap", "Missingness",
    "MAF_bin", "Num_SNPs",
    "Bin_Acc", "Bin_R2", "Bin_Precision", "Bin_Recall", "Bin_F1",
    "Overall_Acc", "Overall_R2", "Overall_Precision", "Overall_Recall", "Overall_F1",
]


def per_bin_csvs(method: str, run_id: str, dataset: str, chrom: int, pop: str,
                 segLen: int, overlap: int, rate: int) -> Path | None:
    """Locate one benchmark CSV (per-rate, per-dataset, per-method)."""
    tag = "BAT_PV"  # benchmarkAll=True, perVariantR2=True
    name = (
        f"{run_id}_phased_{method}_{tag}_rand42_"
        f"{dataset}_chr{chrom}_{pop}_seg{segLen}_overlap{overlap}_missing{rate}%.csv"
    )
    p = ANALYSIS / name
    return p if p.exists() else None


def load_one(method: str, run_id: str, dataset: str, chrom: int, pop: str,
             segLen: int, overlap: int, rate: int, model_label: str,
             dataset_label: str) -> pd.DataFrame:
    p = per_bin_csvs(method, run_id, dataset, chrom, pop, segLen, overlap, rate)
    if p is None:
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["MAF_bin"] = df["MAF_bin"].astype(str).str.strip()
    # Drop the Overall row — original tables use "Overall_*" columns instead
    bins = df[df["MAF_bin"] != "Overall"].copy()
    overall = df[df["MAF_bin"] == "Overall"]
    if overall.empty:
        return pd.DataFrame()
    o = overall.iloc[0]
    bins["Model"] = model_label
    bins["Dataset"] = dataset_label
    bins["Segment_Length"] = segLen if model_label != "Beagle" else -1
    bins["Overlap"] = overlap if model_label != "Beagle" else 0
    bins["Missingness"] = rate / 100.0
    bins["Overall_Acc"] = o["Overall Acc"]
    bins["Overall_R2"] = o["Overall R2"]
    bins["Overall_Precision"] = o["Overall Precision"]
    bins["Overall_Recall"] = o["Overall Recall"]
    bins["Overall_F1"] = o["Overall F1"]
    return bins[OUT_COLS]


def build_dataset(run_id: str, dataset: str, chrom: int, pop: str,
                  segLen: int, overlap: int, dataset_label: str,
                  rates: Iterable[int] = (5, 15, 25)) -> pd.DataFrame:
    """Build a single Results table for one dataset/population: BiUNet (v3_low) rows
    + Beagle rows on the same rates/bins."""
    pieces = []
    for rate in rates:
        for method, model_label in [("biunet", "BiUNet"), ("beagle", "Beagle")]:
            df = load_one(method, run_id, dataset, chrom, pop, segLen, overlap,
                          rate, model_label, dataset_label)
            if not df.empty:
                pieces.append(df)
    if not pieces:
        return pd.DataFrame(columns=OUT_COLS)
    out = pd.concat(pieces, ignore_index=True)
    # Add a leading space to MAF_bin to match original formatting
    out["MAF_bin"] = " " + out["MAF_bin"].astype(str).str.lstrip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v3_low_seg128",
                    help="runId prefix in analysis CSVs (default: v3_low_seg128)")
    ap.add_argument("--segLen", type=int, default=128)
    ap.add_argument("--out-suffix", default="_revision",
                    help="suffix for new output files (default: _revision)")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)

    # exp1: dataset-level tables (population=ALL)
    exp1 = [
        ("1KGP", 22, "ALL", "1KGP", 16, "exp1_1KGP"),
        ("SGDP", 22, "ALL", "SGDP", 16, "exp1_SGDP"),
        ("HLA",  6,  "ALL", "HLA",  64, "exp1_HLA"),
        ("LOS",  22, "ALL", "LOS",  16, "exp1_LOS"),
    ]
    # exp2: LOS population-specific (BiUNet trained + tested per-pop)
    exp2 = [
        ("LOS", 22, "AA", "LOS", 16, "exp2_LOS_AA"),
        ("LOS", 22, "CA", "LOS", 16, "exp2_LOS_CA"),
    ]

    for tables in (exp1, exp2):
        for ds, chrom, pop, label, overlap, base_name in tables:
            new_rows = build_dataset(args.run_id, ds, chrom, pop, args.segLen,
                                     overlap, label)
            out_path = RESULTS / f"{base_name}{args.out_suffix}.csv"
            orig_path = RESULTS / f"{base_name}.csv"
            if not orig_path.exists():
                # Just write the BiUNet+Beagle revision rows
                new_rows.to_csv(out_path, index=False)
                print(f"Wrote {out_path} (new file, {len(new_rows)} rows)")
                continue
            # Read original, replace only the BiUNet rows for the new segLen (if any),
            # and append the new BiUNet+Beagle rows from the revision benchmark.
            orig = pd.read_csv(orig_path)
            # Keep original SCDA + original Beagle rows; we re-use the new Beagle
            # numbers from this benchmark for direct comparison at the new bin scheme.
            keep = orig[~orig["Model"].isin(["BiUNet"])].copy()
            # Remove old Beagle rows; we replace with our re-benchmarked Beagle (per-variant r²)
            keep = keep[~keep["Model"].isin(["Beagle"])]
            combined = pd.concat([keep, new_rows], ignore_index=True)
            combined.to_csv(out_path, index=False)
            print(f"Wrote {out_path} ({len(new_rows)} new BiUNet+Beagle rows, "
                  f"{len(keep)} carried over from original)")


if __name__ == "__main__":
    main()
