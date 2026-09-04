"""Per-bin tables for the region-specific models: Results/exp1_*.csv, exp2_*.csv.

Usage: python scripts/build_region_tables.py [table ...]
Writes Results/<table>.csv for each table named, or for all of them.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analysis_cells  # noqa: E402
from config.modelconfig import ModelConfig  # noqa: E402

ANALYSIS, OUTDIR = "analysis", "Results"
RANDS = [0, 42, 1024]
RATES = [("5%", 0.05), ("15%", 0.15), ("25%", 0.25)]
BIN_ORDER = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%",
             "30%~40%", "40%~50%"]
# benchmark.py writes the four overall columns with a space; the published
# tables spell them with an underscore, so the cell name maps to the table name.
CELL_TO_TABLE = {"Bin_Acc": "Bin_Acc", "Bin_R2": "Bin_R2",
                 "Bin_Precision": "Bin_Precision", "Bin_Recall": "Bin_Recall",
                 "Bin_F1": "Bin_F1", "Overall Acc": "Overall_Acc",
                 "Overall R2": "Overall_R2",
                 "Overall Precision": "Overall_Precision",
                 "Overall Recall": "Overall_Recall", "Overall F1": "Overall_F1"}
METRICS = list(CELL_TO_TABLE)
COLUMNS = (["Model", "Dataset", "Segment_Length", "Overlap", "Missingness",
            "MAF_Bin", "Num_SNPs"] + list(CELL_TO_TABLE.values()))

# table -> (config stem shared by the two model configs, Dataset column value)
TABLES = {
    "exp1_1KGP": ("1KGP_chr22_ALL", "1KGP"),
    "exp1_LOS": ("LOS_chr22_ALL", "LOS"),
    "exp1_SGDP": ("SGDP_chr22_ALL", "SGDP"),
    "exp1_HLA": ("HLA_chr6_ALL", "HLA"),
    "exp1_SGDP_chr19": ("SGDP_chr19_ALL", "SGDP_chr19"),
    "exp2_LOS_AA": ("LOS_chr22_AA", "LOS"),
    "exp2_LOS_CA": ("LOS_chr22_CA", "LOS"),
}


def seg_tag(cfg):
    return f"seg{cfg.segLen}_overlap{cfg.overlap}"


def flags(cfg, method_tag):
    """The marker-set flag benchmark.py stamped on this method's cells."""
    if method_tag == "beagle" or cfg.overlappedOnly:
        return "BAT"
    return "BAT_FA"


def cells(cfg, method_tag, miss, epoch):
    """The three seeds of one (method, rate) cell, or None if any is absent."""
    out = []
    for rand in RANDS:
        path = analysis_cells.resolve(
            ANALYSIS, cfg.runId, "phased", method_tag, flags(cfg, method_tag),
            rand, cfg.dataset, cfg.chromosome, cfg.population, seg_tag(cfg),
            miss, epoch=epoch)
        if path is None:
            return None
        df = pd.read_csv(path)
        df["MAF_bin"] = df["MAF_bin"].astype(str).str.strip()
        out.append(df.set_index("MAF_bin"))
    return out


def rows_for(label, cfg, method_tag, dataset, seg, overlap, epoch):
    rows, missing = [], []
    for miss, fraction in RATES:
        frames = cells(cfg, method_tag, miss, epoch)
        if frames is None:
            missing.append(f"{label} {miss}")
            continue
        mean = sum(f[METRICS] for f in frames) / len(frames)
        for b in BIN_ORDER:
            if b not in mean.index:
                continue
            row = {"Model": label, "Dataset": dataset, "Segment_Length": seg,
                   "Overlap": overlap, "Missingness": fraction, "MAF_Bin": b,
                   "Num_SNPs": int(frames[0].loc[b, "Num_SNPs"])}
            row.update({CELL_TO_TABLE[m]: round(float(mean.loc[b, m]), 9)
                        for m in METRICS})
            rows.append(row)
    return rows, missing


def build(table):
    stem, dataset = TABLES[table]
    unet = ModelConfig.from_yaml(os.path.join("configs", f"test_seg128_{stem}.yaml"))
    scda = ModelConfig.from_yaml(os.path.join("configs", f"test_scda_{stem}.yaml"))
    rows, missing = [], []
    # Beagle's cells are written under the BiU-Net evaluation, which is what
    # defines the target and the masking; the imputer itself is unaffected.
    for label, cfg, tag, seg, overlap, epoch in (
            ("Beagle", unet, "beagle", -1, 0, None),
            ("SCDA", scda, "biunet", -1, 0, scda.epoch),
            ("BiUNet", unet, "biunet", unet.segLen, unet.overlap, unet.epoch)):
        got, absent = rows_for(label, cfg, tag, dataset, seg, overlap, epoch)
        rows += got
        missing += absent
    if missing:
        print(f"  WARNING: no cells for {', '.join(missing)}")
    if not rows:
        print(f"  {table}: nothing to write")
        return
    frame = pd.DataFrame(rows, columns=COLUMNS)
    path = os.path.join(OUTDIR, f"{table}.csv")
    frame.to_csv(path, index=False)
    print(f"  wrote {path} ({len(frame)} rows)")


def main():
    wanted = sys.argv[1:] or list(TABLES)
    os.makedirs(OUTDIR, exist_ok=True)
    for table in wanted:
        if table not in TABLES:
            raise SystemExit(f"unknown table {table}; choose from {', '.join(TABLES)}")
        print(table)
        build(table)


if __name__ == "__main__":
    main()
