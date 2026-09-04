"""The markers Beagle left uncalled, per frequency bin: Tables S22-S24.

Usage: python scripts/build_ignored_marker_tables.py
Writes Results/beagle_ignored_markers_<region>.csv, one per region.
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
MISSING = "5%"
BIN_ORDER = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%",
             "30%~40%", "40%~50%"]
# label -> the evaluation config whose Beagle cells carry the counts
REGIONS = {
    "SGDP_chr22": "configs/test_seg128_SGDP_chr22_ALL.yaml",
    "HLA_chr6": "configs/test_seg128_HLA_chr6_ALL.yaml",
    "SGDP_chr19": "configs/test_seg128_SGDP_chr19_ALL.yaml",
}


def counts(cfg, method_tag, flags, epoch=None):
    """Markers per bin in one method's cells, or None when they are absent."""
    for rand in RANDS:
        path = analysis_cells.resolve(
            ANALYSIS, cfg.runId, "phased", method_tag, flags, rand,
            cfg.dataset, cfg.chromosome, cfg.population,
            f"seg{cfg.segLen}_overlap{cfg.overlap}", MISSING, epoch=epoch)
        if path is None:
            continue
        df = pd.read_csv(path)
        df["MAF_bin"] = df["MAF_bin"].astype(str).str.strip()
        return df.set_index("MAF_bin")["Num_SNPs"]
    return None


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for label, conf in REGIONS.items():
        cfg = ModelConfig.from_yaml(conf)
        whole = counts(cfg, "biunet", "BAT" if cfg.overlappedOnly else "BAT_FA",
                       epoch=cfg.epoch)
        matched = counts(cfg, "beagle", "BAT")
        if whole is None or matched is None:
            print(f"{label}: cells absent under {ANALYSIS}/")
            continue
        rows = []
        for b in BIN_ORDER + ["Overall"]:
            if b not in whole.index or b not in matched.index:
                continue
            total, kept = int(whole[b]), int(matched[b])
            dropped = total - kept
            rows.append({"MAF_bin": b, "Ignored": dropped,
                         "Total_SNPs_in_bin": total,
                         "Proportion": round(100 * dropped / total, 2) if total else 0.0})
        frame = pd.DataFrame(rows)
        path = os.path.join(OUTDIR, f"beagle_ignored_markers_{label}.csv")
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
