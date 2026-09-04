"""Rewrite the SCDA rows of Tables S7-S9 from the rebuilt per-bin table.

Usage: python scripts/update_sgdp_scda_tables.py IN.docx OUT.docx
"""
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd

TABLE = "Results/exp1_SGDP.csv"
# caption prefix -> the masking rate its table reports
CAPTIONS = {
    "Table S7. Model's imputation performances on chromosome 22 in the SGDP": 0.05,
    "Table S8. Model's imputation performances on chromosome 22 in the SGDP": 0.15,
    "Table S9. Model's imputation performances on chromosome 22 in the SGDP": 0.25,
}
# the row each metric's SCDA line sits on, and the two columns it is read from
ROWS = {5: ("Bin_Acc", "Overall_Acc"), 8: ("Bin_R2", "Overall_R2"),
        11: ("Bin_Precision", "Overall_Precision"),
        14: ("Bin_Recall", "Overall_Recall"), 17: ("Bin_F1", "Overall_F1")}
BINS = ["0.1%~0.5%", "0.5%~1%", "1%~10%", "10%~20%", "20%~30%", "30%~40%",
        "40%~50%"]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    d = pd.read_csv(TABLE)
    d.columns = [c.strip() for c in d.columns]
    d = d[d.Model == "SCDA"]
    if d.empty:
        raise SystemExit(f"no SCDA rows in {TABLE}")

    ops = []
    for caption, rate in CAPTIONS.items():
        block = d[d.Missingness == rate].set_index("MAF_Bin")
        if len(block) != len(BINS):
            raise SystemExit(f"{TABLE} has {len(block)} bins at {rate}, expected {len(BINS)}")
        for row, (bin_col, overall_col) in ROWS.items():
            values = [f"{float(block.loc[b, bin_col]):.4f}" for b in BINS]
            values.append(f"{float(block.iloc[0][overall_col]):.4f}")
            for offset, text in enumerate(values):
                ops.append({"op": "replace_table_cell", "match": caption,
                            "row": row, "col": 2 + offset, "text": text})

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"author": "Revision", "operations": ops}, f, ensure_ascii=False)
        path = f.name
    print(f"{len(ops)} SCDA cells across Tables S7-S9")
    subprocess.run([sys.executable, "scripts/docx_track_edit.py", path, src, dst],
                   check=True)
    os.unlink(path)


if __name__ == "__main__":
    main()
