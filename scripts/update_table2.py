"""Rewrite Table 2 and the sentences that quote it from the measured report.

Usage: python scripts/update_table2.py IN.docx OUT.docx
"""
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd

REPORT = "Results/complexity_report/complexity/complexity_report_models.csv"
CAPTION = "Computational complexity of BiU-Net and SCDA"


def fmt_params(n):
    return f"{n / 1e6:.2f} M" if n < 1e7 else f"{n / 1e6:.1f} M"


def fmt_g(x):
    """Three significant figures, matching how the table already reads."""
    return f"{x:.3g}"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    d = pd.read_csv(REPORT).set_index("model")
    u, s = d.loc["unet"], d.loc["scda"]
    ch = int(u.nchannels)
    if int(s.nchannels) != ch:
        raise SystemExit("the two models were profiled at different widths; "
                         "Table 2 compares them at one configuration")

    rows = {
        "BiU-Net": (u, 1),
        "SCDA": (s, 2),
    }
    ops = []
    for label, (m, row) in rows.items():
        cells = [fmt_params(int(m.params_total)),
                 f"{fmt_g(m.gmacs)} G / {fmt_g(m.gflops)} G",
                 f"~{m.forward_ms_mean:.1f} ms / 256 segments",
                 f"{m.peak_memory_mb:.0f} MB"]
        for col, text in enumerate(cells, start=1):
            ops.append({"op": "replace_table_cell", "match": CAPTION,
                        "row": row, "col": col, "text": text})

    # Edit phrase by phrase: rewriting a whole sentence pulls its cross-reference
    # field into the revision, which Word will not read back.
    def phrase(match, old, replacement):
        ops.append({"op": "replace_phrase", "match": match,
                    "old": old, "new": replacement})

    phrase("Table 2 summarizes the measured computational complexity",
           "48 channels", f"{ch} channels")

    quantified = "We quantified computational complexity for the architecture"
    phrase(quantified,
           "22.0 million parameters compared with 38.5 million",
           f"{int(u.params_total)/1e6:.2f} million parameters compared with "
           f"{int(s.params_total)/1e6:.1f} million")
    phrase(quantified, "0.193 GMACs (0.385 GFLOPs)",
           f"{fmt_g(u.gmacs)} GMACs ({fmt_g(u.gflops)} GFLOPs)")
    phrase(quantified, "0.255 GMACs (0.509 GFLOPs)",
           f"{fmt_g(s.gmacs)} GMACs ({fmt_g(s.gflops)} GFLOPs)")
    phrase(quantified, "approximately 2.4 ms for BiU-Net and 2.7 ms",
           f"approximately {u.forward_ms_mean:.1f} ms for BiU-Net and "
           f"{s.forward_ms_mean:.1f} ms")
    phrase(quantified, "190 MB and 243 MB",
           f"{u.peak_memory_mb:.0f} MB and {s.peak_memory_mb:.0f} MB")

    phrase("remaining smaller than the six-layer SCDA baseline evaluated here",
           "(22.0 million vs. 38.5 million parameters)",
           f"({int(u.params_total)/1e6:.2f} million vs. "
           f"{int(s.params_total)/1e6:.1f} million parameters)")


    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"author": "Revision", "operations": ops}, f, ensure_ascii=False)
        path = f.name
    print(f"Table 2 at {ch} channels: BiU-Net {fmt_params(int(u.params_total))}, "
          f"SCDA {fmt_params(int(s.params_total))}")
    subprocess.run([sys.executable, "scripts/docx_track_edit.py", path, src, dst],
                   check=True)
    os.unlink(path)


if __name__ == "__main__":
    main()
