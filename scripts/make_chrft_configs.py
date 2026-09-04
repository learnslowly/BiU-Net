"""Write the per-chromosome fine-tuning configs for both datasets.

Usage: python scripts/make_chrft_configs.py
Writes configs/train_{DS}_chrft_{C}.yaml and configs/test_{DS}_chrft_{C}.yaml
"""
import os

CHRS = range(1, 23)
OUT = "configs"

FAMILIES = {
    "1KGP": dict(
        run_id="v3low_ftchr",
        source="configs/train_1KGP_chr22_ft_from_g22.yaml",
        train_on_val=True,
        report_epoch="30",          # fixed schedule, val is in training
        extra=None,
    ),
    "SGDP": dict(
        run_id="ft_sgdp_ftchr",
        source="configs/train_SGDP_chr19_ft_from_full.yaml",
        train_on_val=False,
        report_epoch="best",        # val is held out
        extra="./res/SGDP_chr{c}_ALLPANEL_train_chunk000_seg1024_overlap128.hdf5",
    ),
}

HEADER = """# Per-chromosome fine-tuning of the genome-wide merged-data model, chromosome {c}.
# One of 22 runs differing only in the chromosome. Starting weights are the
# genome-wide checkpoint, prebaked as epoch_0 with optimizer state stripped.
"""


def anchor_keys(src_path):
    """key -> value text from the anchor config, comments and blanks dropped.

    An ordered mapping rather than a list of lines, so that an override cannot
    leave a duplicate key behind: YAML would silently take the later one.
    """
    out = {}
    for line in open(src_path):
        t = line.strip()
        if not t or t.startswith("#") or ":" not in t:
            continue
        k, v = t.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def emit(path, keys, header):
    with open(path, "w") as fh:
        fh.write(header.rstrip() + "\n")
        for k, v in keys.items():
            fh.write(f"{k}: {v}\n")

os.makedirs(OUT, exist_ok=True)
n = 0
for ds, f in FAMILIES.items():
    anchor = anchor_keys(f["source"])
    for c in CHRS:
        k = dict(anchor)
        k.update({"runId": f["run_id"], "dataset": ds, "chromosome": c,
                  "chromosomes": f"[{c}]", "trainOnVal": f["train_on_val"],
                  "totalEpochs": 30, "epoch": 0})
        if f["extra"]:
            k["extraTrainFiles"] = f["extra"].format(c=c)
        else:
            k.pop("extraTrainFiles", None)
        emit(os.path.join(OUT, f"train_{ds}_chrft_{c}.yaml"), k,
             HEADER.format(c=c, ds=ds))
        n += 1

        # Evaluation reuses the same settings with the reporting epoch pinned
        # and the seven-bin frequency convention the tables use.
        t = dict(k)
        t.update({"epoch": f["report_epoch"],
                  "bins": "[0.001, 0.005, 0.010, 0.100, 0.200, 0.300, 0.400, 0.500]",
                  "missing": "[0.05, 0.15, 0.25]",
                  "perVariantR2": False, "benchmarkAll": True})
        emit(os.path.join(OUT, f"test_{ds}_chrft_{c}.yaml"), t,
             HEADER.format(c=c, ds=ds)
             + f"# Evaluation copy: reporting checkpoint pinned to epoch {f['report_epoch']}.")
        n += 1
print(f"wrote {n} configs into {OUT}/")
