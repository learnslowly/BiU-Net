"""Locate benchmark cell CSVs under analysis/, across their filename spellings.
"""
import glob
import os
import re

# An epoch tag is a number or the literal .best.; .best. sorts above every number.
EPOCH_RE = re.compile(r"_epoch(\d+|best)_phased_")

# Maps a run.s reported name to the name its cells and checkpoints carry.
LEGACY_RUN_IDS = {
    "1KGP_genomewide": "v3low_ref_g22",
    "1KGP_finetuned": "v3low_ftchr",
    "1KGP_finetuned_anchor": "v3low_chr22_ft",
    "1KGP_scda_genomewide": "scda_fl_g22",
    "SGDP_genomewide": "ft_sgdp_full",
    "SGDP_finetuned": "ft_sgdp_ftchr",
    "SGDP_finetuned_anchor": "ft_sgdp_chr19_ft",
    "SGDP_scda_genomewide": "scda_sgdp_scratch",
    "SGDP_ablation_noreference": "ft_sgdp_noref",
    "SGDP_ablation_unaligned": "ft_sgdp_g22",
    "SGDP_ablation_aligned": "ft_sgdp_aln",
}


def _epoch_key(path):
    m = EPOCH_RE.search(os.path.basename(path))
    if not m:
        return -1
    tag = m.group(1)
    return float("inf") if tag == "best" else int(tag)


def resolve(analysis_dir, run_id, kind, method_tag, flags, rand_state,
            dataset, chrom, population, seg_tag, miss, epoch=None):
    """Path of one benchmark cell, or None when it was never written.

    epoch=None takes the highest-numbered epoch present (the furthest-trained
    checkpoint of that run), falling back to the untagged legacy file.
    Passing an explicit epoch pins that checkpoint and never falls back, so a
    table can name exactly the model it reports.
    """
    tail = (f"_{kind}_{method_tag}_{flags}_rand{rand_state}_{dataset}"
            f"_chr{chrom}_{population}_{seg_tag}_missing{miss}.csv")
    for name in (run_id, LEGACY_RUN_IDS.get(run_id)):
        if name is None:
            continue
        hit = _under(analysis_dir, name, tail, epoch)
        if hit:
            return hit
    return None


def _under(analysis_dir, run_id, tail, epoch):
    """Path of one cell written under this exact runId, or None."""
    if epoch is not None:
        p = os.path.join(analysis_dir, f"{run_id}_epoch{epoch}{tail}")
        return p if os.path.exists(p) else None

    hits = glob.glob(os.path.join(analysis_dir, f"{run_id}_epoch*{tail}"))
    hits = [h for h in hits if EPOCH_RE.search(os.path.basename(h))]
    if hits:
        return max(hits, key=_epoch_key)

    untagged = os.path.join(analysis_dir, f"{run_id}{tail}")
    return untagged if os.path.exists(untagged) else None


def describe(paths):
    """One-line provenance summary of which spellings a table consumed."""
    epochs, legacy = set(), 0
    for p in paths:
        if not p:
            continue
        m = EPOCH_RE.search(os.path.basename(p))
        if m:
            epochs.add(m.group(1))
        else:
            legacy += 1
    parts = []
    if epochs:
        parts.append("epoch-tagged cells from epoch(s) "
                     + ", ".join(sorted(epochs, key=lambda e: (e == "best", e))))
    if legacy:
        parts.append(f"{legacy} untagged (pre-2026-08-24) cells")
    return "; ".join(parts) if parts else "no cells resolved"

# A cell where the model predicted a constant scores near-perfect accuracy and
# zero correlation at the same time, because R2 has no variance to explain.
# Real data never produces that pair, so it marks a degenerate imputation rather
# than a hard chromosome, and averaging it in would quietly depress a bin.
DEGENERATE_ACC = 0.99
DEGENERATE_R2 = 1e-6


def degenerate_rows(df):
    """Rows of one benchmark cell that show the constant-prediction signature."""
    if "Bin_Acc" not in df.columns or "Bin_R2" not in df.columns:
        return df.iloc[0:0]
    return df[(df["Bin_Acc"] > DEGENERATE_ACC) & (df["Bin_R2"] < DEGENERATE_R2)]


def check(path, df):
    """Return a description of the problem with this cell, or None if it is sound."""
    bad = degenerate_rows(df)
    if len(bad) == 0:
        return None
    bins = ", ".join(str(b).strip() for b in bad["MAF_bin"])
    return (f"{os.path.basename(path)}: constant prediction in {len(bad)} bin(s) "
            f"({bins}); accuracy above {DEGENERATE_ACC} with R2 below {DEGENERATE_R2}")
