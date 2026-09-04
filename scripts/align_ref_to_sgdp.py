"""Project 1KGP test genotypes onto the SGDP variant axis, per chromosome.

Writes data/SGDP/chunked/SGDP_chr{C}_ALLREF_train_chunk000.csv.gz — the ALLREF
population tag routes the standard segmenting pipeline to produce aligned
seg1024 HDF5s consumed via extraTrainFiles.
"""
import gzip
import os
import sys

import numpy as np
import pandas as pd

# The cohort trees live outside the repository; GENOTYPE_DATA_DIR names where,
# and configs/credentials.sh sets it alongside the other site-specific values.
DATA = os.environ.get("GENOTYPE_DATA_DIR", "./data")
SGDP = f"{DATA}/SGDP"
KGP = f"{DATA}/1KGP"

full = len(sys.argv) > 1 and sys.argv[1] == "full"
# optional resume point: chromosomes below it are already projected (a file
# whose "chrN: ..." line never reached the log is partial and gets redone)
start_chr = int(sys.argv[2]) if len(sys.argv) > 2 else 1

for c in range(start_chr, 23):
    sgdp_axis = pd.read_csv(f"{SGDP}/split/SGDP_chr{c}_ALL_test.csv.gz",
                            compression="gzip", index_col=0, usecols=[0]).index
    subsets = ("train", "val", "test") if full else ("test",)
    # int32 dtype keeps the big-chromosome reads ~5x smaller than the
    # object-dtype default, which exhausts memory on a chromosome-sized frame.
    ref = pd.concat([pd.read_csv(f"{KGP}/split/1KGP_chr{c}_ALL_{sub}.csv.gz",
                                 compression="gzip", index_col=0, dtype=np.int32)
                     for sub in subsets], axis=1)
    aligned = ref.reindex(sgdp_axis).fillna(0).astype(np.int8)
    shared = int(ref.index.isin(sgdp_axis).sum())
    tag = "ALLPANEL" if full else "ALLREF"
    out = f"{SGDP}/chunked/SGDP_chr{c}_{tag}_train_chunk000.csv.gz"
    with gzip.open(out, "wt") as f:
        aligned.to_csv(f, index=True)
    print(f"chr{c}: SGDP axis {len(sgdp_axis)}, shared with 1KGP {shared} "
          f"({shared/len(sgdp_axis):.1%}), ref samples {ref.shape[1]} -> {out}",
          flush=True)
print("DONE")
