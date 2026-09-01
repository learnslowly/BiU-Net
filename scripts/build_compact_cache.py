"""Prebuild .compact.npz sidecar caches for segmented HDF5 files.

Usage: python scripts/build_compact_cache.py <glob> [<glob> ...]
Every glob must match at least one file (fail-loudly, as segmenting.py does).
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SNPsDataset_HDF5

if len(sys.argv) < 2:
    sys.exit("usage: build_compact_cache.py <glob> [<glob> ...]")

files = []
for pattern in sys.argv[1:]:
    hits = sorted(glob.glob(pattern))
    if not hits:
        sys.exit(f"FAILED: pattern matched no files: {pattern}")
    files.extend(hits)

for path in files:
    t0 = time.time()
    ds = SNPsDataset_HDF5(path, preload=True)
    n = len(ds)
    if ds.factorized:
        gb = (ds.snps.numel() + ds.loci.numel() * 8
              + ds.item_loci.numel() * 4 + ds.item_samp.numel() * 8) / 1e9
        print(f"{path}: {n} items, {len(ds.loci)} unique loci rows, "
              f"{gb:.2f} GB compact, {time.time()-t0:.0f}s", flush=True)
    else:
        print(f"{path}: {n} items, FELL BACK to legacy full preload "
              f"(layout assumption violated) — no cache written, "
              f"{time.time()-t0:.0f}s", flush=True)
    ds.close()
print("DONE")
