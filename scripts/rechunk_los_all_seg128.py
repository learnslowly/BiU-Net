"""Re-chunk LOS_ALL train HDF5 files into N evenly-sized output chunks.
"""
import argparse
import glob
import os
import re
import sys
import time

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-glob", required=True,
                   help="Glob for source HDF5 train chunks, e.g. "
                        "'res/LOS_chr22_ALL_train_chunk*_seg128_overlap16.hdf5'")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for new even chunks")
    p.add_argument("--out-prefix", required=True,
                   help="Output filename prefix, e.g. "
                        "'LOS_chr22_ALL_train' "
                        "(suffix _chunk{idx}_seg128_overlap16.hdf5 appended)")
    p.add_argument("--out-suffix", default="_seg128_overlap16",
                   help="Filename suffix between _chunkNNN and .hdf5")
    p.add_argument("--num-chunks", type=int, default=24,
                   help="Number of output chunks (default 24)")
    p.add_argument("--slab-rows", type=int, default=100000,
                   help="Rows per streaming slab read (default 100K)")
    return p.parse_args()


def main():
    args = parse_args()
    src_files = sorted(glob.glob(args.src_glob),
                       key=lambda f: int(re.search(r"chunk(\d+)", f).group(1)))
    if not src_files:
        sys.exit(f"No source files matched {args.src_glob!r}")

    # First pass: total rows + dtypes/shapes
    total = 0
    snps_shape_tail = None
    snps_dtype = None
    idx_shape_tail = None
    idx_dtype = None
    for f in src_files:
        with h5py.File(f, "r") as hf:
            n = hf["snps"].shape[0]
            total += n
            if snps_shape_tail is None:
                snps_shape_tail = hf["snps"].shape[1:]
                snps_dtype = hf["snps"].dtype
                idx_shape_tail = hf["snpsIndex"].shape[1:]
                idx_dtype = hf["snpsIndex"].dtype
        print(f"[scan] {os.path.basename(f)}: {n:,} rows")
    print(f"[scan] TOTAL: {total:,} rows across {len(src_files)} source files")

    # Plan output partitions
    base = total // args.num_chunks
    rem = total - base * args.num_chunks
    chunk_sizes = [base + (1 if i < rem else 0) for i in range(args.num_chunks)]
    print(f"[plan] {args.num_chunks} output chunks, sizes "
          f"{chunk_sizes[0]:,}..{chunk_sizes[-1]:,}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Pre-create output files with full datasets
    out_files = []
    for i, n in enumerate(chunk_sizes):
        out_path = os.path.join(args.out_dir,
                                f"{args.out_prefix}_chunk{i:03d}"
                                f"{args.out_suffix}.hdf5")
        hf = h5py.File(out_path, "w")
        hf.create_dataset("snps",
                          shape=(n,) + snps_shape_tail, dtype=snps_dtype,
                          compression="gzip", chunks=True)
        hf.create_dataset("snpsIndex",
                          shape=(n,) + idx_shape_tail, dtype=idx_dtype,
                          compression="gzip", chunks=True)
        out_files.append({"hf": hf, "path": out_path, "n": n, "written": 0})

    # Streaming copy: walk sources sequentially, fill outputs in order
    out_idx = 0
    t0 = time.time()
    for src_path in src_files:
        with h5py.File(src_path, "r") as src:
            n_src = src["snps"].shape[0]
            slab = args.slab_rows
            for r0 in range(0, n_src, slab):
                r1 = min(r0 + slab, n_src)
                snps_buf = src["snps"][r0:r1]
                idx_buf = src["snpsIndex"][r0:r1]
                rows_left = r1 - r0
                buf_off = 0
                while rows_left > 0:
                    out = out_files[out_idx]
                    cap = out["n"] - out["written"]
                    take = min(cap, rows_left)
                    w_off = out["written"]
                    out["hf"]["snps"][w_off:w_off + take] = \
                        snps_buf[buf_off:buf_off + take]
                    out["hf"]["snpsIndex"][w_off:w_off + take] = \
                        idx_buf[buf_off:buf_off + take]
                    out["written"] += take
                    buf_off += take
                    rows_left -= take
                    if out["written"] >= out["n"]:
                        out["hf"].close()
                        elapsed = time.time() - t0
                        print(f"[write] {os.path.basename(out['path'])}: "
                              f"{out['n']:,} rows (elapsed {elapsed:.0f}s)")
                        out_idx += 1
                        if out_idx >= len(out_files):
                            break
                if out_idx >= len(out_files):
                    break
        if out_idx >= len(out_files):
            break

    # Sanity check
    for out in out_files:
        if out["hf"].id.valid:
            out["hf"].close()
    print(f"[done] {args.num_chunks} chunks written in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
