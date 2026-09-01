import os

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

# HDF5 layout (segmenting.py): 'snps' is (N, L) int64 genotype codes
# (0..vocabSize-1, padId=5); 'snpsIndex' is (N, L, 2) int64 [locus, sample_id]
# pairs. Within one item the sample_id column is constant, and the locus row is
# one of only ~num_segments distinct windows repeated across samples — the
# factorized preload below stores each unique locus row once instead of
# materializing N*L*2 int64 (which made genome-wide preload impossible:
# ~137 GB/rank for 1KGP chr1-22 against 17 GB/rank factorized).
_BLOCK = 2048


class SNPsDataset_HDF5(Dataset):
    def __init__(self, hdf5_filename: str, preload=True):
        self.hdf5_filename = hdf5_filename
        self.preload = preload
        self.data_file = None  # For lazy loading
        self.factorized = False

        if self.preload:
            self._load_compact()
        else:
            with h5py.File(self.hdf5_filename, 'r') as f:
                self.length = len(f['snps'])

    # ---- compact preload -------------------------------------------------
    def _cache_path(self):
        return self.hdf5_filename + ".compact.npz"

    def _try_load_cache(self, src_stat):
        cache = self._cache_path()
        if not os.path.exists(cache):
            return False
        try:
            z = np.load(cache)
            if (int(z["src_size"]) != src_stat.st_size
                    or int(z["src_mtime"]) != int(src_stat.st_mtime)):
                return False
            self.snps = torch.from_numpy(z["snps"])
            self.loci = torch.from_numpy(z["loci"])
            self.item_loci = torch.from_numpy(z["item_loci"])
            self.item_samp = torch.from_numpy(z["item_samp"])
        except Exception:
            return False
        self.length = len(self.snps)
        self.factorized = True
        return True

    def _load_compact(self):
        src_stat = os.stat(self.hdf5_filename)
        if self._try_load_cache(src_stat):
            return
        with h5py.File(self.hdf5_filename, 'r') as f:
            n, seg_len = f['snps'].shape
            snps = np.empty((n, seg_len), dtype=np.int8)
            ok = True
            for s in range(0, n, _BLOCK):
                blk = f['snps'][s:s + _BLOCK]
                if blk.size and (blk.min() < -128 or blk.max() > 127):
                    ok = False
                    break
                snps[s:s + len(blk)] = blk.astype(np.int8)

            uniq = {}
            loci_rows = []
            item_loci = np.empty(n, dtype=np.int32)
            item_samp = np.empty(n, dtype=np.int64)
            if ok:
                si = f['snpsIndex']
                for s in range(0, n, _BLOCK):
                    blk = si[s:s + _BLOCK]
                    samp_col = blk[:, :, 1]
                    # factorization requires a constant sample id per item
                    if not (samp_col == samp_col[:, :1]).all():
                        ok = False
                        break
                    item_samp[s:s + len(blk)] = samp_col[:, 0]
                    for j, row in enumerate(blk[:, :, 0]):
                        key = row.tobytes()
                        rid = uniq.get(key)
                        if rid is None:
                            rid = len(loci_rows)
                            uniq[key] = rid
                            loci_rows.append(row)
                        item_loci[s + j] = rid

            if not ok:
                # Layout assumption violated (or codes out of int8 range):
                # fall back to the legacy full preload, correctness first.
                self.snps = torch.from_numpy(f['snps'][:].astype('int16'))
                self.snpsIndex = torch.from_numpy(f['snpsIndex'][:].astype('int32'))
                self.length = len(self.snps)
                self.factorized = False
                return

        self.snps = torch.from_numpy(snps)
        self.loci = torch.from_numpy(np.stack(loci_rows))
        self.item_loci = torch.from_numpy(item_loci)
        self.item_samp = torch.from_numpy(item_samp)
        self.length = n
        self.factorized = True
        self._write_cache(src_stat)

    def _write_cache(self, src_stat):
        # Atomic last-wins write: concurrent DDP ranks may build simultaneously;
        # every writer produces identical content, os.replace keeps it valid.
        cache = self._cache_path()
        tmp = f"{cache}.tmp.{os.getpid()}"
        try:
            np.savez(tmp, snps=self.snps.numpy(), loci=self.loci.numpy(),
                     item_loci=self.item_loci.numpy(),
                     item_samp=self.item_samp.numpy(),
                     src_size=np.int64(src_stat.st_size),
                     src_mtime=np.int64(int(src_stat.st_mtime)))
            os.replace(tmp + ".npz", cache)
        except OSError:
            # cache is an optimization only — never fail the run over it
            for p in (tmp + ".npz", tmp):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    # ---------------------------------------------------------------------
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.preload:
            snps = self.snps[idx].long()
            if self.factorized:
                loci = self.loci[self.item_loci[idx]].long()
                samp = torch.full_like(loci, int(self.item_samp[idx]))
                return snps, torch.stack([loci, samp], dim=-1)
            return snps, self.snpsIndex[idx].long()

        if self.data_file is None:
            self.data_file = h5py.File(self.hdf5_filename, 'r')

        snps = torch.from_numpy(self.data_file['snps'][idx]).long()
        snpsIndex = torch.from_numpy(self.data_file['snpsIndex'][idx]).long()
        return snps, snpsIndex

    def __del__(self):
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()

    def close(self):
        if hasattr(self, 'data_file') and self.data_file is not None:
            self.data_file.close()
            self.data_file = None
        for attr in ('snps', 'snpsIndex', 'loci', 'item_loci', 'item_samp'):
            if hasattr(self, attr):
                delattr(self, attr)

