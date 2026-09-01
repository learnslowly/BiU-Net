"""Bio-info lookup for the bioAware input channel(s).

Encodings (config.bioEncoding):
  - 'normPos'           : legacy — per-segment (POS - min)/(max - min)  [1 channel]
  - 'cM_gap'            : cM gap to previous SNP, log1p-scaled          [1 channel]
  - 'cM_norm'           : cM coordinate normalized to chromosome length [1 channel]
  - 'maf'               : per-SNP minor allele frequency                [1 channel]
  - 'cM_gap+maf'        : cM_gap + maf as two channels                  [2 channels]
  - 'cM_norm+maf'       : cM_norm + maf as two channels                 [2 channels]
  - 'cM_gap+cM_norm+maf': three channels                                [3 channels]
"""
from __future__ import annotations
import os
import numpy as np
import torch


_CACHE = {}


def _load_table(path):
    if path in _CACHE:
        return _CACHE[path]
    data = np.load(path)
    table = {k: data[k] for k in data.files}
    # build pos -> idx map for O(log n) lookup via np.searchsorted (table is sorted by pos)
    _CACHE[path] = table
    return table


def encoding_n_channels(encoding: str) -> int:
    if encoding == 'normPos':
        return 1
    parts = encoding.split('+')
    return len(parts)


def compute_bio_channels(batch_pos_bp: torch.Tensor, encoding: str, bio_file: str | None) -> torch.Tensor:
    """Return tensor of shape (B, L, C) where C = encoding_n_channels(encoding).

    batch_pos_bp: integer tensor (B, L) of bp positions for each SNP in the batch.
    """
    device = batch_pos_bp.device
    B, L = batch_pos_bp.shape

    if encoding == 'normPos':
        # Legacy: per-segment normalization of bp positions
        bp_f = batch_pos_bp.float()
        bp_min = bp_f.min(dim=1, keepdim=True)[0]
        bp_max = bp_f.max(dim=1, keepdim=True)[0]
        denom = (bp_max - bp_min).clamp(min=1.0)
        return ((bp_f - bp_min) / denom).unsqueeze(-1)  # (B, L, 1)

    if bio_file is None or not os.path.exists(bio_file):
        raise FileNotFoundError(f"bio_file not found: {bio_file}")
    table = _load_table(bio_file)
    pos_arr = table['pos']  # sorted int64

    parts = encoding.split('+')
    flat_pos = batch_pos_bp.detach().cpu().numpy().reshape(-1)
    # locate each bp in table; clamp out-of-range to 0 (padding positions will be e.g. -1)
    idx = np.searchsorted(pos_arr, flat_pos)
    # Safe-clip indices
    idx_clip = np.clip(idx, 0, len(pos_arr) - 1)
    valid = (idx >= 0) & (idx < len(pos_arr)) & (pos_arr[idx_clip] == flat_pos)

    chans = []
    for part in parts:
        if part == 'cM_gap':
            v = table['cM_gap']
            # Log1p compression: cM_gap is very small (median 4e-5). log1p(x*1e3) gives a usable range.
            v_chan = np.log1p(v * 1000.0).astype(np.float32)
        elif part == 'cM_norm':
            v_chan = table['cM_norm'].astype(np.float32)
        elif part == 'maf':
            v_chan = table['maf'].astype(np.float32)
        elif part == 'cM_abs':
            v_chan = (table['cM_abs'] / 75.0).astype(np.float32)  # ~chr22 length
        else:
            raise ValueError(f"Unknown bio encoding part: {part}")

        # Gather by idx; invalid positions get 0
        gathered = np.where(valid, v_chan[idx_clip], 0.0)
        chans.append(gathered.reshape(B, L))

    bio = np.stack(chans, axis=-1)  # (B, L, C)
    return torch.from_numpy(bio).to(device=device, dtype=torch.float32)
