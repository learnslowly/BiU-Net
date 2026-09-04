#!/usr/bin/env python3
"""Generate manuscript-ready computational complexity summaries.

Example:
    python scripts/complexity_report.py       --configs configs/train_1KGP_chr22_ALL_seg128.yaml                 configs/train_LOS_chr22_ALL_seg128.yaml                 configs/train_SGDP_chr22_ALL_seg128.yaml       --models unet scda       --logs logs/*.log       --out analysis/complexity_report       --deep-hdf5-scan
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.modelconfig import ModelConfig
from data.utils import get_dataset_paths
from model.ae import SCDA
from model.unet import BiUNet


def human_int(x: Optional[int]) -> str:
    if x is None:
        return "NA"
    return f"{x:,}"


def safe_float(x: Optional[float], digits: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_param_memory_bytes(model: torch.nn.Module, dtype_bytes: int = 4) -> int:
    return sum(p.numel() * dtype_bytes for p in model.parameters())


def build_model(config: ModelConfig, model_name: str) -> torch.nn.Module:
    cfg = ModelConfig(**dataclasses.asdict(config))
    model_name = model_name.lower()
    if model_name in {"unet", "biunet", "biu-net"}:
        cfg.model = "unet"
        return BiUNet(cfg)
    if model_name in {"scda", "ae", "autoencoder"}:
        cfg.model = "ae"
        return SCDA(cfg)
    raise ValueError(f"Unsupported model: {model_name}")


def input_channels(config: ModelConfig, model_name: str) -> int:
    model_name = model_name.lower()
    if model_name in {"scda", "ae", "autoencoder"}:
        return int(config.vocabSize)
    if getattr(config, "bioAware", False) and not getattr(config, "useFiLM", False):
        return int(config.vocabSize) + int(getattr(config, "bioChannels", 1))
    return int(config.vocabSize)


def make_dummy_inputs(config: ModelConfig, model_name: str, batch_size: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    length = int(config.segLen)
    channels = input_channels(config, model_name)
    x = torch.zeros((batch_size, length, channels), dtype=torch.float32)
    kwargs: Dict[str, Any] = {}
    if (
        model_name.lower() in {"unet", "biunet", "biu-net"}
        and getattr(config, "bioAware", False)
        and getattr(config, "useFiLM", False)
    ):
        bio_channels = int(getattr(config, "bioChannels", 1))
        kwargs["bio"] = torch.zeros((batch_size, length, bio_channels), dtype=torch.float32)
    return x, kwargs


def hook_flops(model: torch.nn.Module, x: torch.Tensor, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate forward MACs/FLOPs for Conv1d and Linear modules.

    FLOPs are reported as 2 * MACs, the common multiply-add convention. The
    per-module table makes the estimate checkable how it was made.
    """
    records: List[Dict[str, Any]] = []
    handles = []

    def conv_hook(name: str):
        def _hook(module: torch.nn.Conv1d, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            y = output
            batch = int(y.shape[0])
            out_ch = int(y.shape[1])
            out_len = int(y.shape[2])
            in_ch = int(module.in_channels)
            groups = int(module.groups)
            kernel = int(module.kernel_size[0])
            macs = batch * out_len * out_ch * (in_ch // groups) * kernel
            records.append({
                "name": name,
                "type": "Conv1d",
                "output_shape": list(y.shape),
                "macs": int(macs),
                "flops": int(2 * macs),
            })
        return _hook

    def linear_hook(name: str):
        def _hook(module: torch.nn.Linear, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
            y = output
            out_elems = int(y.numel())
            macs = out_elems * int(module.in_features)
            records.append({
                "name": name,
                "type": "Linear",
                "output_shape": list(y.shape),
                "macs": int(macs),
                "flops": int(2 * macs),
            })
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            handles.append(module.register_forward_hook(conv_hook(name)))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook(name)))

    model.eval()
    with torch.no_grad():
        model(x, **kwargs)

    for handle in handles:
        handle.remove()

    total_macs = sum(r["macs"] for r in records)
    total_flops = sum(r["flops"] for r in records)
    return {
        "macs": int(total_macs),
        "flops": int(total_flops),
        "gmacs": total_macs / 1e9,
        "gflops": total_flops / 1e9,
        "records": records,
    }


def benchmark_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    kwargs: Dict[str, Any],
    device: str,
    warmup: int,
    repeat: int,
) -> Dict[str, Optional[float]]:
    if repeat <= 0:
        return {"forward_ms_mean": None, "forward_ms_sd": None, "peak_memory_mb": None}

    model = model.to(device)
    x = x.to(device)
    kwargs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
    model.eval()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            model(x, **kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            model(x, **kwargs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)

    peak_mb = None
    if device.startswith("cuda"):
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "forward_ms_mean": float(np.mean(times)) if times else None,
        "forward_ms_sd": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
        "peak_memory_mb": peak_mb,
    }


def existing_hdf5_paths(config: ModelConfig) -> Tuple[List[str], List[str]]:
    train_files, val_files = get_dataset_paths(config, segmentation=False)
    train_files = [p for p in train_files if os.path.exists(p)]
    val_files = [p for p in val_files if os.path.exists(p)]
    return train_files, val_files


def hdf5_shape_summary(paths: Iterable[str], deep_scan: bool = False) -> Dict[str, Any]:
    total_segments = 0
    segment_lengths = set()
    files = []
    unique_snps = set() if deep_scan else None
    unique_samples = set() if deep_scan else None

    for path in paths:
        with h5py.File(path, "r") as f:
            snps_shape = tuple(f["snps"].shape) if "snps" in f else None
            idx_shape = tuple(f["snpsIndex"].shape) if "snpsIndex" in f else None
            n_segments = int(snps_shape[0]) if snps_shape else 0
            seg_len = int(snps_shape[1]) if snps_shape and len(snps_shape) > 1 else None
            total_segments += n_segments
            if seg_len is not None:
                segment_lengths.add(seg_len)
            file_record = {
                "path": path,
                "snps_shape": snps_shape,
                "snpsIndex_shape": idx_shape,
                "segments": n_segments,
                "attrs": {k: _jsonable_attr(v) for k, v in f.attrs.items()},
            }
            if deep_scan and "snpsIndex" in f:
                arr = f["snpsIndex"]
                # snpsIndex[..., 0] is the locus identifier; the last column is the sample
                # identifier when present.
                for start in range(0, arr.shape[0], 8192):
                    block = arr[start:start + 8192]
                    if block.ndim >= 2 and block.shape[-1] >= 1:
                        snp_ids = block[..., 0].reshape(-1)
                        unique_snps.update(int(v) for v in np.unique(snp_ids) if int(v) >= 0)
                    if block.ndim >= 3 and block.shape[-1] >= 2:
                        sample_ids = block[..., -1].reshape(-1)
                        unique_samples.update(int(v) for v in np.unique(sample_ids) if int(v) >= 0)
            files.append(file_record)

    return {
        "files": files,
        "num_files": len(files),
        "total_segments": total_segments,
        "segment_lengths": sorted(segment_lengths),
        "unique_snps": len(unique_snps) if unique_snps is not None else None,
        "unique_samples": len(unique_samples) if unique_samples is not None and unique_samples else None,
    }


def _jsonable_attr(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def parse_logs(paths: Iterable[str]) -> List[Dict[str, Any]]:
    patterns = {
        "total_samples": re.compile(r"Total samples:\s*([0-9,]+)"),
        "batches_per_epoch": re.compile(r"Batches per epoch:\s*([0-9,]+)"),
        "batch_size": re.compile(r"Batch size:\s*([0-9,]+)"),
        "total_epochs": re.compile(r"Total epochs:\s*([0-9,]+)"),
        "epoch_time": re.compile(r"Epoch\s+(\d+)/(\d+).*?([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)\b", re.I),
        "converged": re.compile(r"converg(?:e|ed|ence).*?(?:epoch\s*)?~?(\d+)", re.I),
        "early_stop": re.compile(r"early stop.*?epoch\s+(\d+)", re.I),
        "best_epoch": re.compile(r"best.*?epoch.*?(\d+)", re.I),
    }
    rows = []
    for path in paths:
        p = Path(path)
        if not p.exists() or not p.is_file():
            continue
        text = p.read_text(errors="ignore")
        row: Dict[str, Any] = {"log": str(p)}
        for key in ("total_samples", "batches_per_epoch", "batch_size", "total_epochs"):
            m = patterns[key].search(text)
            if m:
                row[key] = int(m.group(1).replace(",", ""))
        epoch_times = []
        for m in patterns["epoch_time"].finditer(text):
            epoch_times.append(float(m.group(3)))
        if epoch_times:
            row["epoch_time_s_mean"] = float(np.mean(epoch_times))
            row["epoch_time_s_n"] = len(epoch_times)
        for key in ("converged", "early_stop", "best_epoch"):
            m = patterns[key].search(text)
            if m:
                row[key] = int(m.group(1))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_markdown(path: Path, model_rows: List[Dict[str, Any]], dataset_rows: List[Dict[str, Any]], log_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Computational Complexity Summary\n\n")
        f.write("## Model-Level Table\n\n")
        f.write("| Config | Model | Segment length | Input channels | Parameters | Param memory MB | GFLOPs/segment | Train segments/epoch | Approx train GFLOPs/epoch | Forward ms/batch | Peak memory MB |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in model_rows:
            f.write(
                f"| {r['config']} | {r['model']} | {r['segLen']} | {r['input_channels']} | "
                f"{human_int(r['params_total'])} | {safe_float(r['param_memory_mb'])} | "
                f"{safe_float(r['gflops'], 4)} | {human_int(r.get('train_segments_per_epoch'))} | "
                f"{safe_float(r.get('train_gflops_per_epoch_approx_3x'), 1)} | "
                f"{safe_float(r.get('forward_ms_mean'))} | {safe_float(r.get('peak_memory_mb'))} |\n"
            )
        f.write("\n## Dataset/Scaling Table\n\n")
        f.write("| Config | Split | HDF5 files | Total segments | Unique samples | Unique SNPs | Segment lengths |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for r in dataset_rows:
            f.write(
                f"| {r['config']} | {r['split']} | {r['num_files']} | {human_int(r['total_segments'])} | "
                f"{human_int(r.get('unique_samples'))} | {human_int(r.get('unique_snps'))} | "
                f"{','.join(str(x) for x in r.get('segment_lengths', []))} |\n"
            )
        if log_rows:
            f.write("\n## Parsed Log Table\n\n")
            f.write("| Log | Total samples | Batches/epoch | Batch size | Total epochs | Mean epoch time s | Best epoch | Early stop epoch |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in log_rows:
                f.write(
                    f"| {Path(r['log']).name} | {human_int(r.get('total_samples'))} | "
                    f"{human_int(r.get('batches_per_epoch'))} | {human_int(r.get('batch_size'))} | "
                    f"{human_int(r.get('total_epochs'))} | {safe_float(r.get('epoch_time_s_mean'))} | "
                    f"{human_int(r.get('best_epoch'))} | {human_int(r.get('early_stop'))} |\n"
                )
        f.write("\n## Summary\n\n")
        f.write(
            "For a fixed architecture, BiU-Net processes a genotype matrix by segmenting each sample into "
            "overlapping windows. Let N be the number of training samples, P the number of SNPs, L the "
            "segment length, O the overlap, and S = L - O the stride. The number of training examples per "
            "epoch is approximately N x ceil(P/S), so training and inference scale approximately linearly "
            "with both sample size and SNP count for fixed L and O. The model-level table above reports "
            "parameter counts and per-segment forward-pass MAC/FLOP estimates computed from the actual "
            "PyTorch modules used in this study.\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", required=True, help="YAML training configs to summarize")
    ap.add_argument("--models", nargs="+", default=["unet", "scda"], help="Models to instantiate: unet scda")
    ap.add_argument("--logs", nargs="*", default=[], help="Optional training log files to parse")
    ap.add_argument("--out", default="analysis/complexity_report", help="Output prefix or directory")
    ap.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for model FLOP estimate")
    ap.add_argument("--benchmark-batch-size", type=int, default=0, help="If >0, time this batch size")
    ap.add_argument("--device", default="cpu", help="cpu or cuda for optional timing")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--deep-hdf5-scan", action="store_true", help="Scan snpsIndex to count unique samples/SNPs")
    args = ap.parse_args()

    out_prefix = Path(args.out)
    if out_prefix.suffix:
        out_dir = out_prefix.parent
        stem = out_prefix.stem
    else:
        out_dir = out_prefix
        stem = "complexity_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows: List[Dict[str, Any]] = []
    dataset_rows: List[Dict[str, Any]] = []
    raw: Dict[str, Any] = {"models": [], "datasets": [], "logs": []}

    for cfg_path in args.configs:
        config = ModelConfig.from_yaml(cfg_path)
        config_name = Path(cfg_path).name

        train_files, val_files = existing_hdf5_paths(config)
        train_summary_for_compute: Optional[Dict[str, Any]] = None
        for split, files in [("train", train_files), ("val", val_files)]:
            summary = hdf5_shape_summary(files, deep_scan=args.deep_hdf5_scan)
            if split == "train":
                train_summary_for_compute = summary
            row = {"config": config_name, "split": split, **{k: v for k, v in summary.items() if k != "files"}}
            dataset_rows.append(row)
            raw["datasets"].append({"config": config_name, "split": split, **summary})

        for model_name in args.models:
            model = build_model(config, model_name)
            total_params, trainable_params = count_parameters(model)
            x, kwargs = make_dummy_inputs(config, model_name, args.batch_size)
            flop = hook_flops(model, x, kwargs)
            timing: Dict[str, Optional[float]] = {"forward_ms_mean": None, "forward_ms_sd": None, "peak_memory_mb": None}
            if args.benchmark_batch_size > 0:
                bx, bkwargs = make_dummy_inputs(config, model_name, args.benchmark_batch_size)
                timing = benchmark_forward(
                    model,
                    bx,
                    bkwargs,
                    device=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            row = {
                "config": config_name,
                "model": model_name,
                "segLen": config.segLen,
                "overlap": config.overlap,
                "stride": config.segLen - config.overlap,
                "depth": config.depth,
                "nchannels": config.nchannels,
                "kernelSize": config.kernelSize,
                "bioAware": getattr(config, "bioAware", False),
                "input_channels": input_channels(config, model_name),
                "params_total": total_params,
                "params_trainable": trainable_params,
                "param_memory_mb": estimate_param_memory_bytes(model) / (1024 ** 2),
                **{k: flop[k] for k in ("macs", "flops", "gmacs", "gflops")},
                **timing,
            }
            train_segments = int(train_summary_for_compute["total_segments"]) if train_summary_for_compute else 0
            if train_segments > 0:
                row["train_segments_per_epoch"] = train_segments
                row["forward_gflops_per_epoch"] = row["gflops"] * train_segments
                row["train_gflops_per_epoch_approx_3x"] = row["forward_gflops_per_epoch"] * 3.0
                row["planned_train_gflops_approx_3x"] = row["train_gflops_per_epoch_approx_3x"] * int(config.totalEpochs)
            else:
                row["train_segments_per_epoch"] = None
                row["forward_gflops_per_epoch"] = None
                row["train_gflops_per_epoch_approx_3x"] = None
                row["planned_train_gflops_approx_3x"] = None
            model_rows.append(row)
            raw["models"].append({**row, "module_flops": flop["records"]})

    log_rows = parse_logs(args.logs)
    raw["logs"] = log_rows

    model_fields = [
        "config", "model", "segLen", "overlap", "stride", "depth", "nchannels", "kernelSize",
        "bioAware", "input_channels", "params_total", "params_trainable", "param_memory_mb",
        "macs", "flops", "gmacs", "gflops", "forward_ms_mean", "forward_ms_sd", "peak_memory_mb",
        "train_segments_per_epoch", "forward_gflops_per_epoch", "train_gflops_per_epoch_approx_3x",
        "planned_train_gflops_approx_3x",
    ]
    dataset_fields = [
        "config", "split", "num_files", "total_segments", "unique_samples", "unique_snps", "segment_lengths",
    ]
    log_fields = [
        "log", "total_samples", "batches_per_epoch", "batch_size", "total_epochs",
        "epoch_time_s_mean", "epoch_time_s_n", "best_epoch", "early_stop", "converged",
    ]

    write_csv(out_dir / f"{stem}_models.csv", model_rows, model_fields)
    write_csv(out_dir / f"{stem}_datasets.csv", dataset_rows, dataset_fields)
    write_csv(out_dir / f"{stem}_logs.csv", log_rows, log_fields)
    write_markdown(out_dir / f"{stem}.md", model_rows, dataset_rows, log_rows)
    with (out_dir / f"{stem}.json").open("w") as f:
        json.dump(raw, f, indent=2)

    print(f"Wrote {out_dir / f'{stem}.md'}")
    print(f"Wrote {out_dir / f'{stem}_models.csv'}")
    print(f"Wrote {out_dir / f'{stem}_datasets.csv'}")
    print(f"Wrote {out_dir / f'{stem}_logs.csv'}")
    print(f"Wrote {out_dir / f'{stem}.json'}")


if __name__ == "__main__":
    main()
