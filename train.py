#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Pin one GPU per task before torch imports CUDA.
if 'SLURM_LOCALID' in os.environ:
    local_rank = int(os.environ['SLURM_LOCALID'])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

import logging
import time
import random
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from functools import partial
from dataclasses import asdict
from datetime import datetime
import math

# Silence dynamo's "Profiler function ... will be ignored" warning.
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

from config.modelconfig import ModelConfig
from data.utils import (
    get_dataset_paths,
    mask_random_positions,
    mask_random_positions_bias,
    one_hot_encode,
    find_latest_checkpoint,
    save_checkpoint,
    cleanup_memory,
    FocalLoss,
    HybridFocalLoss,
    WeightedFocalLoss,
    HybridWeightedFocalLoss,
    F1Loss
)
from data.dataset import SNPsDataset_HDF5
from model.ae import print_model_summary
import wandb

def get_optimal_num_workers():
    # Slurm tells each process how many CPUs it *should* use
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    # Reserve 1 CPU for the main DDP process
    return max(1, cpus_per_task - 1)

def print_exp_summary(use_gpu, world_size, rank, backend, config, train_hdf5_files, val_hdf5_files, latest_checkpoint_file):
    print("============= TRAINING CONFIGURATION =============")
    print("Is CUDA available:", use_gpu)
    if use_gpu:
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "No CUDA device")
        print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Training on CPU nodes")

    print(f"World size: {world_size}, Rank: {rank}, Backend: {backend}")

    print("\n============= MODEL CONFIGURATION =============")
    for field in vars(config):
        value = getattr(config, field)
        # Print arrays more compactly
        if isinstance(value, list) and len(value) > 10:
            print(f"{field}: [{value[0]}, {value[1]}, ..., {value[-1]}] (length {len(value)})")
        else:
            print(f"{field}: {value}")

    print("\n============= DATASET INFORMATION =============")
    print("Train files:")
    for file in train_hdf5_files:
        print(f"  - {file}")
    print("Validation files:")
    for file in val_hdf5_files:
        print(f"  - {file}")

    if config.finetuning and config.targetBin is not None:
        print(f"\nWill use SNPs in range: {config.targetBin} for finetuning.")

    total_train_samples = get_num_samples(train_hdf5_files)
    num_batches_per_epoch = total_train_samples // config.batchSize
    total_steps = num_batches_per_epoch * config.totalEpochs
    batches_per_device_per_epoch = num_batches_per_epoch // max(1, world_size)
    print(f"\n============= TRAINING STATISTICS =============")
    print(f"Total samples: {total_train_samples:,}")
    print(f"Batches per device in one epoch: {batches_per_device_per_epoch:,.1f}")
    print(f"Batch size: {config.batchSize}")
    print(f"Batches per epoch: {num_batches_per_epoch:,}")
    print(f"Total epochs: {config.totalEpochs}")
    print(f"Total training steps: {total_steps:,}")
    print(f"Save checkpoint frequency: Every {config.saveCheckpointFreq} epoch(s)")
    print(f"Profiling batches: {config.maxProfilingBatches}")

    if latest_checkpoint_file:
        print(f"\nLoading checkpoint from '{latest_checkpoint_file}'")
    else:
        print("\nStarting fresh training run without checkpoint")

    print("=================================================")

def get_optimal_bucket_size(num_gpus):
    # One gradient bucket: a single all-reduce per step.
    if num_gpus <= 8:
        return 128
    else:
        return 256

def trace_handler_with_log(dir, experiment_id):
    def handler(prof):
        global os
        print(f"TRACE SAVED 🚀 - Writing to: {dir}")
        # Make sure the directory exists
        os.makedirs(dir, exist_ok=True)

        try:
            # Export temporary Chrome trace
            temp_path = os.path.join(dir, "temp_trace.json")
            prof.export_chrome_trace(temp_path)

            # Read the trace and convert to proper Chrome format
            import json
            try:
                with open(temp_path, 'r') as f:
                    trace_data = json.load(f)

                # Chrome trace format expects a dict with 'traceEvents' key
                if isinstance(trace_data, list):
                    chrome_data = {"traceEvents": trace_data}
                else:
                    chrome_data = trace_data

                # Write the properly formatted Chrome trace
                chrome_path = os.path.join(dir, f"{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(chrome_path, 'w') as f:
                    json.dump(chrome_data, f)

                # Remove the temporary file
                import os
                os.remove(temp_path)

                print(f"Converted trace saved to: {chrome_path}")
            except Exception as e:
                print(f"Error converting trace format: {e}")
                import traceback
                traceback.print_exc()

            # Continue with TensorBoard writing...
        except Exception as e:
            print(f"Error in trace handler: {e}")
            import traceback
            traceback.print_exc()
    return handler


def lr_lambda(config: ModelConfig, current_epoch: int) -> float:
    warmup_epochs = config.warmupEpochs
    total_epochs = config.totalEpochs
    cooldown_epochs = config.cooldownEpochs
    scheduler_type = config.scheduler

    # Warmup Phase: Linearly increase LR from 0% to 100% of base LR
    if current_epoch < warmup_epochs:
        return float(current_epoch + 1) / float(max(1, warmup_epochs))

    # Cooldown Phase: Linearly decrease LR
    elif current_epoch > total_epochs - cooldown_epochs:
        cooldown_epoch = current_epoch - (total_epochs - cooldown_epochs)
        return float(cooldown_epochs - cooldown_epoch) / float(max(1, cooldown_epochs))

    # Main Training Phase
    else:
        if scheduler_type == "cosineAnn":
            # Cosine annealing from epoch warmup_epochs to total_epochs - cooldown_epochs
            cosine_epoch = current_epoch - warmup_epochs
            cosine_total = max(1, total_epochs - warmup_epochs - cooldown_epochs)
            return 0.5 * (1 + math.cos(math.pi * cosine_epoch / cosine_total))

        elif scheduler_type == "stepLR":
            # Step decay
            step_size = max(1, config.schedulerStepSize)
            step_factor = (current_epoch - warmup_epochs) // step_size
            return config.schedulerGamma ** step_factor

        # Default: Keep LR unchanged
        return 1.0

def aggregate_scalar(value, device):
    """
    Aggregate values across all processes.
    Works with both scalar values and tensor values.
    """
    if isinstance(value, torch.Tensor):
        # If it's already a tensor, make a clone to avoid modifying the original
        tensor = value.clone().detach()
    else:
        # If it's a scalar, convert to tensor
        tensor = torch.tensor(value, device=device)

    # Ensure the tensor is on the correct device
    tensor = tensor.to(device)

    # Perform the all-reduce operation
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Return the tensor itself, not the scalar value
    return tensor


def mask_random_columns(config, data_batch, bert_strategy=False):
    """Mask whole variant columns across every row of the batch."""
    B, S = data_batch.shape
    # Per-batch rate: drawn in (colwiseRateMin, missingRatio] when
    # colwiseDynamicRange is set, else missingRatio.
    if getattr(config, 'colwiseDynamicRange', False):
        lo = float(getattr(config, 'colwiseRateMin', 0.0))
        rate = lo + float(torch.rand(1).item()) * (config.missingRatio - lo)
    else:
        rate = float(config.missingRatio)
    n_target = int(round(rate * S))
    # Mask the same columns for every row in the batch. Ranking a random
    # score fixes the count exactly.
    scores = torch.rand(S, device=data_batch.device)
    maf_min = float(getattr(config, 'colwiseMafMin', 0.0))
    if maf_min > 0.0:
        # Sort columns below the frequency threshold first so they are masked
        # before any column above it.
        valid_cols = ((data_batch >= 1) & (data_batch < config.vocabSize - 1))
        alt = (((data_batch == 2) | (data_batch == 3)).to(torch.float32)
               + 2.0 * (data_batch == 4).to(torch.float32))
        n_valid = valid_cols.sum(dim=0).clamp(min=1).to(torch.float32)
        af = alt.sum(dim=0) / (2.0 * n_valid)
        maf = torch.minimum(af, 1.0 - af)
        scores = torch.where(maf < maf_min, scores - 1.0, scores)
    col_mask = torch.zeros(S, dtype=torch.bool, device=data_batch.device)
    if n_target > 0:
        idx = torch.topk(scores, k=min(n_target, S), largest=False).indices
        col_mask[idx] = True
    mask = col_mask.unsqueeze(0).expand(B, S)
    # Only apply to valid genotype positions (mirror mask_random_positions).
    valid_positions = (data_batch >= 1) & (data_batch < config.vocabSize - 1)
    mask = mask & valid_positions
    masked_data = data_batch.clone()
    masked_data[mask] = config.missingId
    return masked_data, mask


def mask_chip_columns(config, data_batch, snps_index, chip_lut):
    """Mask the positions absent from the array site list."""
    pos = snps_index[:, :, 0].clamp(min=0, max=chip_lut.numel() - 1)
    on_chip = chip_lut[pos]
    valid_positions = (data_batch >= 1) & (data_batch < config.vocabSize - 1)
    mask = (~on_chip) & valid_positions
    masked_data = data_batch.clone()
    masked_data[mask] = config.missingId
    return masked_data, mask


def load_chip_lut(config, device):
    """Boolean lookup over genomic position: True where the array has a site."""
    import numpy as _np
    path = config.chipSitesFile
    positions = []
    with open(path) as f:
        for line in f:
            if line.startswith('chrom') or not line.strip():
                continue
            positions.append(int(line.split()[1]))
    pos = _np.asarray(positions, dtype=_np.int64)
    lut = torch.zeros(int(pos.max()) + 1, dtype=torch.bool, device=device)
    lut[torch.from_numpy(pos).to(device)] = True
    return lut


def _model_forward(mdl, x, bio_arg, model_type):
    """Call the model, passing `bio` only to architectures that accept it."""
    if model_type == 'scda':
        return mdl(x)
    return mdl(x, bio=bio_arg)


def train_ddp(use_gpu, rank, world_size, config, train_hdf5_files, val_hdf5_files, checkpoint_file):
    """
    Distributed training function with optional profiling.

    Args:
        use_gpu: Is training on GPU nodes
        rank: Process rank
        world_size: Total number of processes
        config: Model configuration
        train_hdf5_files: List of training HDF5 files
        val_hdf5_files: List of validation HDF5 files
        checkpoint_file: Path to checkpoint file or None
    """
    # Set random seeds for reproducibility
    seed = config.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if use_gpu:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'

    if rank == 0:
        print_exp_summary(use_gpu, world_size, rank, backend, config, train_hdf5_files, val_hdf5_files, checkpoint_file)

    # Set RANK/WORLD_SIZE env vars (the names env:// init_method looks for).
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Initialize process group using SLURM environment
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    if config.segLen == -1:
        with h5py.File(val_hdf5_files[0], 'r') as f:
            config.segLen = f['snps'].shape[1]

    # Initialize model
    if config.model == 'unet':
        from model.unet import BiUNet
        model = BiUNet(config).to(device)
    elif config.model == 'scda':
        from model.ae import SCDA
        model = SCDA(config).to(device)

    if checkpoint_file == None and rank == 0:
        print_model_summary(model)

    # Different DDP initialization based on device type
    if device.type == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],  # Changed to use current device
            bucket_cap_mb=get_optimal_bucket_size(world_size)
        )
    else:
        # For CPU, don't specify device_ids or output_device
        model = torch.nn.parallel.DistributedDataParallel(
            model
        )
    _cfg_workers = int(getattr(config, 'dataLoaderWorkers', 0) or 0)
    num_workers = _cfg_workers if _cfg_workers > 0 else get_optimal_num_workers()

    # Auto mixed precision - only for GPU
    scaler = None
    if config.mixedPrecisionTraining and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')

    # Initialize wandb only for process 0
    if config.useWandB and rank == 0 and wandb is not None:
        os.environ["WANDB_API_KEY"] = config.WandBKey
        wandb.init(project=config.WandBProjName, name=config.run, config=asdict(config), resume='allow', id=config.runId, mode='offline')
        wandb.watch(model, log_freq=100)

    criterion = {
        "crossEntropy": nn.CrossEntropyLoss(),
        "focalLoss": FocalLoss(config=config),
        "hybridFocalLoss": HybridFocalLoss(config=config),
        "weightedFocalLoss": WeightedFocalLoss(config=config),
        "hybridWeightedFocalLoss": HybridWeightedFocalLoss(config=config),
        "f1Loss": F1Loss(config=config)
    }[config.loss]

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learningRate,
        betas=(config.adamwBeta1, config.adamwBeta2),
        eps=config.adamwEps,
        weight_decay=config.adamwWeightDecay
    )

    # Load checkpoint if available
    checkpoint = None
    if checkpoint_file and os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        # Optimizer state is optional.
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        current_epoch = checkpoint['epoch'] + 1

        # Pin lr and initial_lr to the configured value on resume.
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.learningRate
            param_group["initial_lr"] = config.learningRate
    else:
        # No checkpoint: initialize from scratch
        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = param_group["lr"]
        current_epoch = 1

    # Set last_epoch correctly for the scheduler
    last_epoch_val = -1 if current_epoch == 1 else current_epoch - 1

    # Create the LambdaLR scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(lr_lambda, config),
        last_epoch=last_epoch_val
    )

    # Restore scheduler state if it was saved
    if checkpoint is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Compile after loading the state dict, else the _orig_mod. prefix
    # does not match a non-compiled checkpoint.
    if config.useTorchCompile and device.type == 'cuda':
        model = torch.compile(model, mode=config.torchCompileMode)
        if rank == 0:
            print(f"torch.compile enabled (mode={config.torchCompileMode})")

    # One ConcatDataset loader so workers persist for the whole epoch.
    from torch.utils.data import ConcatDataset

    def _make_loader(hdf5_files, *, shuffle: bool, drop_last: bool = True):
        children = [SNPsDataset_HDF5(f, preload=config.preloadData) for f in hdf5_files]
        # Group files by sequence length: with segLen=-1 each chromosome has its
        # own length and a batch cannot mix them.
        import h5py as _h5py
        by_len = {}
        for f, child in zip(hdf5_files, children):
            with _h5py.File(f, 'r') as h:
                seq_len = h['snps'].shape[1]
            by_len.setdefault(seq_len, []).append(child)
        samplers, loaders = [], []
        # Per-rank positions budget: bucket_batch = budget // seq_len, capped at
        # config.batchSize.
        pos_budget = int(getattr(config, 'maxBatchPositions', 0) or 0)
        for seq_len in sorted(by_len):
            combined = ConcatDataset(by_len[seq_len])
            bucket_batch = config.batchSize
            if pos_budget > 0:
                bucket_batch = max(4, min(config.batchSize, pos_budget // seq_len))
            # Disable drop_last when the per-rank sample count is below batchSize,
            # otherwise the loader yields no batches.
            per_rank = len(combined) // max(1, world_size)
            group_drop_last = drop_last and (per_rank >= bucket_batch)
            sampler = DistributedSampler(combined, num_replicas=world_size, rank=rank,
                                         shuffle=shuffle, drop_last=group_drop_last)
            if rank == 0 and pos_budget > 0 and bucket_batch != config.batchSize:
                print(f"length-budget: seq_len={seq_len} -> per-rank batch {bucket_batch}")
            loader_kwargs = dict(
                dataset=combined,
                batch_size=bucket_batch,
                num_workers=num_workers,
                sampler=sampler,
                pin_memory=True,
                drop_last=group_drop_last,
            )
            if num_workers > 0:
                loader_kwargs["prefetch_factor"] = min(8, 2 + num_workers // 4)
                loader_kwargs["persistent_workers"] = True
            samplers.append(sampler)
            loaders.append(DataLoader(**loader_kwargs))
        if rank == 0 and len(loaders) > 1:
            print(f"length-bucketed loaders: {len(loaders)} groups "
                  f"(seq lengths {sorted(by_len)})")
        return children, samplers, loaders

    chip_lut = None
    if getattr(config, 'maskMode', '') == 'chip':
        chip_lut = load_chip_lut(config, device)
        if rank == 0:
            print(f"chip mask: {int(chip_lut.sum())} array sites from "
                  f"{config.chipSitesFile}", flush=True)

    train_children, train_samplers, train_loaders = _make_loader(train_hdf5_files, shuffle=True)
    val_children, val_samplers, val_loaders = _make_loader(val_hdf5_files, shuffle=False, drop_last=False)

    # Flat list of child datasets, so cleanup_memory can close each one.
    train_datasets = train_children
    val_datasets = val_children

    # Setup profiler if enabled with enhanced logging at rank 0
    if config.enableProfiling:
        should_stop = torch.tensor([0], dtype=torch.int, device=device)
        if rank == 0:
            profile_log_dir = f"profiling_logs/{config.model}_{config.run}"
            # Create a directory for profiling logs
            os.makedirs("profiling_logs", exist_ok=True)
            os.makedirs(profile_log_dir, exist_ok=True)

            profiling_batch_count = 0

            # Set activities based on environment
            profiler_activities = [torch.profiler.ProfilerActivity.CPU]

            # Only add CUDA profiling if GPU is available and we're using it
            if torch.cuda.is_available():
                try:
                    # Test CUDA accessibility in a safe way
                    torch.cuda.device_count()
                    profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
                except Exception as e:
                    print(f"CUDA profiling not available: {e}")

            # Create the profiler with appropriate activities
            try:
                profiler = torch.profiler.profile(
                    activities=profiler_activities,
                    schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=config.maxProfilingBatches,
                        repeat=1
                    ),
                    on_trace_ready=trace_handler_with_log(profile_log_dir, config.runId),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                )
            except Exception as e:
                print(f"Error initializing profiler: {e}")
                profiler = None
        else:
            profiler = None
            profiling_batch_count = 0

    do_profile = config.enableProfiling and rank == 0 and profiler
    if do_profile:
        profiler.start()

    # ----------------- Epochs loop ------------------

    # Early stopping state, read after epoch_val_loss is all-reduced.
    best_val_loss = float('inf')
    epochs_since_best = 0

    # Anneal missingRatio from curriculumStartRatio over curriculumWarmupEpochs.
    final_missing_ratio = config.missingRatio
    use_curriculum = (
        config.curriculumStartRatio >= 0
        and config.curriculumWarmupEpochs > 0
    )
    if use_curriculum and rank == 0:
        print(f"[curriculum] anneal missingRatio from {config.curriculumStartRatio} "
              f"to {final_missing_ratio} over first {config.curriculumWarmupEpochs} epochs")

    for epoch in range(current_epoch, config.totalEpochs + 1):

        if use_curriculum:
            if epoch <= config.curriculumWarmupEpochs:
                denom = max(1, config.curriculumWarmupEpochs - 1)
                progress = (epoch - 1) / denom
                config.missingRatio = (
                    config.curriculumStartRatio
                    + progress * (final_missing_ratio - config.curriculumStartRatio)
                )
            else:
                config.missingRatio = final_missing_ratio
            if rank == 0 and epoch % 10 == 1:
                print(f"[curriculum] epoch {epoch}: missingRatio = {config.missingRatio:.4f}")

        epoch_train_loss = torch.tensor(0.0, device=device)
        epoch_train_correct = torch.tensor(0, device=device)
        epoch_train_total = torch.tensor(0, device=device)

        model.train()
        if rank == 0:

            pbar_epoch = tqdm(
                total=len(train_loaders),
                desc=f"Epoch {epoch}/{config.totalEpochs} - Training",
                dynamic_ncols=True
            )

        if do_profile:
            print(f"[Rank {rank}] is profiling epoch {epoch}")

        # ------------------ Training phase -------------------
        for dataset_idx, loader in enumerate(train_loaders):
            # Set the epoch for the sampler to change shuffling pattern
            train_samplers[dataset_idx].set_epoch(epoch)

            dataset_train_loss = torch.tensor(0.0, device=device)
            dataset_train_correct = torch.tensor(0, device=device)
            dataset_train_total = torch.tensor(0, device=device)

            for batch_idx, (batch_snps, batch_snpsIndex) in enumerate(loader):

                # Move data to device
                batch_snps = batch_snps.to(device, non_blocking=True)
                batch_snpsIndex = batch_snpsIndex.to(device, non_blocking=True)
                padding_mask = (batch_snps != config.padId).to(device)

                # Data preprocessing
                if config.maskMode == "chip":
                    masked_input_data, missing_mask = mask_chip_columns(
                        config, batch_snps, batch_snpsIndex, chip_lut)
                elif config.maskMode == "colwise":
                    masked_input_data, missing_mask = mask_random_columns(config, batch_snps)
                elif config.sampling == "upsampling":
                    masked_input_data, missing_mask = mask_random_positions_bias(config, batch_snps)
                elif config.sampling == "normal":
                    masked_input_data, missing_mask = mask_random_positions(config, batch_snps, bert_strategy=config.bertStrategy)


                masked_input = one_hot_encode(masked_input_data, num_categories=config.vocabSize, device=device)

                bio_arg = None
                if config.bioAware:
                    from data.bio_lookup import compute_bio_channels
                    bio = compute_bio_channels(
                        batch_snpsIndex[:, :, 0],
                        encoding=getattr(config, 'bioEncoding', 'normPos'),
                        bio_file=getattr(config, 'bioFile', None),
                    )
                    if getattr(config, 'useFiLM', False):
                        bio_arg = bio
                    else:
                        masked_input = torch.cat([masked_input, bio], dim=-1)

                labels = batch_snps.flatten()
                # Exclude structurally-missing labels (class 0) from the loss and accuracy.
                valid_positions = padding_mask.flatten() & (labels != config.missingId)

                # Forward, backward, and optimizer steps
                optimizer.zero_grad()

                # Keep logits, labels and the mask at [B*S, ...] so the graph shape is
                # constant and no host sync is needed per batch.
                padding_mask_flat = padding_mask.flatten()
                pad_id_const = torch.full((), config.padId, dtype=torch.long,
                                           device=device)
                if config.benchmarkAll:
                    loss_mask = valid_positions
                else:
                    loss_mask = missing_mask.flatten() & valid_positions

                use_masked = hasattr(criterion, "forward_masked")

                if device.type == 'cuda':
                    if config.mixedPrecisionTraining:
                        with torch.amp.autocast('cuda'):
                            logits, _ = _model_forward(model, masked_input, bio_arg, config.model)  # [batch, segLen, vocabSize]
                            logits_flat = logits.reshape(-1, config.vocabSize)
                            pred_flat = torch.where(
                                padding_mask_flat,
                                logits.argmax(dim=2).flatten(),
                                pad_id_const,
                            )
                            if use_masked:
                                loss = criterion.forward_masked(logits_flat, labels, loss_mask)
                            else:
                                loss = criterion(logits_flat[loss_mask], labels[loss_mask])
                            batch_correct = ((pred_flat == labels) & loss_mask).sum()
                            batch_total = loss_mask.sum()

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    else:  # no mixed precision training
                        logits, _ = _model_forward(model, masked_input, bio_arg, config.model)
                        logits_flat = logits.reshape(-1, config.vocabSize)
                        pred_flat = torch.where(
                            padding_mask_flat,
                            logits.argmax(dim=2).flatten(),
                            pad_id_const,
                        )
                        if use_masked:
                            loss = criterion.forward_masked(logits_flat, labels, loss_mask)
                        else:
                            loss = criterion(logits_flat[loss_mask], labels[loss_mask])
                        batch_correct = ((pred_flat == labels) & loss_mask).sum()
                        batch_total = loss_mask.sum()

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                else:
                    # CPU training
                    logits, _ = _model_forward(model, masked_input, bio_arg, config.model)
                    logits_flat = logits.reshape(-1, config.vocabSize)
                    pred_flat = torch.where(
                        padding_mask_flat,
                        logits.argmax(dim=2).flatten(),
                        pad_id_const,
                    )
                    if use_masked:
                        loss = criterion.forward_masked(logits_flat, labels, loss_mask)
                    else:
                        loss = criterion(logits_flat[loss_mask], labels[loss_mask])
                    batch_correct = ((pred_flat == labels) & loss_mask).sum()
                    batch_total = loss_mask.sum()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                dataset_train_loss += loss * batch_total
                dataset_train_correct += batch_correct
                dataset_train_total += batch_total

                # Step the profiler and check if we should stop profiling
                if do_profile:
                    profiler.step()
                    profiling_batch_count += 1
                    # Stop profiling after a few batches to avoid huge trace files
                    if profiling_batch_count >= config.maxProfilingBatches + 1 + 1:
                        print(f"Rank {rank} stopping profiler after {profiling_batch_count} steps...")
                        profiler.stop()  # Stop the profiler to save trace
                        do_profile = False
                        should_stop.fill_(1)

                if config.enableProfiling:
                    dist.barrier()
                    dist.broadcast(should_stop, src=0)
                    if should_stop.item() == 1:
                        print(f"Rank {rank} received exit signal from rank 0")
                        cleanup_memory(
                            datasets=train_datasets + val_datasets,
                            dataloaders=train_loaders + val_loaders,
                            samplers=train_samplers + val_samplers,
                            models=[model],
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler if config.mixedPrecisionTraining else None,
                            force_os_release=True
                        )
                        dist.destroy_process_group()
                        sys.exit(0)

            dataset_train_loss = aggregate_scalar(dataset_train_loss, device)
            dataset_train_correct = aggregate_scalar(dataset_train_correct, device)
            dataset_train_total = aggregate_scalar(dataset_train_total, device)

            epoch_train_loss += dataset_train_loss
            epoch_train_correct += dataset_train_correct
            epoch_train_total += dataset_train_total
            if rank == 0:
                avg_loss = (dataset_train_loss / dataset_train_total).item() if dataset_train_total.item() > 0 else 0
                avg_acc = (dataset_train_correct / dataset_train_total).item() if dataset_train_total.item() > 0 else 0
                pbar_epoch.set_description(
                    f"Epoch {epoch}/{config.totalEpochs} - Dataset {dataset_idx + 1} Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}"
                )
                pbar_epoch.update(1)

        # ------------------- Validation phase -------------------
        # Validate on every valFreq-th epoch and on the final one.
        run_val = (epoch % config.valFreq == 0) or (epoch == config.totalEpochs)

        epoch_val_loss = torch.tensor(0.0, device=device)
        epoch_val_correct = torch.tensor(0, device=device)
        epoch_val_total = torch.tensor(0, device=device)

        pbar_val = None
        if run_val:
            model.eval()
            if rank == 0:
                pbar_val = tqdm(
                    total=len(val_loaders),
                    desc=f"Epoch {epoch}/{config.totalEpochs} - Validation",
                    dynamic_ncols=True
                )

        with torch.no_grad():
         if run_val:
            for dataset_idx, loader in enumerate(val_loaders):

                dataset_val_loss = torch.tensor(0.0, device=device)
                dataset_val_correct = torch.tensor(0, device=device)
                dataset_val_total = torch.tensor(0, device=device)

                val_subset_n = max(1, int(len(loader) * config.valSubsetFraction)) if config.valSubsetFraction < 1.0 else None
                for batch_idx, (batch_snps, batch_snpsIndex) in enumerate(loader):
                    if val_subset_n is not None and batch_idx >= val_subset_n:
                        break

                    batch_snps = batch_snps.to(device, non_blocking=True)
                    batch_snpsIndex = batch_snpsIndex.to(device, non_blocking=True)
                    padding_mask = (batch_snps != config.padId).to(device, non_blocking=True)

                    if config.maskMode == "chip":
                        masked_input_data, missing_mask = mask_chip_columns(
                            config, batch_snps, batch_snpsIndex, chip_lut)
                    elif config.maskMode == "colwise":
                        masked_input_data, missing_mask = mask_random_columns(config, batch_snps)
                    else:
                        masked_input_data, missing_mask = mask_random_positions(config, batch_snps)
                    masked_input = one_hot_encode(masked_input_data, num_categories=config.vocabSize, device=device)

                    bio_arg = None
                    if config.bioAware:
                        from data.bio_lookup import compute_bio_channels
                        bio = compute_bio_channels(
                            batch_snpsIndex[:, :, 0],
                            encoding=getattr(config, 'bioEncoding', 'normPos'),
                            bio_file=getattr(config, 'bioFile', None),
                        )
                        if getattr(config, 'useFiLM', False):
                            bio_arg = bio
                        else:
                            masked_input = torch.cat([masked_input, bio], dim=-1)

                    labels = batch_snps.flatten()
                    valid_positions = padding_mask.flatten()

                    if device.type == 'cuda':
                        if config.mixedPrecisionTraining:
                            with torch.amp.autocast('cuda'):
                                logits, _ = _model_forward(model, masked_input, bio_arg, config.model)
                                predicted_genotypes = logits.argmax(dim=2)
                                predicted_genotypes[~padding_mask] = config.padId
                                predicted_genotypes = predicted_genotypes.flatten()

                                if config.benchmarkAll:
                                    logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                                    labels_valid = labels[valid_positions]
                                    if not labels_valid.any():
                                        continue
                                    loss = criterion(logits_valid, labels_valid)
                                    batch_correct = (predicted_genotypes[valid_positions] == labels_valid).sum()
                                    batch_total = valid_positions.sum()
                                else:
                                    missing_mask_flat = missing_mask.flatten()
                                    missing_positions = missing_mask_flat & valid_positions
                                    if missing_positions.sum() == 0:
                                        continue
                                    logits_missing = logits.reshape(-1, config.vocabSize)[missing_positions]
                                    labels_missing = labels[missing_positions]
                                    loss = criterion(logits_missing, labels_missing)
                                    batch_correct = (predicted_genotypes[missing_positions] == labels_missing).sum()
                                    batch_total = missing_positions.sum()

                        else: # No mixed precision

                            logits, _ = _model_forward(model, masked_input, bio_arg, config.model)
                            predicted_genotypes = logits.argmax(dim=2)
                            predicted_genotypes[~padding_mask] = config.padId
                            predicted_genotypes = predicted_genotypes.flatten()

                            if config.benchmarkAll:
                                logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                                labels_valid = labels[valid_positions]
                                if not labels_valid.any():
                                    continue
                                loss = criterion(logits_valid, labels_valid)
                                batch_correct = (predicted_genotypes[valid_positions] == labels_valid).sum()
                                batch_total = valid_positions.sum()
                            else:
                                missing_mask_flat = missing_mask.flatten()
                                missing_positions = missing_mask_flat & valid_positions
                                if missing_positions.sum() == 0:
                                    continue
                                logits_missing = logits.reshape(-1, config.vocabSize)[missing_positions]
                                labels_missing = labels[missing_positions]
                                loss = criterion(logits_missing, labels_missing)
                                batch_correct = (predicted_genotypes[missing_positions] == labels_missing).sum()
                                batch_total = missing_positions.sum()


                    else: # cpu validation

                        logits, _ = _model_forward(model, masked_input, bio_arg, config.model)
                        predicted_genotypes = logits.argmax(dim=2)
                        predicted_genotypes[~padding_mask] = config.padId
                        predicted_genotypes = predicted_genotypes.flatten()

                        if config.benchmarkAll:
                            logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                            labels_valid = labels[valid_positions]
                            if not labels_valid.any():
                                continue

                            loss = criterion(logits_valid, labels_valid)
                            batch_correct = (predicted_genotypes[valid_positions] == labels_valid).sum()
                            batch_total = valid_positions.sum()
                        else:
                            missing_mask_flat = missing_mask.flatten()
                            missing_positions = missing_mask_flat & valid_positions
                            if missing_positions.sum() == 0:
                                continue

                            logits_missing = logits.reshape(-1, config.vocabSize)[missing_positions]
                            labels_missing = labels[missing_positions]
                            loss = criterion(logits_missing, labels_missing)
                            batch_correct = (predicted_genotypes[missing_positions] == labels_missing).sum()
                            batch_total = missing_positions.sum()

                    # Accumulate batch metrics for this dataset
                    dataset_val_loss += loss * batch_total
                    dataset_val_correct += batch_correct
                    dataset_val_total += batch_total

                # Aggregate metrics for this dataset
                dataset_val_loss = aggregate_scalar(dataset_val_loss, device)
                dataset_val_correct = aggregate_scalar(dataset_val_correct, device)
                dataset_val_total = aggregate_scalar(dataset_val_total, device)

                epoch_val_loss += dataset_val_loss
                epoch_val_correct += dataset_val_correct
                epoch_val_total += dataset_val_total

                if rank == 0:
                    avg_loss = (dataset_val_loss / dataset_val_total).item() if dataset_val_total > 0 else 0
                    avg_acc = (dataset_val_correct / dataset_val_total).item() if dataset_val_total > 0 else 0
                    pbar_val.set_description(
                        f"Epoch {epoch}/{config.totalEpochs} - Dataset {dataset_idx + 1} Val Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}"
                    )
                    pbar_val.update(1)

        # Calculate final epoch metrics using scalars
        if epoch_train_total > 0:
            epoch_train_loss = epoch_train_loss / epoch_train_total
            epoch_train_accuracy = epoch_train_correct / epoch_train_total
        else:
            epoch_train_loss = float('inf')
            epoch_train_accuracy = 0.0

        if epoch_val_total > 0:
            epoch_val_loss = epoch_val_loss / epoch_val_total
            epoch_val_accuracy = epoch_val_correct / epoch_val_total
        else:
            epoch_val_loss = float('inf')
            epoch_val_accuracy = 0.0

        scheduler.step()

        # Save every epoch so a cancelled run can resume; keep only milestones
        # on disk.
        if rank == 0:
            # Save the inner DDP module's state dict, which loads without compile.
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint({
                'epoch': epoch,
                'state_dict': save_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, epoch, config, filename_prefix="checkpoint")

            prev_epoch = epoch - 1
            prev_is_milestone = (
                prev_epoch >= 1 and (prev_epoch % config.saveCheckpointFreq == 0)
            )
            if prev_epoch >= 1 and not prev_is_milestone:
                prev_path = os.path.join(
                    config.modelDir,
                    f"checkpoint_{config.run}_epoch_{prev_epoch}.pth",
                )
                if os.path.isfile(prev_path):
                    os.remove(prev_path)

            if config.useWandB and wandb is not None:
                wandb.log({
                    "train_loss": epoch_train_loss,
                    "train_accuracy": epoch_train_accuracy,
                    "val_loss": epoch_val_loss,
                    "val_accuracy": epoch_val_accuracy,
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

            # Close progress bars
            if pbar_epoch:
                pbar_epoch.close()
            if pbar_val:
                pbar_val.close()

        # Check early stopping on validation epochs, after the curriculum ends.
        in_curriculum = use_curriculum and epoch <= config.curriculumWarmupEpochs
        if run_val and epoch_val_total > 0 and config.earlyStoppingPatience > 0 and not in_curriculum:
            val_loss_now = float(epoch_val_loss) if hasattr(epoch_val_loss, 'item') is False else epoch_val_loss.item() if hasattr(epoch_val_loss, 'item') else float(epoch_val_loss)
            if val_loss_now < best_val_loss - config.earlyStoppingMinDelta:
                best_val_loss = val_loss_now
                epochs_since_best = 0
                # Save 'best' checkpoint snapshot
                if rank == 0:
                    save_model_es = model._orig_mod if hasattr(model, "_orig_mod") else model
                    best_ckpt_path = os.path.join(
                        config.modelDir,
                        f"checkpoint_{config.run}_epoch_best.pth"
                    )
                    torch.save({
                        'epoch': epoch,
                        'state_dict': save_model_es.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, best_ckpt_path)
            else:
                epochs_since_best += 1

            if epochs_since_best >= config.earlyStoppingPatience:
                if rank == 0:
                    print(f"[early stop] val_loss did not improve for {epochs_since_best} val epochs "
                          f"(best={best_val_loss:.4f} at epoch <={epoch}); stopping.")
                break

    # Clean up datasets at the end of training
    cleanup_memory(
        datasets=train_datasets + val_datasets,
        dataloaders=train_loaders + val_loaders,
        samplers=train_samplers + val_samplers,
        models=[model],
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler if config.mixedPrecisionTraining else None,
        force_os_release=True
    )

    # Cleanup
    dist.destroy_process_group()
    if rank == 0 and config.useWandB and wandb is not None:
        wandb.finish()


def get_num_samples(hdf5_files):
    """
    Count the total number of samples across multiple HDF5 files.

    Args:
        hdf5_files: List of HDF5 file paths

    Returns:
        Total number of samples
    """
    total_samples = 0
    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, 'r') as f:
            total_samples += f['snps'].shape[0]
    return total_samples

def main():
    # Create a parser that handles all arguments
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training for SNPs Model')
    parser.add_argument('--configFile', type=str, required=False, help='Path to configuration file')
    parser.add_argument('--local_gpu', type=int, default=None, help='Explicitly set local GPU ID (overrides automatic assignment)')

    # Parse all arguments
    args = parser.parse_args()

    config = ModelConfig.from_yaml(args.configFile)
    if config.segLen % (2 ** (config.depth - 1)) and config.segLen != -1:
        raise ValueError(f"segLen must be compatible with model depth.")
    if (config.segLen != -1) and (not 0 <= config.overlap < config.segLen):
        raise ValueError("Overlap must be between 0 and segLen-1")

    config.run = f"{config.runId}_{config.dataset}_chr{config.chromosome}_{config.population}_seg{config.segLen}_overlap{config.overlap}"
    train_hdf5_files, val_hdf5_files = get_dataset_paths(config)
    latest_checkpoint_file = find_latest_checkpoint(config)

    # Get SLURM environment variables for distributed setup
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    use_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0

    train_ddp(
        use_gpu,
        rank=rank,
        world_size=world_size,
        config=config,
        train_hdf5_files=train_hdf5_files,
        val_hdf5_files=val_hdf5_files,
        checkpoint_file=latest_checkpoint_file
    )


if __name__ == "__main__":
    main()
