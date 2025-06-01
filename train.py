#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pdb import set_trace
import sys
import os
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
from torch.autograd.profiler import record_function
import math

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
    if num_gpus <= 2:
        return 100
    elif num_gpus <= 4:
        return 100
    elif num_gpus <= 8:
        return 100
    else:
        return 200

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

        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
        torch.cuda.set_device(device)
        backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'

    if rank == 0:
        print_exp_summary(use_gpu, world_size, rank, backend, config, train_hdf5_files, val_hdf5_files, checkpoint_file)

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
    num_workers = get_optimal_num_workers()

    # Auto mixed precision - only for GPU
    scaler = None
    if config.mixedPrecisionTraining and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

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
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        current_epoch = checkpoint['epoch'] + 1

        # Ensure each optimizer param group has an 'initial_lr'
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = param_group["lr"]
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

    # Pre-load all datasets before the epoch loop
    train_datasets = []
    train_samplers = []
    train_loaders = []

    for hdf5_file in train_hdf5_files:
        dataset = SNPsDataset_HDF5(hdf5_file)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

        loader_kwargs = dict(
            dataset=dataset,
            batch_size=config.batchSize,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True
        )
        # Only set prefetch_factor if using multiprocessing
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = min(4, 2 + num_workers // 4)
            loader_kwargs["persistent_workers"] = True
            loader_kwargs['drop_last'] = True
        loader = DataLoader(**loader_kwargs)

        train_datasets.append(dataset)
        train_samplers.append(sampler)
        train_loaders.append(loader)

    # Similarly for validation datasets
    val_datasets = []
    val_samplers = []
    val_loaders = []

    for hdf5_file in val_hdf5_files:
        dataset = SNPsDataset_HDF5(hdf5_file)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

        loader_kwargs = dict(
            dataset=dataset,
            batch_size=config.batchSize,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=True
        )
        # Only set prefetch_factor if using multiprocessing
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = min(4, 2 + num_workers // 4)
            loader_kwargs["persistent_workers"] = True
        loader = DataLoader(**loader_kwargs)

        val_datasets.append(dataset)
        val_samplers.append(sampler)
        val_loaders.append(loader)

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

    for epoch in range(current_epoch, config.totalEpochs + 1):

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
                if config.sampling == "upsampling":
                    masked_input_data, missing_mask = mask_random_positions_bias(config, batch_snps)
                elif config.sampling == "normal":
                    masked_input_data, missing_mask = mask_random_positions(config, batch_snps, bert_strategy=config.bertStrategy)


                masked_input = one_hot_encode(masked_input_data, num_categories=config.vocabSize, device=device)

                if config.bioAware:
                    normalized_pos = (
                        (batch_snpsIndex[:, :, 0] - batch_snpsIndex[:, :, 0].min(dim=1, keepdim=True)[0]) /
                        (batch_snpsIndex[:, :, 0].max(dim=1, keepdim=True)[0] - batch_snpsIndex[:, :, 0].min(dim=1, keepdim=True)[0])
                    ).unsqueeze(-1)

                    masked_input = torch.cat([masked_input, normalized_pos], dim=-1)

                labels = batch_snps.flatten()
                valid_positions = padding_mask.flatten()

                # Forward, backward, and optimizer steps
                optimizer.zero_grad()

                if device.type == 'cuda':
                    if config.mixedPrecisionTraining:
                        with torch.amp.autocast('cuda'):
                            logits, _ = model(masked_input)             # [batch, segLen, vocabSize]
                            predicted_genotypes = logits.argmax(dim=2)  # [batch, segLen]
                            predicted_genotypes[~padding_mask] = config.padId # [batch, segLen]
                            predicted_genotypes = predicted_genotypes.flatten() # [batch*segLen]
                            if config.benchmarkAll:
                                logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]  # [batch*seglen-invalid, 6]
                                labels_valid = labels[valid_positions] # [batch*seglen-invalid]
                                if labels_valid.numel() == 0:
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

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                    else: #no mixed precision training
                        logits, _ = model(masked_input)             # [batch, segLen, vocabSize]
                        predicted_genotypes = logits.argmax(dim=2)  # [batch, segLen]
                        predicted_genotypes[~padding_mask] = config.padId # [batch, segLen]
                        predicted_genotypes = predicted_genotypes.flatten() # [batch*segLen]

                        if config.benchmarkAll:
                            logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]  # [batch*seglen-invalid, 6]
                            labels_valid = labels[valid_positions] # [batch*seglen-invalid]
                            if labels_valid.numel() == 0:
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


                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                else:
                    # CPU training
                    logits, _ = model(masked_input)

                    predicted_genotypes = logits.argmax(dim=2)
                    predicted_genotypes[~padding_mask] = config.padId
                    predicted_genotypes = predicted_genotypes.flatten()

                    if config.benchmarkAll:
                        logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                        labels_valid = labels[valid_positions]
                        if labels_valid.numel() == 0:
                            continue
                        loss = criterion(logits_valid, labels_valid)
                        batch_correct = (predicted_genotypes[valid_positions] == labels_valid).sum()
                        batch_total = valid_positions.sum()
                    else:
                        missing_mask_flat = missing_mask.flatten()
                        missing_positions = missing_mask_flat & valid_positions
                        if missing_positions.sum() == 0:
                            continue
                        loss = criterion(logits.reshape(-1, config.vocabSize)[missing_positions], labels[missing_positions])
                        batch_correct = (predicted_genotypes[missing_positions] == labels[missing_positions]).sum()
                        batch_total = missing_positions.sum()

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

        epoch_val_loss = torch.tensor(0.0, device=device)
        epoch_val_correct = torch.tensor(0, device=device)
        epoch_val_total = torch.tensor(0, device=device)

        model.eval()
        if rank == 0:
            pbar_val = tqdm(
                total=len(val_loaders),
                desc=f"Epoch {epoch}/{config.totalEpochs} - Validation",
                dynamic_ncols=True
            )

        with torch.no_grad():
            for dataset_idx, loader in enumerate(val_loaders):

                dataset_val_loss = torch.tensor(0.0, device=device)
                dataset_val_correct = torch.tensor(0, device=device)
                dataset_val_total = torch.tensor(0, device=device)

                for batch_idx, (batch_snps, batch_snpsIndex) in enumerate(loader):

                    batch_snps = batch_snps.to(device, non_blocking=True)
                    batch_snpsIndex = batch_snpsIndex.to(device, non_blocking=True)
                    padding_mask = (batch_snps != config.padId).to(device, non_blocking=True)

                    masked_input_data, missing_mask = mask_random_positions(config, batch_snps)
                    masked_input = one_hot_encode(masked_input_data, num_categories=config.vocabSize, device=device)

                    if config.bioAware:
                        normalized_pos = (
                            (batch_snpsIndex[:, :, 0] - batch_snpsIndex[:, :, 0].min(dim=1, keepdim=True)[0]) /
                            (batch_snpsIndex[:, :, 0].max(dim=1, keepdim=True)[0] - batch_snpsIndex[:, :, 0].min(dim=1, keepdim=True)[0])
                        ).unsqueeze(-1)

                        masked_input = torch.cat([masked_input, normalized_pos], dim=-1)

                    labels = batch_snps.flatten()
                    valid_positions = padding_mask.flatten()

                    if device.type == 'cuda':
                        if config.mixedPrecisionTraining:
                            with torch.amp.autocast('cuda'):
                                logits, _ = model(masked_input)
                                predicted_genotypes = logits.argmax(dim=2)
                                predicted_genotypes[~padding_mask] = config.padId
                                predicted_genotypes = predicted_genotypes.flatten()

                                if config.benchmarkAll:
                                    logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                                    labels_valid = labels[valid_positions]
                                    if labels_valid.numel() == 0:
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

                            logits, _ = model(masked_input)
                            predicted_genotypes = logits.argmax(dim=2)
                            predicted_genotypes[~padding_mask] = config.padId
                            predicted_genotypes = predicted_genotypes.flatten()

                            if config.benchmarkAll:
                                logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                                labels_valid = labels[valid_positions]
                                if labels_valid.numel() == 0:
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

                        logits, _ = model(masked_input)
                        predicted_genotypes = logits.argmax(dim=2)
                        predicted_genotypes[~padding_mask] = config.padId
                        predicted_genotypes = predicted_genotypes.flatten()

                        if config.benchmarkAll:
                            logits_valid = logits.reshape(-1, config.vocabSize)[valid_positions]
                            labels_valid = labels[valid_positions]
                            if labels_valid.numel() == 0:
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
            epoch_train_loss = (epoch_train_loss / epoch_train_total).item()
            epoch_train_accuracy = (epoch_train_correct / epoch_train_total).item()
        else:
            epoch_train_loss = float('inf')
            epoch_train_accuracy = 0.0

        if epoch_val_total > 0:
            epoch_val_loss = (epoch_val_loss / epoch_val_total).item()
            epoch_val_accuracy = (epoch_val_correct / epoch_val_total).item()
        else:
            epoch_val_loss = float('inf')
            epoch_val_accuracy = 0.0

        scheduler.step()

        # Save checkpoint and log metrics
        if rank == 0:
            if epoch % config.saveCheckpointFreq == 0 or epoch == config.totalEpochs:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()  # Save scheduler state
                }, epoch, config, filename_prefix="checkpoint")

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

    # run is the naming pattern for the checkpoint file, modify it at your wish
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
