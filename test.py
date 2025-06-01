import sys
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import gzip
import argparse
import time
import socket
import pandas as pd
from config.modelconfig import ModelConfig
from data.dataset import SNPsDataset_HDF5
from data.utils import one_hot_encode, cleanup_memory
from data.segmentation import de_segmentation

def get_optimal_num_workers():
    # Slurm tells each process how many CPUs it *should* use
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    # Reserve 1 CPU for the main DDP process
    return max(1, cpus_per_task - 1)

def get_optimal_bucket_size(num_gpus):
    if num_gpus <= 2:
        return 20
    elif num_gpus <= 4:
        return 50
    elif num_gpus <= 8:
        return 100
    else:
        return 200

def parse_args():
    parser = argparse.ArgumentParser(description="Testing using DDP")
    parser.add_argument('--configFile', type=str, required=True, help="configuration file")
    args = parser.parse_args()
    return args


def setup_ddp():
    """Initialize the distributed environment."""
    # Get rank and world size from Slurm
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
#    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    # Determine device based on availability
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        backend = "nccl"  # NCCL backend for GPU
    else:
        device = torch.device("cpu")
        backend = "gloo"  # Gloo backend for CPU

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    if rank == 0:
        print(f"Initialized DDP: world_size={world_size}, backend={backend}")
        print(f"Running on {socket.gethostname()}, rank {rank}, device {device}")

    return rank, world_size, device


def test_combination(config, random_state, missing_index, rank, world_size, device):
    """Test a specific combination of random state and missing level using DDP."""
    # Setup logging for rank 0
    log_dir = f"archive/exp1/{config.dataset}_chr{config.chromosome}_{config.population}_seg{config.segLen}_overlap{config.overlap}"
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = f"{log_dir}/test_ddp_{config.runId}_rand{str(random_state)}_missingIdx{missing_index}.log"
        log_file = open(log_file_path, "w")
        old_stdout = sys.stdout
        sys.stdout = log_file
        print(f"Starting combination: random_state={random_state}, missing_index={missing_index}")
        print(f"World size: {world_size}, Device: {device}")

    # Get dataset files
    masked_dataset_file = config.masked_dataset_files(random_state)[missing_index]

    # Load datasets
    test_dataset = SNPsDataset_HDF5(config.test_dataset_file)
    masked_dataset = SNPsDataset_HDF5(masked_dataset_file)

    # Create distributed samplers
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )

    mask_sampler = DistributedSampler(
        masked_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )

    # Create dataloaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batchSizeTest,
        sampler=test_sampler,
        num_workers=get_optimal_num_workers(),
        persistent_workers=True,
        pin_memory=True if torch.cuda.is_available() else False
    )

    mask_loader = DataLoader(
        masked_dataset,
        batch_size=config.batchSizeTest,
        sampler=mask_sampler,
        num_workers=get_optimal_num_workers(),
        persistent_workers=True,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Load model
    if config.model == 'unet':
        from model.unet import BiUNet
        model = BiUNet(config).to(device)
    elif config.model == 'scda':
        from model.ae import SCDA
        model = SCDA(config).to(device)

    # Load checkpoint
    checkpoint = torch.load(config.checkpoint, map_location=device)

    # Handle potential module prefix in state dict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('module.'):
            new_k = k[7:]  # Remove 'module.' prefix
        else:
            new_k = k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)

    # Wrap model with DDP
    model = DDP(model, device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None, bucket_cap_mb=get_optimal_bucket_size(world_size))
    model.eval()

    # Initialize metrics
    test_correct = torch.tensor(0.0, device=device)
    test_total = torch.tensor(0.0, device=device)
    segments_predictions = []
    segments_indices = []
    segments_confidences = []

    # Testing loop
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(zip(test_loader, mask_loader), total=len(test_loader),
                       desc=f'Testing RS{random_state} Miss{missing_index}', mininterval=5)
        else:
            pbar = zip(test_loader, mask_loader)

        for batch_idx, (test_batch, mask_batch) in enumerate(pbar):
            # Process test data
            test_snps, test_snpsIndex = test_batch
            test_snps = test_snps.to(device)
            test_snpsIndex = test_snpsIndex.to(device)
            padding_mask = (test_snps != config.padId).to(device)
            valid_positions = padding_mask.flatten()

            # Process masked data
            masked_snps, masked_snpsIndex = mask_batch
            masked_snps = masked_snps.to(device)
            masked_snpsIndex = masked_snpsIndex.to(device)
            missing_positions = (masked_snps == config.missingId).flatten()

            # Prepare input
            labels = test_snps.flatten()
            masked_input = one_hot_encode(masked_snps, num_categories=config.vocabSize, device=device)

            if config.bioAware:
                normalized_pos = (
                    (masked_snpsIndex[:, :, 0] - masked_snpsIndex[:, :, 0].min(dim=1, keepdim=True)[0]) /
                    (masked_snpsIndex[:, :, 0].max(dim=1, keepdim=True)[0] - masked_snpsIndex[:, :, 0].min(dim=1, keepdim=True)[0])
                ).unsqueeze(-1)

                masked_input = torch.cat([masked_input, normalized_pos], dim=-1)

            # Forward pass
            logits, _ = model(masked_input)
            predicted = logits.argmax(dim=2)
            predicted[~padding_mask] = config.padId
            predicted_flat = predicted.flatten()

            # Store predictions
            segments_predictions.append(predicted)
            segments_indices.append(test_snpsIndex)
            segments_confidences.append(F.softmax(logits, dim=-1))

            # Calculate accuracy
            if config.benchmarkAll:
                # Calculate metrics on all valid positions
                labels_valid = labels[valid_positions]
                predictions_valid = predicted_flat[valid_positions]
            else:
                # Calculate metrics only on masked positions
                masked_valid_positions = missing_positions & valid_positions
                labels_valid = labels[masked_valid_positions]
                predictions_valid = predicted_flat[masked_valid_positions]

            if labels_valid.numel() > 0:
                batch_correct = (predictions_valid == labels_valid).sum()
                test_correct += batch_correct
                #test_total += torch.tensor(labels_valid.size(0), device=device)
                test_total += labels_valid.numel()
            # Update progress bar on rank 0
            if rank == 0 and batch_idx % 10 == 0:
                current_acc = (test_correct / test_total).item() if test_total > 0 else 0
                pbar.set_postfix({'Accuracy': f'{current_acc:.4f}'})

    # Concatenate results for this process
    segments_predictions = torch.cat(segments_predictions, dim=0)
    segments_indices = torch.cat(segments_indices, dim=0)
    segments_confidences = torch.cat(segments_confidences, dim=0)

    # Compute local accuracy
    local_acc = (test_correct / test_total).item() if test_total > 0 else 0

    # Gather tensor sizes from all processes
    local_pred_size = torch.tensor([segments_predictions.size(0)], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_pred_size)
    all_sizes = [size.item() for size in all_sizes]

    # Find maximum size for padding
    max_size = max(all_sizes)

    # Create padded tensors to use with all_gather
    padded_predictions = torch.zeros((max_size, segments_predictions.size(1)), 
                                    dtype=segments_predictions.dtype, device=device)
    padded_indices = torch.zeros((max_size, segments_indices.size(1), segments_indices.size(2)), 
                                dtype=segments_indices.dtype, device=device)
    padded_confidences = torch.zeros((max_size, segments_confidences.size(1), segments_confidences.size(2)), 
                                    dtype=segments_confidences.dtype, device=device)
    
    # Copy local data to padded tensors
    padded_predictions[:segments_predictions.size(0)] = segments_predictions
    padded_indices[:segments_indices.size(0)] = segments_indices
    padded_confidences[:segments_confidences.size(0)] = segments_confidences
    
    # Create tensors to gather results from all processes
    gathered_predictions = [torch.zeros_like(padded_predictions) for _ in range(world_size)]
    gathered_indices = [torch.zeros_like(padded_indices) for _ in range(world_size)]
    gathered_confidences = [torch.zeros_like(padded_confidences) for _ in range(world_size)]
    
    # Gather results from all processes in one go
    dist.all_gather(gathered_predictions, padded_predictions)
    dist.all_gather(gathered_indices, padded_indices)
    dist.all_gather(gathered_confidences, padded_confidences)

    # Reduce accuracy metrics
    dist.reduce(test_correct, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(test_total, dst=0, op=dist.ReduceOp.SUM)
    
    # Only rank 0 needs to process the gathered data
    if rank == 0:
        # Trim padding and convert to numpy arrays
        all_predictions = []
        all_indices = []
        all_confidences = []
        
        for r in range(world_size):
            size = all_sizes[r]
            all_predictions.append(gathered_predictions[r][:size].cpu().numpy())
            all_indices.append(gathered_indices[r][:size].cpu().numpy())
            all_confidences.append(gathered_confidences[r][:size].cpu().numpy())
        
        # Combine all predictions
        combined_predictions = np.concatenate(all_predictions, axis=0)
        combined_indices = np.concatenate(all_indices, axis=0)
        combined_confidences = np.concatenate(all_confidences, axis=0)
        
        # Calculate global accuracy
        global_acc = (test_correct / test_total).item() if test_total > 0 else 0
        
        # De-segmentation and save results
        print("Starting de-segmentation process...")
        imputed_testset = de_segmentation(
            config, combined_predictions, combined_indices, combined_confidences
        )
        
        print(f"Test set average acc ({config.missing_percent_strs[missing_index]} missing, "
              f"{'masked positions' if not config.benchmarkAll else 'all positions'}): {global_acc:.4f}")
        
        # Save results
        imputed_csv_gz = config.imputed_csv_gzs(random_state)[missing_index]
        imputed_path = os.path.dirname(imputed_csv_gz)
        os.makedirs(imputed_path, exist_ok=True)
        
        print(f"Saving results to {imputed_csv_gz}")
        with gzip.open(imputed_csv_gz, 'wt') as f:
            imputed_testset.to_csv(f, index=True)
        
        print(f"Completed combination: random_state={random_state}, missing_index={missing_index}")
        
        # Reset stdout
        sys.stdout = old_stdout
        log_file.close()
    
    # Ensure all processes complete before returning
    dist.barrier()
    # # Explicit cleanup
    cleanup_memory(
        datasets=[test_dataset, masked_dataset],
        dataloaders=[test_loader, mask_loader],
        tensors=[segments_predictions, segments_indices, segments_confidences] + 
        ([] if 'gathered_predictions' not in locals() else 
         [gathered_predictions, gathered_indices, gathered_confidences]),
        force_os_release=True
    )
    return local_acc


def main():
    args = parse_args()
    config = ModelConfig.from_yaml(args.configFile)

    # Setup distributed environment
    rank, world_size, device = setup_ddp()

    # Main process logs configuration
    if rank == 0:
        print(f"Using checkpoint: {config.checkpoint}")
        print(f"Model (runid: {config.runId}, epoch: {config.epoch}) loaded.")
        print(f"Testing with {world_size} processes.")
        print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Define combinations to process
    rand_states = [0, 42, 1024]
    missing_indices = list(range(len(config.missing_percent_strs)))  # [0,1,2,3] for 5%, 15%, 25%, 50%

    # Process each combination sequentially, but using all processes for each
    start_time = time.time()
    results = {}

    for random_state in rand_states:
        for missing_index in missing_indices:
            combination_key = f"rs{random_state}_miss{missing_index}"

            if rank == 0:
                print(f"Starting combination: {combination_key}")
                comb_start = time.time()

            # All processes participate in testing this combination
            local_acc = test_combination(
                config, random_state, missing_index, rank, world_size, device
            )

            if rank == 0:
                comb_time = time.time() - comb_start
                results[combination_key] = (local_acc, comb_time)
                print(f"Completed {combination_key} in {comb_time:.2f} seconds")

    # Final summary from rank 0
    if rank == 0:
        total_time = time.time() - start_time
        print("\nResults Summary:")
        for key, (acc, elapsed) in results.items():
            print(f"{key}: Accuracy={acc:.4f}, Time={elapsed:.2f}s")
        print(f"Total processing time: {total_time:.2f} seconds")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
