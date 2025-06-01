import multiprocessing
import time
import pandas as pd
import h5py
import numpy as np
import os
import glob
from typing import Tuple, List, Optional, Dict
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import argparse
import re
from config.modelconfig import ModelConfig
from data.segmentation import segmentation, prepare_target_snps
from data.metrics import calculate_maf
from data.utils import get_dataset_paths
from dataclasses import asdict
from pdb import set_trace
import sys

def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser(description="Preparing segmented dataset for training/testing Autoencoder Genotype Data Imputation")
    parser.add_argument('--configFile', type=str, required=True, help="configuration file")
    parser.add_argument('--createFilteredSNPs', type=str2bool, required=False, default=True, help="False when segLen == -1")
    # No longer require randState as an argument - we'll process all random states
    args = parser.parse_args()
    return args


def extract_chunk_index(filename: str) -> int:
    match = re.search(r'chunk(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract chunk index from filename {filename}")


def rm_dup(snps, snpsIndex):
    # Generate unique identifiers for each SNP segment
    unique_ids = np.array([''.join(map(str, row)) for row in snps])

    # Identify unique indices and count duplicates
    _, unique_indices = np.unique(unique_ids, return_index=True)

    # Count total and unique SNP segments
    total_segments = len(snps)
    unique_segments = len(unique_indices)
    duplicate_segments = total_segments - unique_segments
    proportion_duplicates = duplicate_segments / total_segments

    # Print duplication info
    print(f"Total SNP segments: {total_segments}")
    print(f"Unique SNP segments after trimming: {unique_segments}")
    print(f"Duplicated SNP segments found: {duplicate_segments}")
    print(f"Proportion of duplicates: {proportion_duplicates:.2%}")

    # Sort unique_indices to maintain order if needed
    unique_indices = sorted(unique_indices)

    # Filter the datasets using these unique indices
    unique_snps = snps[unique_indices]
    unique_snpsIndex = snpsIndex[unique_indices]

    return unique_snps, unique_snpsIndex


def process_and_filter_hdf5(config_dict: dict, file_info: Tuple[str, str], set_type: str = None, create_filtered_snps: bool = False) -> None:
    config = ModelConfig(**config_dict)
    csv_file, hdf5_file = file_info

    # Get rank for better logging
    rank = int(os.environ.get('SLURM_PROCID', 0))
    
    # First, make sure the original HDF5 exists
    if not os.path.exists(hdf5_file):
        print(f"Rank {rank}: Processing {csv_file} to create original HDF5...")
        start_time = time.time()
        
        try:
            df = pd.read_csv(csv_file, compression='gzip', index_col=0)
            df.columns = df.columns.astype(int)
            snps, snpsIndex, _ = segmentation(config, df)

            if set_type == "train" and config.unique is True:
                snps, snpsIndex = rm_dup(snps, snpsIndex)

            with h5py.File(hdf5_file, 'w') as f:
                f.create_dataset('snps', data=snps, dtype='i8', compression="gzip", chunks=True)
                f.create_dataset('snpsIndex', data=snpsIndex, dtype='i8', compression="gzip", chunks=True)
            print(f"Rank {rank}: {csv_file}: original HDF5 created in {time.time() - start_time:.2f}s.")
        except Exception as e:
            print(f"Rank {rank}: Error processing {csv_file}: {str(e)}")
            return
    else:
        print(f"Rank {rank}: {hdf5_file} exists, skipping.")

    if create_filtered_snps:
        # Now create filtered versions for each bin range
        print(f"Rank {rank}: Creating filtered versions of {hdf5_file} for all bin ranges...")

        # Load the original HDF5 file
        try:
            with h5py.File(hdf5_file, 'r') as f:
                original_snps = f['snps'][:]
                original_snpsIndex = f['snpsIndex'][:]
        except Exception as e:
            print(f"Rank {rank}: Error loading {hdf5_file}: {str(e)}")
            return

        # Convert target_snps to a set for faster lookups
        bin_target_snps = {}

        # Process each bin range
        for bin_range in config.binRanges:
            # Generate filtered file name
            base_name, ext = os.path.splitext(hdf5_file)
            filtered_hdf5_file = f"{base_name}_filtered_{bin_range}{ext}"

            # Skip if filtered file already exists
            if os.path.exists(filtered_hdf5_file):
                print(f"Rank {rank}: Filtered HDF5 file {filtered_hdf5_file} already exists, skipping.")
                continue

            # Get target SNPs for this bin range
            start_time = time.time()
            try:
                target_snps = prepare_target_snps(config, bin_range)
                bin_target_snps[bin_range] = set(target_snps)  # Convert to set for faster lookups
                print(f"Rank {rank}: Found {len(target_snps)} target SNPs in bin range {bin_range}")
            except Exception as e:
                print(f"Rank {rank}: Error preparing target SNPs for bin range {bin_range}: {str(e)}")
                continue

            # Process and filter each segment
            filtered_segments = []
            filtered_indices = []

            # Use set for faster lookups
            target_snps_set = bin_target_snps[bin_range]

            for i in range(len(original_snps)):
                segment_index = original_snpsIndex[i]
                keep_segment = False

                # Check if any position in this segment matches target SNPs
                for j in range(segment_index.shape[0]):
                    # segment_index[j][0] gives the SNP position ID (locus)
                    if segment_index[j][0] in target_snps_set:
                        keep_segment = True
                        break

                if keep_segment:
                    filtered_segments.append(original_snps[i])
                    filtered_indices.append(segment_index)

            # Convert to numpy arrays and save if we have matches
            if filtered_segments:
                filtered_snps = np.array(filtered_segments)
                filtered_snpsIndex = np.array(filtered_indices)

                # Save the filtered data
                try:
                    with h5py.File(filtered_hdf5_file, 'w') as f:
                        f.create_dataset('snps', data=filtered_snps, dtype='i8', compression="gzip", chunks=True)
                        f.create_dataset('snpsIndex', data=filtered_snpsIndex, dtype='i8', compression="gzip", chunks=True)
                        # Add metadata
                        f.attrs['original_file'] = hdf5_file
                        f.attrs['target_bin'] = bin_range
                        f.attrs['original_segments'] = len(original_snps)
                        f.attrs['filtered_segments'] = len(filtered_snps)

                    print(f"Rank {rank}: Created filtered HDF5 {filtered_hdf5_file} with {len(filtered_snps)}/{len(original_snps)} segments ({len(filtered_snps)/len(original_snps):.2%}) in {time.time() - start_time:.2f}s")
                except Exception as e:
                    print(f"Rank {rank}: Error saving {filtered_hdf5_file}: {str(e)}")
            else:
                print(f"Rank {rank}: Warning: No segments match the target SNPs in bin range {bin_range}")


def process_test_file(config_dict: dict, file_info: Tuple[str, str]) -> None:
    config = ModelConfig(**config_dict)
    csv_file, hdf5_file = file_info
    
    # Get rank for better logging
    rank = int(os.environ.get('SLURM_PROCID', 0))

    if os.path.exists(hdf5_file):
        print(f"Rank {rank}: HDF5 file {hdf5_file} already exists, skipping.")
        return

    print(f"Rank {rank}: Processing {csv_file} ...")
    start_time = time.time()
    
    try:
        df = pd.read_csv(csv_file, compression='gzip', index_col=0)
        df.columns = df.columns.astype(int)
        
        snps, snpsIndex, _ = segmentation(config, df)

        with h5py.File(hdf5_file, 'w') as f:
            f.create_dataset('snps', data=snps, dtype='i8', compression="gzip", chunks=True)
            f.create_dataset('snpsIndex', data=snpsIndex, dtype='i8', compression="gzip", chunks=True)
        print(f"Rank {rank}: {csv_file}: process finished in {time.time() - start_time:.2f}s.")
    except Exception as e:
        print(f"Rank {rank}: Error processing {csv_file}: {str(e)}")


def partition_files_among_tasks(file_list: List[Tuple[str, str]], num_tasks: int, rank: int, reverse: bool = False) -> List[Tuple[str, str]]:
    """
    Partition files among tasks in a round-robin manner.
    If reverse is True, start from the last rank and go backwards.
    """
    if reverse:
        # Assign file i to rank (num_tasks-1-i) % num_tasks
        return [file_list[i] for i in range(len(file_list)) if (num_tasks - 1 - i) % num_tasks == rank]
    else:
        # Normal round-robin
        return [file_list[i] for i in range(len(file_list)) if i % num_tasks == rank]


def get_test_masked_file_info(config: ModelConfig, random_states: List[int]) -> List[Tuple[str, str]]:
    """
    Get all test and masked file info for all random states.
    """
    all_file_info = []
    
    # Add test file (same for all random states)
    all_file_info.append((config.test_csv_gz, config.test_dataset_file))
    
    # Add masked files for each random state
    for state in random_states:
        masked_csv_files = config.masked_csv_gzs(state)
        masked_hdf5_files = config.masked_dataset_files(state)
        
        for csv_file, hdf5_file in zip(masked_csv_files, masked_hdf5_files):
            all_file_info.append((csv_file, hdf5_file))
    
    return all_file_info


def main():
    args = parse_args()
    config = ModelConfig.from_yaml(args.configFile)
    
    # Define random states to process
    random_states = [0, 42, 1024]  # Add all random states you want to process

    # Get rank and number of tasks
    rank = int(os.environ.get('SLURM_PROCID', 0))
    num_tasks = int(os.environ.get('SLURM_NTASKS', 1))
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    os.makedirs(config.resDir, exist_ok=True)
    print(f"Rank {rank}: Starting data processing with {num_cpus} CPUs per task...")

    # Get file paths
    train_file_pairs, val_file_pairs = get_dataset_paths(config, segmentation=True)

    # Get test and masked file info for all random states
    test_masked_file_info = get_test_masked_file_info(config, random_states)

    # Partition all files among tasks
    local_train_file_pairs = partition_files_among_tasks(train_file_pairs, num_tasks, rank)
    local_val_file_pairs = partition_files_among_tasks(val_file_pairs, num_tasks, rank, reverse=True)
    local_test_masked_info = partition_files_among_tasks(test_masked_file_info, num_tasks, rank)

    # Prepare functions for processing
    config_dict = asdict(config)
    process_train_val = partial(process_and_filter_hdf5, config_dict, create_filtered_snps = args.createFilteredSNPs)
    process_test_masked = partial(process_test_file, config_dict)

    # Print task assignments
    print(f"Rank {rank} assigned: {len(local_train_file_pairs)} training files, "
          f"{len(local_val_file_pairs)} validation files, "
          f"{len(local_test_masked_info)} test/masked files.")

    # Process all files in parallel
    with ProcessPoolExecutor(max_workers=num_cpus) as pool:
        # Submit all jobs
        all_futures = []
        
        # Process training files
        for fi in local_train_file_pairs:
            future = pool.submit(process_train_val, fi, 'train')
            all_futures.append(future)
        
        # Process validation files
        for fi in local_val_file_pairs:
            future = pool.submit(process_train_val, fi, 'val')
            all_futures.append(future)
        
        # Process test and masked files
        for fi in local_test_masked_info:
            future = pool.submit(process_test_masked, fi)
            all_futures.append(future)
        
        # Wait for all futures to complete
        for future in all_futures:
            try:
                future.result()
            except Exception as e:
                print(f"Rank {rank}: A task failed with error: {str(e)}")

    print(f"Rank {rank}: All processing complete.")

if __name__ == "__main__":
    main()
    sys.exit(0)
