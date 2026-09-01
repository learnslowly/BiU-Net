import os
import pandas as pd
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
import shutil
import argparse

def is_chunk_file(filename):
    """Check if the file is a chunk file based on its name."""
    return 'chunk' in filename.name.lower()

def partition_files_among_tasks(files, num_tasks, rank):
    """Partition files among tasks in a round-robin manner."""
    return [files[i] for i in range(len(files)) if i % num_tasks == rank]

def split2chunks(args):
    """Process a single file, splitting it into chunks or renaming it if small."""
    file_path, chunk_size, output_dir = args

    if is_chunk_file(file_path):
        return f"Skipped chunk file: {file_path}"

    base_name = file_path.stem.replace('.csv', '')
    print(f"\nProcessing: {file_path}")

    try:
        # Read the input CSV file
        df = pd.read_csv(file_path, compression='gzip', index_col=0)

        # Get the total number of columns
        total_columns = df.shape[1]

        # If chunk_size >= total_columns, copy the original and rename it
        if chunk_size >= total_columns:
            chunk_filename = f"{base_name}_chunk000.csv.gz"
            chunk_path = Path(output_dir) / chunk_filename
            shutil.copy(file_path, chunk_path)
            return f"Renamed {file_path} to {chunk_filename}"

        # Split columns into chunks
        chunk_number = 0
        for i in range(0, total_columns, chunk_size):
            chunk = df.iloc[:, i:i+chunk_size]
            chunk_filename = f"{base_name}_chunk{chunk_number:03}.csv.gz"
            chunk_path = Path(output_dir) / chunk_filename
            chunk.to_csv(chunk_path, index=True, compression='gzip')
            print(f"Saved chunk {chunk_number} to {chunk_filename}")
            chunk_number += 1

        return f"Successfully processed {file_path}"

    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

def process_files(files, chunk_size, output_dir, num_cpus):
    """Process files in parallel within a task."""
    os.makedirs(output_dir, exist_ok=True)

    with Pool(processes=num_cpus) as pool:
        results = pool.map(split2chunks, [(file, chunk_size, output_dir) for file in files])

    for result in results:
        print(result)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Chunk CSV files for training')
    parser.add_argument('--chr', type=int, default=None, help='Chromosome number to process')
    parser.add_argument('--suffix', type=str, default='', help='Region suffix (e.g., _HLA for chromosome 6)')
    parser.add_argument('--chunk_size', type=int, default=10000, help='Number of columns per chunk')
    args = parser.parse_args()

    # Get chromosome from args or environment
    chr = args.chr if args.chr is not None else int(os.environ.get('CHR', 6))

    # Get suffix from args or auto-detect
    # Note: args.suffix can be None (not provided) or "" (empty string provided)
    if args.suffix is not None:
        region_suffix = args.suffix  # Use provided suffix (including empty string)
    else:
        region_suffix = "_HLA" if chr == 6 else ""  # Auto-detect only if not provided

    root_directory = "./split/"
    output_dir = "./chunked/"
    os.makedirs(output_dir, exist_ok=True)
    chunk_size = args.chunk_size
    rank = int(os.environ.get('SLURM_PROCID', 0))
    num_tasks = int(os.environ.get('SLURM_NTASKS', 1))
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    print(f"============================================")
    print(f"Chunking chromosome {chr} files")
    print(f"Region suffix: '{region_suffix}'")
    print(f"Chunk size: {chunk_size}")
    print(f"Search pattern: *chr{chr}_*{region_suffix}.csv.gz")
    print(f"============================================")

    # Find all CSV files matching the pattern (includes HLA suffix if applicable)
    search_pattern = f"*chr{chr}_*{region_suffix}.csv.gz"
    all_csv_files = list(Path(root_directory).rglob(search_pattern))
    print(f"Found {len(all_csv_files)} files matching pattern before filtering")
    all_csv_files = [f for f in all_csv_files if ('train' in f.name or 'val' in f.name) and not is_chunk_file(f)]
    print(f"Found {len(all_csv_files)} files after filtering for train/val")

    local_csv_file_list = partition_files_among_tasks(all_csv_files, num_tasks, rank)

    print(f"Task {rank}: Processing {len(local_csv_file_list)} files with {num_cpus} CPUs")
    process_files(local_csv_file_list, chunk_size, output_dir, num_cpus)

if __name__ == "__main__":
    main()
