import os
import pandas as pd
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
import shutil

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
    root_directory = "./split/"
    output_dir = "./chunked/"
    os.makedirs(output_dir, exist_ok=True)

    chunk_size = 1001
    rank = int(os.environ.get('SLURM_PROCID', 0))
    num_tasks = int(os.environ.get('SLURM_NTASKS', 1))
    num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

    all_csv_files = list(Path(root_directory).rglob("*.csv.gz"))
    all_csv_files = [f for f in all_csv_files if ('train' in f.name or 'val' in f.name) and not is_chunk_file(f)]

    local_csv_file_list = partition_files_among_tasks(all_csv_files, num_tasks, rank)

    print(f"Task {rank}: Processing {len(local_csv_file_list)} files with {num_cpus} CPUs")
    process_files(local_csv_file_list, chunk_size, output_dir, num_cpus)

if __name__ == "__main__":
    main()
