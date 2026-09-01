import os
import gzip
import pandas as pd
from pathlib import Path
import shutil


def is_chunk_file(filename):
    """Check if the file is a chunk file based on its name."""
    return 'chunk' in filename.name.lower()


def is_hla_file(filename):
    """Check if the file is an HLA file based on its name."""
    return '_HLA' in filename.name


def filter_files_by_hla(files, hla_filter):
    """
    Filter files based on HLA mode.

    Args:
        files: List of file paths
        hla_filter: "only" for HLA files only, "none" for non-HLA only, "all" for both
    """
    if hla_filter == "only":
        return [f for f in files if is_hla_file(f)]
    elif hla_filter == "none":
        return [f for f in files if not is_hla_file(f)]
    else:  # "all"
        return files


def get_csv_info(file_path):
    """Get column names and count rows without loading entire file."""
    with gzip.open(file_path, 'rt') as f:
        header = f.readline().strip().split(',')
        columns = header[1:]  # First column is index (POS)

        # Count rows
        num_rows = sum(1 for _ in f)

    return columns, num_rows


def split_file_streaming(file_path, chunk_size, output_dir, row_chunk_size=10000):
    """
    Split a CSV file into column chunks using streaming.

    Memory usage: O(row_chunk_size * columns_per_chunk) instead of O(total_rows * total_columns)
    """
    if is_chunk_file(file_path):
        return f"Skipped chunk file: {file_path}"

    base_name = file_path.stem.replace('.csv', '')
    print(f"\nProcessing: {file_path}")

    try:
        # Get file info
        columns, num_rows = get_csv_info(file_path)
        total_columns = len(columns)
        print(f"  Rows: {num_rows}, Columns: {total_columns}")

        # If chunk_size >= total_columns, just copy the file
        if chunk_size >= total_columns:
            chunk_filename = f"{base_name}_chunk000.csv.gz"
            chunk_path = Path(output_dir) / chunk_filename
            shutil.copy(file_path, chunk_path)
            return f"Copied {file_path} to {chunk_filename} (single chunk)"

        # Calculate number of column chunks
        num_chunks = (total_columns + chunk_size - 1) // chunk_size
        print(f"  Will create {num_chunks} chunks of up to {chunk_size} columns each")

        # Process each column chunk
        for chunk_num in range(num_chunks):
            col_start = chunk_num * chunk_size
            col_end = min((chunk_num + 1) * chunk_size, total_columns)
            chunk_columns = columns[col_start:col_end]

            chunk_filename = f"{base_name}_chunk{chunk_num:03}.csv.gz"
            chunk_path = Path(output_dir) / chunk_filename

            print(f"  Creating chunk {chunk_num}: columns {col_start}-{col_end-1}")

            # Stream read and write for this column chunk
            with gzip.open(file_path, 'rt') as f_in, \
                 gzip.open(chunk_path, 'wt') as f_out:

                # Read and write header
                header_line = f_in.readline().strip().split(',')
                index_name = header_line[0]

                # Write header for this chunk
                chunk_header = [index_name] + chunk_columns
                f_out.write(','.join(chunk_header) + '\n')

                # Process rows in batches
                rows_written = 0
                buffer = []
                buffer_limit = 1000

                for line in f_in:
                    fields = line.strip().split(',')
                    index_val = fields[0]

                    # Extract only the columns for this chunk (+1 for index offset)
                    chunk_values = [fields[i + 1] for i in range(col_start, col_end)]
                    buffer.append(index_val + ',' + ','.join(chunk_values) + '\n')

                    if len(buffer) >= buffer_limit:
                        f_out.writelines(buffer)
                        rows_written += len(buffer)
                        buffer = []

                # Write remaining buffer
                if buffer:
                    f_out.writelines(buffer)
                    rows_written += len(buffer)

            print(f"    Saved {chunk_filename} ({rows_written} rows, {len(chunk_columns)} columns)")

        return f"Successfully processed {file_path} into {num_chunks} chunks"

    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"


def main():
    root_directory = "./split/"
    output_dir = "./chunked/"
    os.makedirs(output_dir, exist_ok=True)

    chunk_size = int(os.environ.get('CHUNK_SIZE', 25600))
    hla_filter = os.environ.get('HLA_FILTER', 'all')

    # For SLURM: partition files among tasks
    rank = int(os.environ.get('SLURM_PROCID', 0))
    num_tasks = int(os.environ.get('SLURM_NTASKS', 1))

    # Find all CSV files
    all_csv_files = list(Path(root_directory).rglob("*.csv.gz"))
    all_csv_files = [f for f in all_csv_files if ('train' in f.name or 'val' in f.name) and not is_chunk_file(f)]

    # Apply HLA filter
    all_csv_files = filter_files_by_hla(all_csv_files, hla_filter)

    # Sort for consistent ordering across tasks
    all_csv_files = sorted(all_csv_files)

    print(f"HLA filter mode: {hla_filter}")
    print(f"Chunk size: {chunk_size} columns")
    print(f"Total files to process: {len(all_csv_files)}")

    # Partition files among SLURM tasks
    local_files = [f for i, f in enumerate(all_csv_files) if i % num_tasks == rank]
    print(f"Task {rank}/{num_tasks}: Processing {len(local_files)} files")

    # Process files sequentially (no multiprocessing - memory safe)
    for file_path in local_files:
        result = split_file_streaming(file_path, chunk_size, output_dir)
        print(result)

    print(f"\nTask {rank} complete.")


if __name__ == "__main__":
    main()
