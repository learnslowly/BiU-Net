import pandas as pd
import numpy as np
import argparse
import os
import gzip


def parse_args():
    parser = argparse.ArgumentParser(description='Add missingness to genotype data')
    parser.add_argument('--dataset', type=str, default='LOS', help="Dataset")
    parser.add_argument('--chromosome', type=int, default=22, help="Chromosome number")
    parser.add_argument('--hla', action='store_true', help='Use HLA region suffix (_HLA) for chromosome 6')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Number of variants per chunk (default: 10000)')

    # For single combination mode (array job)
    parser.add_argument('--population', type=str, default=None, help='Single population to process (CA, AA, or ALL)')
    parser.add_argument('--random-state', type=int, default=None, help='Single random state to use')
    parser.add_argument('--missing-ratio', type=float, default=None, help='Single missing ratio (0.05, 0.15, 0.25, or 0.5)')

    # For batch mode (all combinations)
    parser.add_argument('--all', action='store_true', help='Process all combinations (default if no single params given)')

    return parser.parse_args()


def add_missingness_to_chunk(data, missing_perc, rng):
    """
    Add missingness to a chunk of genotype data (numpy array).
    Masking is proportional to variant vs non-variant occurrences.
    """
    num_samples = data.shape[1]

    for i in range(len(data)):
        row = data[i]
        n_to_mask = int(round(num_samples * missing_perc))

        if n_to_mask == 0:
            continue

        variant_positions = np.where((row == 2) | (row == 3) | (row == 4))[0]
        nonvariant_positions = np.where(row == 1)[0]

        num_variants = len(variant_positions)
        num_nonvariants = len(nonvariant_positions)
        total_valid = num_variants + num_nonvariants

        if total_valid == 0:
            continue

        variant_prop = num_variants / total_valid
        n_variants_to_mask = int(round(n_to_mask * variant_prop))
        n_nonvariants_to_mask = n_to_mask - n_variants_to_mask

        n_variants_to_mask = min(n_variants_to_mask, num_variants)
        n_nonvariants_to_mask = min(n_nonvariants_to_mask, num_nonvariants)

        if n_variants_to_mask > 0:
            mask_idx = rng.choice(variant_positions, size=n_variants_to_mask, replace=False)
            data[i, mask_idx] = 0

        if n_nonvariants_to_mask > 0 and num_nonvariants > 0:
            mask_idx = rng.choice(nonvariant_positions, size=n_nonvariants_to_mask, replace=False)
            data[i, mask_idx] = 0

    return data


def process_single_combination(input_csv, output_csv, missing_ratio, random_state, chunk_size=10000):
    """
    Process a single file with a single missing ratio and random state.
    Uses chunked streaming to minimize memory usage.
    """
    rng = np.random.default_rng(random_state)

    first_chunk = True
    total_variants = 0

    reader = pd.read_csv(input_csv, compression='gzip', index_col=0, chunksize=chunk_size)

    with gzip.open(output_csv, 'wt') as gz_out:
        for chunk_num, chunk in enumerate(reader):
            data = chunk.values.astype(np.uint8)
            masked_data = add_missingness_to_chunk(data, missing_ratio, rng)

            masked_chunk = pd.DataFrame(
                masked_data,
                index=chunk.index,
                columns=chunk.columns
            )

            masked_chunk.to_csv(gz_out, header=first_chunk, mode='a')
            first_chunk = False

            total_variants += len(chunk)

    return total_variants


def main():
    args = parse_args()

    suffix = "_HLA" if args.hla else ""

    # Define all options
    all_populations = ['CA', 'AA', 'ALL']
    all_random_states = [0, 42, 1024]
    all_missing_ratios = [0.05, 0.15, 0.25, 0.5]

    # Determine if running single combination or all
    single_mode = (args.population is not None or
                   args.random_state is not None or
                   args.missing_ratio is not None)

    if single_mode:
        # Single combination mode
        if args.population is None or args.random_state is None or args.missing_ratio is None:
            print("Error: For single mode, must specify --population, --random-state, and --missing-ratio")
            return

        populations = [args.population]
        random_states = [args.random_state]
        missing_ratios = [args.missing_ratio]
    else:
        # All combinations mode
        populations = all_populations
        random_states = all_random_states
        missing_ratios = all_missing_ratios


    for population in populations:
        input_csv = f'./split/{args.dataset}_chr{args.chromosome}_{population}_test{suffix}.csv.gz'

        if not os.path.exists(input_csv):
            print(f"Warning: File not found: {input_csv}")
            continue

        for random_state in random_states:
            for missing_ratio in missing_ratios:
                mask_dir = f"./masked/{random_state}/{missing_ratio * 100:.0f}%"
                os.makedirs(mask_dir, exist_ok=True)

                output_csv = os.path.join(
                    mask_dir,
                    f"{args.dataset}_chr{args.chromosome}_{population}_missing{missing_ratio * 100:.0f}%_masked{suffix}.csv.gz"
                )

                try:
                    num_variants = process_single_combination(
                        input_csv, output_csv, missing_ratio, random_state, args.chunk_size
                    )
                    print(f"[CHR{args.chromosome}] {population}/rs{random_state}/{missing_ratio*100:.0f}% -> {num_variants} variants")
                except Exception as e:
                    print(f"[CHR{args.chromosome}] {population}/rs{random_state}/{missing_ratio*100:.0f}% ERROR: {str(e)}")


if __name__ == "__main__":
    main()
