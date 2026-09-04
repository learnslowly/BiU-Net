import pandas as pd
import numpy as np
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description='Add missingness to genotype data')
    parser.add_argument('--dataset', type=str, default='SGDP', help="Dataset")
    parser.add_argument('--chromosome', type=int, default=22, help="Chromosome number")
    parser.add_argument('--missingRatios', nargs='+', type=float, default=[0.05, 0.15, 0.25, 0.5], help="List of missing data percentages")
    args = parser.parse_args()
    args.random_states = [0, 42, 1024]  # Fixed list of random states
    return args

# Add missing site by site, proportionally
def add_missingness(df, missing_perc, random_state):
    """
    Add missingness to genotype data, with masking proportional to variant occurrences.
    Uses a global approach for very small sample sizes.
    
    Args:
        df: Input dataframe where 1=0|0, 2=0|1, 3=1|0, 4=1|1
        missing_perc: Percentage of values to mask (between 0 and 1)
        random_state: Random seed for reproducibility
    """
    np.random.seed(random_state)
    df_missing = df.copy()

    # If sample size is small (fewer than 20 columns, reciprocal to the smallest 5% missing), use global masking approach
    if df.shape[1] < 20:
        # Convert to numpy array for faster operations
        data = df_missing.values
        
        # Identify all variant and non-variant positions
        variant_mask = (data == 2) | (data == 3) | (data == 4)
        nonvariant_mask = (data == 1)
        
        variant_positions = np.where(variant_mask)
        nonvariant_positions = np.where(nonvariant_mask)
        
        # Calculate total number of elements to mask
        total_elements = data.size
        total_to_mask = int(total_elements * missing_perc)
        
        # Calculate proportions
        num_variants = len(variant_positions[0])
        num_nonvariants = len(nonvariant_positions[0])
        total_valid = num_variants + num_nonvariants
        
        if total_valid > 0:
            variant_prop = num_variants / total_valid
            
            # Calculate number to mask for each type
            n_variants_to_mask = int(total_to_mask * variant_prop)
            n_nonvariants_to_mask = total_to_mask - n_variants_to_mask
            
            # Adjust if necessary
            n_variants_to_mask = min(n_variants_to_mask, num_variants)
            n_nonvariants_to_mask = min(n_nonvariants_to_mask, num_nonvariants)
            
            # Select random indices to mask
            if n_variants_to_mask > 0:
                variant_indices = np.random.choice(
                    np.arange(num_variants), 
                    size=n_variants_to_mask, 
                    replace=False
                )
                v_rows = variant_positions[0][variant_indices]
                v_cols = variant_positions[1][variant_indices]
                data[v_rows, v_cols] = 0
            
            if n_nonvariants_to_mask > 0:
                nonvariant_indices = np.random.choice(
                    np.arange(num_nonvariants), 
                    size=n_nonvariants_to_mask, 
                    replace=False
                )
                nv_rows = nonvariant_positions[0][nonvariant_indices]
                nv_cols = nonvariant_positions[1][nonvariant_indices]
                data[nv_rows, nv_cols] = 0
    else:
        # Use the original row-by-row approach for larger sample sizes
        data = df_missing.values
        row_lengths = np.full(len(df), df.shape[1])
        n_to_mask = np.round(row_lengths * missing_perc).astype(int)
        
        for i in range(len(df)):
            row = data[i]
            variant_mask = (row == 2) | (row == 3) | (row == 4)
            variant_positions = np.where(variant_mask)[0]
            nonvariant_positions = np.where(row == 1)[0]
            
            if len(variant_positions) > 0:
                variant_prop = len(variant_positions) / len(row)
                n_variants_to_mask = round(n_to_mask[i] * variant_prop)
                n_nonvariants_to_mask = n_to_mask[i] - n_variants_to_mask
                
                if n_variants_to_mask > 0:
                    mask_idx = np.random.choice(
                        variant_positions,
                        size=min(n_variants_to_mask, len(variant_positions)),
                        replace=False
                    )
                    data[i, mask_idx] = 0
                
                if n_nonvariants_to_mask > 0 and len(nonvariant_positions) > 0:
                    mask_idx = np.random.choice(
                        nonvariant_positions,
                        size=min(n_nonvariants_to_mask, len(nonvariant_positions)),
                        replace=False
                    )
                    data[i, mask_idx] = 0
    
    return pd.DataFrame(data, index=df.index, columns=df.columns)

# Obsolete, add missing globally
def add_missingness_(df, missing_perc, random_state):
    """
    Randomly replace elements with 0 based on missing percentage, using a random state for reproducibility.
    """
    df_missing = df.copy()
    np.random.seed(random_state)
    mask = np.random.random(size=df.shape) < missing_perc
    df_missing[mask] = 0
    return df_missing

def process_single_case(args, population, random_state, missing_ratio):
    """
    Process a single combination of population, random state, and missing ratio.
    """
    # Define input and output paths
    test_csv_gz = f'./split/{args.dataset}_chr{args.chromosome}_{population}_test.csv.gz'
    mask_dir = f"./masked/{random_state}/{missing_ratio * 100:.0f}%"
    os.makedirs(mask_dir, exist_ok=True)
    
    masked_csv_gz = os.path.join(
        mask_dir,
        f"{args.dataset}_chr{args.chromosome}_{population}_missing{missing_ratio * 100:.0f}%_masked.csv.gz"
    )
    
    try:
        # Load the original data
        genotype = pd.read_csv(test_csv_gz, compression='gzip', index_col=0)
        
        # Add missingness
        genotype_missing = add_missingness(
            genotype,
            missing_ratio,
            random_state=random_state
        )
        
        # Save the masked data
        genotype_missing.to_csv(masked_csv_gz, compression='gzip', index=True)
        print(f"Processed {population} with random state {random_state}, missing ratio: {missing_ratio * 100:.0f}%")
        print(f"Saved to: {masked_csv_gz}")
        
    except FileNotFoundError:
        print(f"Warning: File not found for population {population}: {test_csv_gz}")
    except Exception as e:
        print(f"Error processing {population} with random state {random_state}, ratio {missing_ratio}: {str(e)}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Define all populations to process
    populations = ["ALL", "Africa", "America", "CentralAsiaSiberia", "EastAsia", "Oceania", "SouthAsia", "WestEurasia"]
    
    # Create all combinations of populations, random states, and missing ratios
    tasks = [
        (args, pop, rand_state, ratio)
        for pop in populations
        for rand_state in args.random_states
        for ratio in args.missingRatios
    ]
    
    # Process all combinations in parallel
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_single_case, *task)
            for task in tasks
        ]
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    # # Process all combinations sequentially
    # for task in tasks:
    #     process_single_case(*task)

    
    print("All populations, random states, and missing ratios processed successfully.")

if __name__ == "__main__":
    main()
