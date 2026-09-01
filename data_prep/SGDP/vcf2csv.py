import gzip
import pandas as pd
import argparse
import json

def parse_vcf(in_file, out_file):
    # Read and parse the VCF file
    with gzip.open(in_file, 'rt') as f:
        lines = [line for line in f if not line.startswith('##')]
    header = lines[0].split()
    data = lines[1:]
    data = [line.split() for line in data]
    genotype = pd.DataFrame(data, columns=header)
    
    # Select the required columns: POS and genotype fields
    genotype = pd.concat([genotype.iloc[:, [1]], genotype.iloc[:, 9:]], axis=1)
    
    # Define the conversion dictionary
    conversion_dict = {
        #'0/0': 1,
        '0|0': 1,
        #'0/1': 2,
        '0|1': 2,
        #'1/0': 2,
        '1|0': 3,
        #'1/1': 3,
        '1|1': 4
    }
    
    # Apply the conversion to genotype columns using .assign and pd.to_numeric
    genotype = genotype.assign(
        **{col: pd.to_numeric(genotype[col].map(conversion_dict), errors='coerce', downcast='integer')
           for col in genotype.columns[1:]}
    )
    genotype = genotype.set_index('POS')
    
    # Rename the columns
    # Create a sorted list of original names to ensure consistent mapping
    original_names = sorted(genotype.columns)
    
    # Create bidirectional mappings using dictionaries
    name_to_int = {name: i+1 for i, name in enumerate(original_names)}
    int_to_name = {i+1: name for i, name in enumerate(original_names)}
    
    # Convert column names to integers
    genotype.columns = [name_to_int[col] for col in genotype.columns]
    
    # Save the mappings for future use - create unique mapping file based on output filename
    mapping_file = out_file.replace('.csv.gz', '_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump({'name_to_int': name_to_int, 'int_to_name': int_to_name}, f, indent=2)
    
    # Write the result to a gzipped CSV file
    with gzip.open(out_file, 'wt') as f:
        genotype.to_csv(f, index=True)

    # Return the statistics (number of samples and SNPs)
    return genotype.shape[1], genotype.shape[0]

def main(chromosome, population, subset):
    dataset = 'HLA'
    
    # Process a single file
    vcf_file = f"./split/{dataset}_chr{chromosome}_{population}_{subset}.vcf.gz"
    csv_gz_file = f"./split/{dataset}_chr{chromosome}_{population}_{subset}.csv.gz"
    
    num_samples, num_snps = parse_vcf(vcf_file, csv_gz_file)
    
    # Print results
    print(f"{csv_gz_file} contains {num_samples} samples and {num_snps} SNPs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a single VCF file for a specific chromosome, population, and subset')
    parser.add_argument('chromosome', type=int, help='Chromosome number to process')
    parser.add_argument('population', type=str, help='Population to process')
    parser.add_argument('subset', type=str, help='Subset to process (train, val, test)')
    
    args = parser.parse_args()
    main(args.chromosome, args.population, args.subset)
