import gzip
import pandas as pd
import argparse
import re


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
    pattern = re.compile(r'\d+$')
    # Rename the columns
    genotype.columns = [
        pattern.search(col).group() if pattern.search(col) else col
        for col in genotype.columns
    ]

    # Write the result to a gzipped CSV file, no need to convert the indices and columns to integers since they will still be stored as string
    with gzip.open(out_file, 'wt') as f:
        genotype.to_csv(f, index=True)
    # Return the statistics (number of samples and SNPs)
    return genotype.shape[1], genotype.shape[0]

def main(chromosome, population, subset, suffix=''):
    dataset = '1KGP'

    # Process a single file (suffix appended before file extension)
    vcf_file = f"./split/{dataset}_chr{chromosome}_{population}_{subset}{suffix}.vcf.gz"
    csv_gz_file = f"./split/{dataset}_chr{chromosome}_{population}_{subset}{suffix}.csv.gz"

    num_samples, num_snps = parse_vcf(vcf_file, csv_gz_file)

    # Print results
    print(f"{csv_gz_file} contains {num_samples} samples and {num_snps} SNPs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a single VCF file for a specific chromosome, population, and subset')
    parser.add_argument('chromosome', type=int, help='Chromosome number to process')
    parser.add_argument('population', type=str, help='Population to process')
    parser.add_argument('subset', type=str, help='Subset to process (train, val, test)')
    parser.add_argument('--suffix', type=str, default='', help='Region suffix (e.g., _HLA for chromosome 6)')

    args = parser.parse_args()
    main(args.chromosome, args.population, args.subset, args.suffix)
