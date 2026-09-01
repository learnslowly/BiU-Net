import gzip
import argparse


def create_mask_and_process_streaming(original_vcf, masked_csv, output_vcf):
    """
    Process VCF using streaming - read both files line by line in parallel.

    Assumes masked CSV and VCF have same positions in same order.
    Memory usage: O(1) - only one line at a time.
    """
    print("Processing VCF with streaming position matching...")
    total_variants = 0
    masked_genotypes = 0
    num_samples = 0

    # Open both input files and output file
    with gzip.open(original_vcf, 'rt', encoding='utf-8') as vcf_in, \
         gzip.open(masked_csv, 'rt', encoding='utf-8') as csv_in, \
         open(output_vcf, 'w', encoding='utf-8', newline='\n') as vcf_out:

        # Skip CSV header and get sample names
        csv_header = csv_in.readline().strip().split(',')
        csv_samples = csv_header[1:]  # First column is POS/index
        num_samples = len(csv_samples)
        print(f"CSV has {num_samples} samples")

        # Copy VCF header
        sample_start_idx = 9
        for line in vcf_in:
            if line.startswith('#'):
                vcf_out.write(line)
                if line.startswith('#CHROM'):
                    break
            else:
                break

        # Process both files line by line
        buffer = []
        buffer_size = 1000

        for vcf_line in vcf_in:
            csv_line = csv_in.readline()

            if not csv_line:
                print(f"Warning: CSV ended before VCF at variant {total_variants}")
                break

            # Parse VCF line
            vcf_fields = vcf_line.strip().split('\t')
            vcf_pos = vcf_fields[1]

            # Parse CSV line
            csv_fields = csv_line.strip().split(',')
            csv_pos = csv_fields[0]
            mask_values = csv_fields[1:]

            # Verify positions match
            if vcf_pos != csv_pos:
                print(f"Warning: Position mismatch at variant {total_variants}: VCF={vcf_pos}, CSV={csv_pos}")
                # Try to continue anyway

            total_variants += 1

            # Apply mask: where CSV value is 0, set VCF genotype to ./.
            genotypes = vcf_fields[sample_start_idx:]
            for i, mask_val in enumerate(mask_values):
                if i < len(genotypes) and mask_val == '0':
                    genotypes[i] = './.'
                    masked_genotypes += 1

            vcf_fields[sample_start_idx:] = genotypes

            # Add to buffer
            buffer.append('\t'.join(vcf_fields) + '\n')

            # Write buffer when full
            if len(buffer) >= buffer_size:
                vcf_out.writelines(buffer)
                buffer = []

            if total_variants % 50000 == 0:
                print(f"Processed {total_variants} variants...")

        # Write remaining buffer
        if buffer:
            vcf_out.writelines(buffer)

    # Print statistics
    total_genotypes = total_variants * num_samples
    if total_genotypes > 0:
        masking_percentage = (masked_genotypes / total_genotypes) * 100
    else:
        masking_percentage = 0

    print(f"\nMasking Statistics:")
    print(f"Total variants processed: {total_variants}")
    print(f"Total genotypes: {total_genotypes}")
    print(f"Masked genotypes: {masked_genotypes}")
    print(f"Masking percentage: {masking_percentage:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Mask VCF using streaming position-matched operations")
    parser.add_argument("--original", required=True,
                       help="Path to the original VCF file")
    parser.add_argument("--masked", required=True,
                       help="Path to the masked CSV file")
    parser.add_argument("--output", required=True,
                       help="Path to the output VCF file")

    args = parser.parse_args()

    print(f"Processing {args.masked} -> {args.output}")
    create_mask_and_process_streaming(args.original, args.masked, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
