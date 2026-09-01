import pandas as pd
import gzip
import numpy as np
import argparse

def create_mask_and_process(original_vcf, masked_csv, output_vcf):
    """Process VCF using vectorized operations with binary mask."""
    print("Reading masked CSV to create binary mask...")
    # Read CSV and create position-based mask dictionary
    masked_data = pd.read_csv(masked_csv, compression='gzip', index_col=0)
    
    # Create dictionary of positions to mask for each sample
    mask_dict = {}
    for pos in masked_data.index:
        mask_dict[str(pos)] = masked_data.loc[pos] == 0 # 0 repr ./.
    
    del masked_data  # Free memory
    
    print("Processing VCF with position matching...")
    total_variants = 0
    masked_variants = 0
    
    # Process the VCF and write output
    with gzip.open(original_vcf, 'rt', encoding='utf-8') as vcf_in, \
         open(output_vcf, 'w', encoding='utf-8', newline='\n') as vcf_out:
        
        # Copy header and get sample index
        for line in vcf_in:
            if line.startswith('#'):
                vcf_out.write(line)
                if line.startswith('#CHROM'):
                    sample_start_idx = 9
                    break
        
        # Process VCF data line by line
        buffer = []
        buffer_size = 1000
        
        for line in vcf_in:
            fields = line.strip().split('\t')
            pos = fields[1]
            total_variants += 1
            
            if pos in mask_dict:
                # Get mask for current position
                mask_row = mask_dict[pos]
                
                # Get original genotypes
                genotypes = np.array(fields[sample_start_idx:])
                
                # Count masked variants before applying mask
                masked_variants += np.sum(mask_row)
                
                # Apply mask only to positions that need masking
                genotypes[mask_row] = './.'
                
                # Update genotype fields
                fields[sample_start_idx:] = genotypes
            
            # Add to buffer
            buffer.append('\t'.join(fields) + '\n')
            
            # Write buffer when it's full
            if len(buffer) >= buffer_size:
                vcf_out.writelines(buffer)
                buffer = []
            
            if total_variants % 50000 == 0:
                print(f"Processed {total_variants} variants...")
        
        # Write remaining lines in buffer
        if buffer:
            vcf_out.writelines(buffer)
    
    # Print final statistics
    total_genotypes = total_variants * (len(fields) - sample_start_idx)
    masking_percentage = (masked_variants / total_genotypes) * 100
    print(f"\nMasking Statistics:")
    print(f"Total variants processed: {total_variants}")
    print(f"Total genotypes: {total_genotypes}")
    print(f"Masked genotypes: {masked_variants}")
    print(f"Masking percentage: {masking_percentage:.2f}%")

def main():
    parser = argparse.ArgumentParser(
        description="Mask VCF using position-matched operations")
    parser.add_argument("--original", required=True, 
                       help="Path to the original VCF file")
    parser.add_argument("--masked", required=True, 
                       help="Path to the masked CSV file")
    parser.add_argument("--output", required=True, 
                       help="Path to the output VCF file")
    
    args = parser.parse_args()
    
    print(f"Processing {args.masked} -> {args.output}")
    create_mask_and_process(args.original, args.masked, args.output)
    print("Done!")

if __name__ == "__main__":
    main()
