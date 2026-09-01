# 1KGP Pipeline Quick Reference

## One-Liner Grammar (ALL population, all chr1-22)

```bash
cd $GENOTYPE_DATA_DIR/1KGP

# Step 0: Generate references
for CHR in {1..22}; do sbatch --export=CHR=$CHR --array=0 --output=ref/logs/chr${CHR}_ref.out 00_gen_ref_array.batch; done

# Step 1: Preprocess (QC, phasing, splitting)
for CHR in {1..22}; do sbatch --export=CHR=$CHR,POPULATIONS="ALL",HLA_MODE="none" --array=0 --output=qc/logs/chr${CHR}_ALL.out 01_preprocess.batch; done

# Step 2: VCF → CSV
for CHR in {1..22}; do sbatch --export=CHR=$CHR,POPULATIONS="ALL",HLA_MODE="none" --array=0-2 --output=split/vcf2csv_chr${CHR}_%a.out 02_vcf2csv.batch; done

# Step 3: Add masking
for CHR in {1..22}; do sbatch --export=CHR=$CHR,HLA_MODE="none" --array=0 --output=masked/masking_chr${CHR}.out 03_masking.batch; done

# Step 4: Masked CSV → VCF
for CHR in {1..22}; do sbatch --export=CHR=$CHR,POPULATIONS="ALL",HLA_MODE="none" --array=0-11 --output=masked/mask_vcf_chr${CHR}_%a.out 04_maskedCSV2VCF.batch; done

# Step 5: Chunking
for CHR in {1..22}; do sbatch --export=CHR=$CHR,HLA_MODE="none" --array=0 --output=chunked/chunking_chr${CHR}.out 05_chunking.batch; done
```

## Single Chromosome Example (chr22)

```bash
CHR=22
POPS="ALL"

sbatch --export=CHR=$CHR --array=0 --output=ref/logs/chr${CHR}_ref.out 00_gen_ref_array.batch
sbatch --export=CHR=$CHR,POPULATIONS="$POPS",HLA_MODE="none" --array=0 --output=qc/logs/chr${CHR}_ALL.out 01_preprocess.batch
sbatch --export=CHR=$CHR,POPULATIONS="$POPS",HLA_MODE="none" --array=0-2 --output=split/vcf2csv_chr${CHR}_%a.out 02_vcf2csv.batch
sbatch --export=CHR=$CHR,HLA_MODE="none" --array=0 --output=masked/masking_chr${CHR}.out 03_masking.batch
sbatch --export=CHR=$CHR,POPULATIONS="$POPS",HLA_MODE="none" --array=0-11 --output=masked/mask_vcf_chr${CHR}_%a.out 04_maskedCSV2VCF.batch
sbatch --export=CHR=$CHR,HLA_MODE="none" --array=0 --output=chunked/chunking_chr${CHR}.out 05_chunking.batch
```

## Chr6 HLA Region (Optional)

```bash
sbatch --export=CHR=6,HLA_ONLY=true --array=0 --output=ref/logs/chr6_HLA_ref.out 00_gen_ref_array.batch
sbatch --export=CHR=6,POPULATIONS="ALL",HLA_MODE="hla_only" --array=0 --output=qc/logs/chr6_HLA_ALL.out 01_preprocess.batch
sbatch --export=CHR=6,POPULATIONS="ALL",HLA_MODE="hla_only" --array=0-2 --output=split/vcf2csv_chr6_HLA_%a.out 02_vcf2csv.batch
sbatch --export=CHR=6,HLA_MODE="hla_only" --array=0 --output=masked/masking_chr6_HLA.out 03_masking.batch
sbatch --export=CHR=6,POPULATIONS="ALL",HLA_MODE="hla_only" --array=0-11 --output=masked/mask_vcf_chr6_HLA_%a.out 04_maskedCSV2VCF.batch
sbatch --export=CHR=6,HLA_MODE="hla_only" --array=0 --output=chunked/chunking_chr6_HLA.out 05_chunking.batch
```

## Pipeline Steps

| Step | Script | Purpose | Outputs |
|------|--------|---------|---------|
| 0 | 00_gen_ref_array.batch | Generate reference VCFs | ref/1KGP_chr*_ALL_ref.vcf.gz |
| 1 | 01_preprocess.batch | QC, phasing, splitting | split/1KGP_chr*_ALL_{train,val,test}.vcf.gz |
| 2 | 02_vcf2csv.batch | Convert VCF to CSV | split/1KGP_chr*_ALL_{train,val,test}.csv.gz |
| 3 | 03_masking.batch | Add missingness to test | masked/*/1KGP_chr*_ALL_missing*_masked.csv.gz |
| 4 | 04_maskedCSV2VCF.batch | Convert masked CSV to VCF | masked/*/1KGP_chr*_ALL_missing*_masked.vcf.gz |
| 5 | 05_chunking.batch | Chunk train/val CSVs | chunked/1KGP_chr*_ALL_{train,val}_chunk*.csv.gz |

## Parameters

All scripts now support:
- `CHR=N` - Chromosome number (default: array task ID or 6)
- `POPULATIONS="ALL"` - Space-separated populations (default: ALL)
- `HLA_MODE="none|auto|hla_only"` - HLA region handling (default: auto)

## Verification Commands

```bash
# Check references (should be 22)
ls -1 ref/1KGP_chr*_ALL_ref.vcf.gz | wc -l

# Check split VCFs (should be 66)
ls -1 split/1KGP_chr*_ALL_*.vcf.gz | grep -v HLA | wc -l

# Check split CSVs (should be 66)
ls -1 split/1KGP_chr*_ALL_*.csv.gz | grep -v HLA | wc -l

# Check masked files (should be 264)
find masked -name "1KGP_chr*_ALL_missing*_masked.csv.gz" | grep -v HLA | wc -l
find masked -name "1KGP_chr*_ALL_missing*_masked.vcf.gz" | grep -v HLA | wc -l

# Check chunked files (variable)
ls -1 chunked/1KGP_chr*_ALL_*_chunk*.csv.gz | grep -v HLA | wc -l
```

## Monitor Progress

```bash
# Check running jobs
squeue -u $USER

# Watch job count
watch -n 60 'squeue -u $USER | wc -l'

# Check for errors
grep -r "ERROR\|Error\|error" qc/logs/ split/*.out masked/*.out chunked/*.out
```
