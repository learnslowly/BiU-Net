# 1KGP Genotype Processing Pipeline

Complete pipeline for processing 1000 Genomes Project genotype data.

## Prerequisites

```bash
cd $GENOTYPE_DATA_DIR/1KGP
mkdir -p qc/logs qc/temp split subsets ref
```

## Step 1: Generate Reference Panels

### 1a. Array job for chr1-5, 7-21
```bash
sbatch 00_gen_ref_array.batch
```

This submits array jobs with `#SBATCH --array=1-5,7-21`, processing chromosomes 1-5 and 7-21 in parallel.

### 1b. Separate jobs for chr6 and chr22
```bash
# Chromosome 6 (full)
sbatch --export=CHR=6 --array=0 00_gen_ref_array.batch

# Chromosome 22
sbatch --export=CHR=22 --array=0 00_gen_ref_array.batch

# Chromosome 6 HLA region only
sbatch --export=CHR=6,HLA_ONLY=true --array=0 00_gen_ref_array.batch
```

**Wait for all reference generation jobs to complete before proceeding.**

Check completion:
```bash
ls -lh ref/1KGP_chr*_ALL_ref.vcf.gz
ls -lh ref/HLA_chr6_ALL_ref.vcf.gz
```

Expected: 23 files (chr1-22 full + chr6 HLA)

---

## Step 2: Preprocess Genotypes (QC, Phasing, Population Splitting)

### 2a. Process ALL population for all chromosomes

```bash
# Chromosomes 1-22 (ALL population, full chromosomes)
for CHR in {1..22}; do
    sbatch --export=CHR=$CHR,POPULATIONS="ALL",HLA_MODE="none" \
           --array=0 \
           --output=qc/logs/chr${CHR}_ALL.out \
           01_preprocess.batch
done

# Chromosome 6 HLA region (ALL population)
sbatch --export=CHR=6,POPULATIONS="ALL",HLA_MODE="hla_only" \
       --array=0 \
       --output=qc/logs/chr6_HLA_ALL.out \
       01_preprocess.batch
```

**Expected outputs (for ALL population):**
```
split/1KGP_chr1_ALL_train.vcf.gz
split/1KGP_chr1_ALL_val.vcf.gz
split/1KGP_chr1_ALL_test.vcf.gz
...
split/1KGP_chr22_ALL_train.vcf.gz
split/1KGP_chr22_ALL_val.vcf.gz
split/1KGP_chr22_ALL_test.vcf.gz
split/1KGP_chr6_ALL_train_HLA.vcf.gz
split/1KGP_chr6_ALL_val_HLA.vcf.gz
split/1KGP_chr6_ALL_test_HLA.vcf.gz
```

Total: 69 VCF files (22 chr × 3 splits + 1 HLA region × 3 splits)

### 2b. (Optional) Process specific populations

If you need EUR, AFR, or other specific populations:

```bash
# Example: EUR population for all chromosomes
for CHR in {1..22}; do
    sbatch --export=CHR=$CHR,POPULATIONS="EUR",HLA_MODE="none" \
           --array=0 \
           --output=qc/logs/chr${CHR}_EUR.out \
           01_preprocess.batch
done

# Multiple populations (space-separated)
for CHR in {1..22}; do
    sbatch --export=CHR=$CHR,POPULATIONS="EUR AFR EAS",HLA_MODE="none" \
           --array=0 \
           01_preprocess.batch
done
```

---

## Population Reference

Available populations in 1KGP:
- **ALL** - All samples (recommended for pretraining)
- **EUR** - European
- **AFR** - African
- **EAS** - East Asian
- **AMR** - Admixed American
- **SAS** - South Asian

---

## Verification

### Check reference panels exist:
```bash
ls -lh ref/ | grep "vcf.gz$" | wc -l
# Should be 23 (22 chromosomes + 1 HLA region)
```

### Check split files for ALL population:
```bash
ls -lh split/1KGP_chr*_ALL_*.vcf.gz | wc -l
# Should be 69 (22 chr × 3 splits + 1 HLA × 3 splits)
```

### Check sample counts:
```bash
# Train/val/test split should be ~80/10/10
bcftools query -l split/1KGP_chr22_ALL_train.vcf.gz | wc -l
bcftools query -l split/1KGP_chr22_ALL_val.vcf.gz | wc -l
bcftools query -l split/1KGP_chr22_ALL_test.vcf.gz | wc -l
```

### Check sample ID files:
```bash
ls -lh subsets/1KGP_chr*_ALL_*.txt
# Should include:
#   1KGP_chr{1-22}_ALL_ids.txt (all samples)
#   1KGP_chr{1-22}_ALL_train_ids.txt
#   1KGP_chr{1-22}_ALL_val_ids.txt
#   1KGP_chr{1-22}_ALL_test_ids.txt
```

---

## Integration with GEUVADIS

For TIGAR baselines using GEUVADIS expression:

**GEUVADIS EUR** → uses `1KGP_chr*_ALL_*.vcf.gz` (filtered by EUR sample IDs)
**GEUVADIS GEU** → uses `1KGP_chr*_ALL_*.vcf.gz` (filtered by GEU sample IDs)

The sample ID files in `$GENOTYPE_DATA_DIR/GEUVADIS/split/EUR/` and `$GENOTYPE_DATA_DIR/GEUVADIS/split/GEU/` filter the ALL population genotypes to the correct individuals.

---

## Time Estimates

- **Reference generation**: 2-4 hours per chromosome
- **Preprocessing**: 8-20 hours per chromosome (varies by population size)
- **Total for ALL population**: ~1-2 days for all chromosomes

## Tips

1. **Use screen/tmux** for long-running submission loops:
   ```bash
   screen -S 1kgp_preprocess
   # run commands
   # Ctrl+A, D to detach
   # screen -r 1kgp_preprocess to reattach
   ```

2. **Monitor job progress**:
   ```bash
   squeue -u $USER
   tail -f qc/logs/chr22_ALL.out
   ```

3. **Resume if interrupted**: The scripts check for existing files and can resume.

4. **Check for failures**:
   ```bash
   grep -i "error\|fail" qc/logs/*.out
   ```
