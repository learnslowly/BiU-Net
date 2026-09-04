# 1KGP Paired Processing Steps

Complete paired commands for reference generation → preprocessing.

---

## Step 1: Generate Reference Panels (chr1-22)

```bash
cd $GENOTYPE_DATA_DIR/1KGP

# Create log directory
mkdir -p ref/logs

# Generate references for all chromosomes 1-22
for CHR in {1..22}; do
    sbatch --export=CHR=$CHR \
           --array=0 \
           --output=ref/logs/chr${CHR}_ref.out \
           00_gen_ref_array.batch
done
```

**Wait for all jobs to complete** before proceeding to Step 2.

Check completion:
```bash
# Should show 22 files
ls -1 ref/1KGP_chr*_ALL_ref.vcf.gz | wc -l

# Check if all have indexes
ls -1 ref/1KGP_chr*_ALL_ref.vcf.gz.tbi | wc -l
```

---

## Step 2: Preprocess Genotypes - ALL Population (chr1-22)

```bash
# Create log directory
mkdir -p qc/logs

# Preprocess all chromosomes 1-22 for ALL population
for CHR in {1..22}; do
    sbatch --export=CHR=$CHR,POPULATIONS="ALL",HLA_MODE="none" \
           --array=0 \
           --output=qc/logs/chr${CHR}_ALL.out \
           01_preprocess.batch
done
```

**Expected outputs:**
```
split/1KGP_chr1_ALL_train.vcf.gz
split/1KGP_chr1_ALL_val.vcf.gz
split/1KGP_chr1_ALL_test.vcf.gz
...
split/1KGP_chr22_ALL_train.vcf.gz
split/1KGP_chr22_ALL_val.vcf.gz
split/1KGP_chr22_ALL_test.vcf.gz
```

Total: **66 VCF files** (22 chromosomes × 3 splits)

Check completion:
```bash
ls -1 split/1KGP_chr*_ALL_*.vcf.gz | grep -v HLA | wc -l
# Should show 66
```

---

## Step 3 (Optional): Chr6 HLA Region

### 3a. Generate HLA reference
```bash
sbatch --export=CHR=6,HLA_ONLY=true \
       --array=0 \
       --output=ref/logs/chr6_HLA_ref.out \
       00_gen_ref_array.batch
```

Wait for completion:
```bash
ls -lh ref/HLA_chr6_ALL_ref.vcf.gz
```

### 3b. Preprocess HLA region
```bash
sbatch --export=CHR=6,POPULATIONS="ALL",HLA_MODE="hla_only" \
       --array=0 \
       --output=qc/logs/chr6_HLA_ALL.out \
       01_preprocess.batch
```

**Expected outputs:**
```
split/1KGP_chr6_ALL_train_HLA.vcf.gz
split/1KGP_chr6_ALL_val_HLA.vcf.gz
split/1KGP_chr6_ALL_test_HLA.vcf.gz
```

Total: **3 additional VCF files**

---

## Complete Pipeline Summary

**Total reference files:** 23 (chr1-22 + chr6_HLA)
**Total split files (ALL population):** 69 (66 regular + 3 HLA)

**Timeline:**
1. Submit all reference generation jobs (~2-4 hours per chr)
2. Wait for references to complete
3. Submit all preprocessing jobs (~8-20 hours per chr)
4. (Optional) Submit HLA-specific jobs

**Verification:**
```bash
# References (should be 22 or 23 with HLA)
ls -1 ref/*_ref.vcf.gz | wc -l

# Splits (should be 66 or 69 with HLA)
ls -1 split/1KGP_chr*_ALL_*.vcf.gz | wc -l

# Sample ID files (should be 88 or 92 with HLA)
# 22 chr × 4 files/chr = 88, +4 for HLA = 92
ls -1 subsets/1KGP_chr*_ALL_*.txt | wc -l
```
