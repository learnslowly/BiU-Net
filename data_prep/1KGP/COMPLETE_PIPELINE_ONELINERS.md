# 1KGP Complete Pipeline - One-Liners for ALL Population

Complete commands for preprocessing 1KGP genotypes for GenoBERT_LoRA (ALL population, all chromosomes).

---

## Step 0: Generate Reference Panels

### All chromosomes 1-22
```bash
cd $GENOTYPE_DATA_DIR/1KGP
mkdir -p ref/logs

for CHR in {1..22}; do
    sbatch --export=CHR=$CHR \
           --array=0 \
           --output=ref/logs/chr${CHR}_ref.out \
           00_gen_ref_array.batch
done
```

**Wait for all reference generation jobs to complete before proceeding.**

Verify:
```bash
ls -1 ref/1KGP_chr*_ALL_ref.vcf.gz | wc -l
# Should show 22
```

---

## Step 1: Preprocess Genotypes (QC, Phasing, Population Splitting)

### All chromosomes 1-22 (ALL population)
```bash
mkdir -p qc/logs

for CHR in {1..22}; do
    sbatch --export=CHR=$CHR,POPULATIONS="ALL",HLA_MODE="none" \
           --array=0 \
           --output=qc/logs/chr${CHR}_ALL.out \
           01_preprocess.batch
done
```

**Note:** The `--output=qc/logs/chr${CHR}_ALL.out` is **critical** - it overrides the batch script's `%a` placeholder to use the actual chromosome number, preventing all jobs from writing to the same log file.

**Expected outputs:**
- split/1KGP_chr{1-22}_ALL_train.vcf.gz
- split/1KGP_chr{1-22}_ALL_val.vcf.gz
- split/1KGP_chr{1-22}_ALL_test.vcf.gz

Total: **66 VCF files** (22 chr × 3 splits)

Verify:
```bash
ls -1 split/1KGP_chr*_ALL_*.vcf.gz | grep -v HLA | wc -l
# Should show 66
```

---

## Step 2: Convert VCF to CSV

### All chromosomes 1-22 (ALL population)
```bash
mkdir -p split

for CHR in {1..22}; do
    sbatch --export=CHR=$CHR,POPULATIONS="ALL",HLA_MODE="none" \
           --array=0-2 \
           --output=split/vcf2csv_chr${CHR}_%a.out \
           02_vcf2csv.batch
done
```

**Note:** The `%a` in output path is kept here because array=0-2 processes 3 splits, so we want separate logs for each split.

**Note:** Array 0-2 processes 3 splits (train, val, test) for each chromosome

**Expected outputs:**
- split/1KGP_chr{1-22}_ALL_train.csv.gz
- split/1KGP_chr{1-22}_ALL_val.csv.gz
- split/1KGP_chr{1-22}_ALL_test.csv.gz

Total: **66 CSV files** (22 chr × 3 splits)

Verify:
```bash
ls -1 split/1KGP_chr*_ALL_*.csv.gz | grep -v HLA | wc -l
# Should show 66
```

---

## Step 3: Add Masking to Test Data

### All chromosomes 1-22 (ALL population)
```bash
mkdir -p masked

for CHR in {1..22}; do
    sbatch --export=CHR=$CHR,HLA_MODE="none" \
           --array=0 \
           --output=masked/masking_chr${CHR}.out \
           03_masking.batch
done
```

**Note:** The `--output` override prevents all chromosomes from writing to masked/masking_0.out.

**What this does:**
- Adds 4 missing levels (5%, 15%, 25%, 50%) to test data
- Creates 3 random states (0, 42, 1024) for each level
- Processes ALL population only
- Total: 12 masked files per chromosome (4 levels × 3 random states)

**Expected outputs (per chromosome):**
- masked/{0,42,1024}/{5%,15%,25%,50%}/1KGP_chr{CHR}_ALL_missing{X}%_masked.csv.gz

Total: **264 masked CSV files** (22 chr × 12 masked versions)

Verify:
```bash
find masked -name "1KGP_chr*_ALL_missing*_masked.csv.gz" | grep -v HLA | wc -l
# Should show 264
```

---

## Step 4: Convert Masked CSV back to VCF

### All chromosomes 1-22 (ALL population)
```bash
for CHR in {1..22}; do
    sbatch --export=CHR=$CHR,POPULATIONS="ALL",HLA_MODE="none" \
           --array=0-11 \
           --output=masked/mask_vcf_chr${CHR}_%a.out \
           04_maskedCSV2VCF.batch
done
```

**Note:** Array 0-11 processes 12 combinations (3 random states × 4 missing levels). The `chr${CHR}` ensures each chromosome has separate logs.

**Expected outputs (per chromosome):**
- masked/{0,42,1024}/{5%,15%,25%,50%}/1KGP_chr{CHR}_ALL_missing{X}%_masked.vcf.gz
- masked/{0,42,1024}/{5%,15%,25%,50%}/1KGP_chr{CHR}_ALL_missing{X}%_masked.vcf.gz.tbi
- masked/{0,42,1024}/{5%,15%,25%,50%}/1KGP_chr{CHR}_ALL_missing{X}%_missing_with_samples.txt

Total: **264 masked VCF files + 264 indices + 264 missing info files**

Verify:
```bash
find masked -name "1KGP_chr*_ALL_missing*_masked.vcf.gz" | grep -v HLA | wc -l
# Should show 264
```

---

## Step 5: Chunk CSV Files for Training

### All chromosomes 1-22 (ALL population)
```bash
mkdir -p chunked

for CHR in {1..22}; do
    sbatch --export=CHR=$CHR,HLA_MODE="none" \
           --array=0 \
           --output=chunked/chunking_chr${CHR}.out \
           05_chunking.batch
done
```

**Note:** The `--output` override creates separate log files per chromosome (chunking_chr1.out, chunking_chr2.out, etc.).

**What this does:**
- Splits train/val CSV files into chunks of 10,000 samples
- Creates chunked/1KGP_chr{CHR}_ALL_{train,val}_chunk{NNN}.csv.gz
- If file has < 10,000 samples, renames to chunk000

**Expected outputs:**
- chunked/1KGP_chr{1-22}_ALL_train_chunk*.csv.gz
- chunked/1KGP_chr{1-22}_ALL_val_chunk*.csv.gz

Total: Variable (depends on sample counts per chromosome)

Verify:
```bash
ls -1 chunked/1KGP_chr*_ALL_*_chunk*.csv.gz | grep -v HLA | wc -l
```

---

## (Optional) Chr6 HLA Region

If you also need chromosome 6 HLA region specifically:

```bash
# Step 0: Generate HLA reference
sbatch --export=CHR=6,HLA_ONLY=true \
       --array=0 \
       --output=ref/logs/chr6_HLA_ref.out \
       00_gen_ref_array.batch

# Step 1: Preprocess HLA region
sbatch --export=CHR=6,POPULATIONS="ALL",HLA_MODE="hla_only" \
       --array=0 \
       --output=qc/logs/chr6_HLA_ALL.out \
       01_preprocess.batch

# Step 2: VCF to CSV
sbatch --export=CHR=6,POPULATIONS="ALL",HLA_MODE="hla_only" \
       --array=0-2 \
       --output=split/vcf2csv_chr6_HLA_%a.out \
       02_vcf2csv.batch

# Step 3: Masking
sbatch --export=CHR=6,HLA_MODE="hla_only" \
       --array=0 \
       --output=masked/masking_chr6_HLA.out \
       03_masking.batch

# Step 4: Masked CSV to VCF
sbatch --export=CHR=6,POPULATIONS="ALL",HLA_MODE="hla_only" \
       --array=0-11 \
       --output=masked/mask_vcf_chr6_HLA_%a.out \
       04_maskedCSV2VCF.batch

# Step 5: Chunking
sbatch --export=CHR=6,HLA_MODE="hla_only" \
       --array=0 \
       --output=chunked/chunking_chr6_HLA.out \
       05_chunking.batch
```

---

## Single Chromosome Example

To process just one chromosome (e.g., chr22):

```bash
CHR=22
POPS="ALL"

# Step 0: Reference
sbatch --export=CHR=$CHR --array=0 --output=ref/logs/chr${CHR}_ref.out 00_gen_ref_array.batch

# Step 1: Preprocess
sbatch --export=CHR=$CHR,POPULATIONS="$POPS",HLA_MODE="none" --array=0 --output=qc/logs/chr${CHR}_ALL.out 01_preprocess.batch

# Step 2: VCF to CSV
sbatch --export=CHR=$CHR,POPULATIONS="$POPS",HLA_MODE="none" --array=0-2 --output=split/vcf2csv_chr${CHR}_%a.out 02_vcf2csv.batch

# Step 3: Masking
sbatch --export=CHR=$CHR,HLA_MODE="none" --array=0 --output=masked/masking_chr${CHR}.out 03_masking.batch

# Step 4: Masked CSV to VCF
sbatch --export=CHR=$CHR,POPULATIONS="$POPS",HLA_MODE="none" --array=0-11 --output=masked/mask_vcf_chr${CHR}_%a.out 04_maskedCSV2VCF.batch

# Step 5: Chunking
sbatch --export=CHR=$CHR,HLA_MODE="none" --array=0 --output=chunked/chunking_chr${CHR}.out 05_chunking.batch
```

---

## Pipeline Summary

**Total outputs for ALL population (chr1-22):**
1. Reference VCFs: 22 files
2. Split VCFs: 66 files (train/val/test)
3. Split CSVs: 66 files (train/val/test)
4. Masked CSVs: 264 files (4 levels × 3 random states × 22 chr)
5. Masked VCFs: 264 files + indices + missing info
6. Chunked CSVs: Variable (depends on sample counts)

**Time estimates:**
- Step 0 (Reference): 2-4 hours per chromosome
- Step 1 (Preprocess): 8-20 hours per chromosome
- Step 2 (VCF to CSV): 1-3 hours per chromosome
- Step 3 (Masking): 2-4 hours per chromosome
- Step 4 (CSV to VCF): 3-6 hours per chromosome
- Step 5 (Chunking): 4-12 hours per chromosome

**Total pipeline time: ~2-3 days for all 22 chromosomes**

---

## Tips

1. **Use screen/tmux** for long-running submission loops:
   ```bash
   screen -S 1kgp_pipeline
   # run commands
   # Ctrl+A, D to detach
   # screen -r 1kgp_pipeline to reattach
   ```

2. **Monitor job progress**:
   ```bash
   squeue -u $USER
   watch -n 60 'squeue -u $USER | wc -l'
   ```

3. **Check for failures**:
   ```bash
   grep -i "error\|fail" qc/logs/*.out
   grep -i "error\|fail" split/*.out
   grep -i "error\|fail" masked/*.out
   grep -i "error\|fail" chunked/*.out
   ```

4. **Verify completion** at each step before proceeding to the next

5. **Disk space**: Ensure you have sufficient space (~500GB for ALL population, 22 chromosomes)
