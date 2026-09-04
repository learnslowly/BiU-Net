#!/bin/bash
# Submit segmentation jobs for v3_low recipe across SGDP/HLA/LOS datasets.
set -e

for cfg_tag in \
  "SGDP_chr22_ALL" \
  "HLA_chr6_ALL" \
  "LOS_chr22_ALL" \
  "LOS_chr22_AA" \
  "LOS_chr22_CA"; do
  cat > /tmp/seg_${cfg_tag}.batch <<EOF
#!/bin/bash
#SBATCH --output=logs/seg_${cfg_tag}_seg1024_%j.log
#SBATCH --time=08:00:00
#SBATCH --job-name=seg_${cfg_tag}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

source configs/credentials.sh
export OMP_NUM_THREADS=1
unset SLURM_TRES_PER_TASK

cd "${SLURM_SUBMIT_DIR:-.}"
srun python -u segmenting.py --configFile configs/seg_${cfg_tag}_seg1024.yaml --createFilteredSNPs False
EOF
  RAW=$(/usr/local/bin/sbatch /tmp/seg_${cfg_tag}.batch 2>&1)
  echo "[seg ${cfg_tag}] $RAW"
done
