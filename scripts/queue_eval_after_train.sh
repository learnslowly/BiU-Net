#!/bin/bash
# Queue a SLURM job that, after a training job finishes (afterany), runs
# eval_v3low_seg128.sh for that dataset (per-rate inference + 7-bin BAT_PV).
# Usage: queue_eval_after_train.sh <train_jid> <dataset> <chrom> <population> <overlap>
set -e
TJID="$1"; DS="$2"; CHR="$3"; POP="$4"; OVERLAP="$5"
TAG="${DS}_chr${CHR}_${POP}"

cat > /tmp/eval_dep_${TAG}.batch <<EOF
#!/bin/bash
#SBATCH --output=logs/eval_dep_${TAG}_%j.log
#SBATCH --time=00:30:00
#SBATCH --job-name=evald_${TAG}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --dependency=afterany:${TJID}

source configs/credentials.sh
cd "${SLURM_SUBMIT_DIR:-.}"
echo "Eval-dependency dispatcher for ${TAG} started at \$(date)"
bash scripts/eval_v3low_seg128.sh ${DS} ${CHR} ${POP} ${OVERLAP}
echo "Eval-dependency dispatcher for ${TAG} done at \$(date)"
EOF
/usr/local/bin/sbatch /tmp/eval_dep_${TAG}.batch
