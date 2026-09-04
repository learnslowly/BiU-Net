#!/bin/bash
# Submit a v3_low seg128 training job.
# Usage: submit_v3low_seg128_train.sh <dataset> <chrom> <population> <num_nodes>
set -e
DS="$1"; CHR="$2"; POP="$3"; NODES="${4:-4}"
NTASKS=$((NODES * 2))
TAG="${DS}_chr${CHR}_${POP}"
CFG="configs/train_${TAG}_seg128_v3low.yaml"
if [ ! -f "$CFG" ]; then
  echo "Missing config: $CFG"; exit 1
fi
PORT=$((12700 + RANDOM % 200))

cat > /tmp/train_${TAG}_seg128_v3low.batch <<EOF
#!/bin/bash
#SBATCH --output=logs/train_${TAG}_seg128_v3low_%j.log
#SBATCH --time=72:00:00
#SBATCH --job-name=v3l128_${TAG}
#SBATCH --partition=gpu2
#SBATCH --nodes=${NODES}
#SBATCH --ntasks=${NTASKS}
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1

source configs/credentials.sh
export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=\$(scontrol show hostname \$SLURM_NODELIST | head -n 1)
export MASTER_PORT=${PORT}
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo
export TRITON_CACHE_DIR=/tmp/\$USER/triton_cache
mkdir -p \$TRITON_CACHE_DIR

cd "${SLURM_SUBMIT_DIR:-.}"
echo "Job \$SLURM_JOB_ID (v3_low seg128 ${TAG}) started at \$(date)"
srun --export=ALL --gpu-bind=closest python -u train.py --configFile ${CFG}
echo "Job \$SLURM_JOB_ID (v3_low seg128 ${TAG}) finished at \$(date)"
EOF
/usr/local/bin/sbatch /tmp/train_${TAG}_seg128_v3low.batch
