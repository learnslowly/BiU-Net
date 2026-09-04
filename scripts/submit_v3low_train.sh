#!/bin/bash
# Submit a v3_low recipe training job for one dataset.
# Usage: submit_v3low_train.sh <dataset_tag> <num_nodes>
# e.g.: submit_v3low_train.sh SGDP_chr22_ALL 4

set -e
TAG="$1"
NODES="${2:-4}"
NTASKS=$((NODES * 2))
if [ -z "$TAG" ]; then
  echo "Usage: $0 <dataset_tag> [num_nodes]"; exit 2
fi
CFG="configs/train_${TAG}_seg1024_v3_low.yaml"
if [ ! -f "$CFG" ]; then
  echo "Missing config: $CFG"; exit 1
fi
PORT=$((12500 + RANDOM % 200))

cat > /tmp/train_${TAG}_v3_low.batch <<EOF
#!/bin/bash
#SBATCH --output=logs/train_${TAG}_v3_low_%j.log
#SBATCH --time=20:00:00
#SBATCH --job-name=v3l_${TAG}
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
echo "Job \$SLURM_JOB_ID (v3_low ${TAG}) started at \$(date)"
srun --export=ALL --gpu-bind=closest python -u train.py --configFile ${CFG}
echo "Job \$SLURM_JOB_ID (v3_low ${TAG}) finished at \$(date)"
EOF
/usr/local/bin/sbatch /tmp/train_${TAG}_v3_low.batch
