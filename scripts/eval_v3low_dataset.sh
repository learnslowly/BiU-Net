#!/bin/bash
# Per-rate inference + 7-bin BAT_PV benchmark for v3_low-recipe models across datasets.
# Usage: eval_v3low_dataset.sh <dataset> <chrom> <population> <segLen> <overlap>
# Example: eval_v3low_dataset.sh SGDP 22 ALL 1024 128
set -e
DS="$1"; CHR="$2"; POP="$3"; SEGLEN="$4"; OVERLAP="$5"
if [ -z "$DS" ] || [ -z "$CHR" ] || [ -z "$POP" ] || [ -z "$SEGLEN" ] || [ -z "$OVERLAP" ]; then
  echo "Usage: $0 <dataset> <chrom> <population> <segLen> <overlap>"; exit 2
fi

cd "${SLURM_SUBMIT_DIR:-.}"
source configs/credentials.sh
RUNID="v3_low"
TAG="${DS}_chr${CHR}_${POP}"
RUN_NAME="${RUNID}_${TAG}_seg${SEGLEN}_overlap${OVERLAP}"
BEST_OUTPUT=$(python -u scripts/find_best_and_copy.py --run "$RUN_NAME" 2>&1)
echo "$BEST_OUTPUT"
BEST_EPOCH=$(echo "$BEST_OUTPUT" | grep "BEST_EPOCH=" | tail -n 1 | sed 's/BEST_EPOCH=//')
if [ -z "$BEST_EPOCH" ]; then
  echo "ERROR: no best epoch found for $RUN_NAME"; exit 1
fi
echo "Using BEST_EPOCH=$BEST_EPOCH for $TAG"

RATES=(5 15 25)
for RATE in "${RATES[@]}"; do
  R_FRAC=$(echo "$RATE/100" | bc -l | awk '{printf "%.2f", $1}')
  CONFIG="configs/test_v3_low_${TAG}_${RATE}.yaml"
  cat > $CONFIG <<EOF
runId: ${RUNID}
dataset: ${DS}
chromosome: ${CHR}
chromosomes: [${CHR}]
segLen: ${SEGLEN}
overlap: ${OVERLAP}
population: ${POP}
model: unet
bioAware: True
depth: 6
nchannels: 48
kernelSize: 7
stride: 1
dropoutRate: 0.05
useGroupNorm: False
batchSize: 256
batchSizeTest: 8
learningRate: 0.0005
loss: hybridFocalLoss
dosageLossLambda: 0.1
focalAlpha: 0.25
focalGamma: 2.0
sampling: normal
useWandB: False
finetuning: False
totalEpochs: 700
warmupEpochs: 10
scheduler: cosineAnn
adamwBeta1: 0.9
adamwBeta2: 0.99
adamwEps: 0.00000001
adamwWeightDecay: 0.01
seed: 42
missingRatio: 0.95
dynamicRatio: True
bertStrategy: False
benchmarkAll: True
perVariantR2: True
epoch: $BEST_EPOCH
bins: [0.001, 0.005, 0.010, 0.100, 0.200, 0.300, 0.400, 0.500]
missing: [$R_FRAC]
testRandStates: [42]
EOF
done
echo "Wrote v3_low ${TAG} test configs for rates: ${RATES[@]}"

INF_JIDS=()
for RATE in "${RATES[@]}"; do
  cat > /tmp/test_v3low_${TAG}_${RATE}.batch <<EOF
#!/bin/bash
#SBATCH --output=logs/test_v3low_${TAG}_${RATE}_%j.log
#SBATCH --time=02:00:00
#SBATCH --job-name=v3l_${TAG}_${RATE}
#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:2

source configs/credentials.sh
export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=\$(scontrol show hostname \$SLURM_NODELIST | head -n 1)
export MASTER_PORT=$((14000 + RATE + RANDOM % 200))
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo
export TRITON_CACHE_DIR=/tmp/\$USER/triton_cache
mkdir -p \$TRITON_CACHE_DIR

cd "${SLURM_SUBMIT_DIR:-.}"
srun --export=ALL --gpu-bind=closest python -u test.py --configFile configs/test_v3_low_${TAG}_${RATE}.yaml
EOF
  RAW=$(/usr/local/bin/sbatch /tmp/test_v3low_${TAG}_${RATE}.batch 2>&1)
  echo "[v3l_${TAG}_${RATE}] $RAW"
  JID=$(echo "$RAW" | awk '/Submitted batch job/ {print $NF}')
  INF_JIDS+=($JID)
done
echo "Inference jobs: ${INF_JIDS[@]}"

DEP=$(IFS=:; echo "${INF_JIDS[*]}")
cat > /tmp/benchmark_v3low_${TAG}.batch <<EOF
#!/bin/bash
#SBATCH --output=logs/benchmark_v3low_${TAG}_%j.log
#SBATCH --time=02:00:00
#SBATCH --job-name=bnc_${TAG}
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --dependency=afterany:${DEP}

source configs/credentials.sh
cd "${SLURM_SUBMIT_DIR:-.}"
echo "v3_low ${TAG} benchmark started at \$(date)"

for RATE in 5 15 25; do
  echo "=== BiUNet rate \$RATE ==="
  python -u benchmark.py --configFile configs/test_v3_low_${TAG}_\${RATE}.yaml --randState 42 --missingLevelIdx 0 2>&1 | grep -E "Overall|MAF_bin"
  echo "=== Beagle rate \$RATE ==="
  python -u benchmark.py --configFile configs/test_v3_low_${TAG}_\${RATE}.yaml --randState 42 --missingLevelIdx 0 --impMethod beagle 2>&1 | grep -E "Overall|MAF_bin"
done

echo "v3_low ${TAG} benchmark done at \$(date)"
EOF
/usr/local/bin/sbatch /tmp/benchmark_v3low_${TAG}.batch
echo "${TAG} benchmark queued with dep=$DEP"
