#!/bin/bash
# Per-rate inference + phased per-variant BAT benchmark for revision models.
# Usage: eval_v3_v4_per_rate.sh <runId> <segLen> <overlap> [<extra_yaml_lines>]
# Example: eval_v3_v4_per_rate.sh v3_low 1024 128 ""
#          eval_v3_v4_per_rate.sh v4_cMgap 1024 128 "bioEncoding: 'cM_gap'\nbioChannels: 1\nbioFile: '$GENOTYPE_DATA_DIR/bio/chr22_bio.npz'"

set -e
RUNID="$1"
SEGLEN="$2"
OVERLAP="$3"
EXTRA="$4"
if [ -z "$RUNID" ] || [ -z "$SEGLEN" ] || [ -z "$OVERLAP" ]; then
  echo "Usage: $0 <runId> <segLen> <overlap> [extra_yaml_lines]"
  exit 2
fi

cd "${SLURM_SUBMIT_DIR:-.}"
source configs/credentials.sh
RUN_NAME="${RUNID}_1KGP_chr22_ALL_seg${SEGLEN}_overlap${OVERLAP}"
BEST_OUTPUT=$(python -u scripts/find_best_and_copy.py --run "$RUN_NAME" 2>&1)
echo "$BEST_OUTPUT"
BEST_EPOCH=$(echo "$BEST_OUTPUT" | grep "BEST_EPOCH=" | tail -n 1 | sed 's/BEST_EPOCH=//')
if [ -z "$BEST_EPOCH" ]; then
  echo "ERROR: no best epoch found for $RUN_NAME"
  exit 1
fi
echo "Using BEST_EPOCH=$BEST_EPOCH for $RUNID"

RATES=(5 15 25 50 95)
for RATE in "${RATES[@]}"; do
  R_FRAC=$(echo "$RATE/100" | bc -l | awk '{printf "%.2f", $1}')
  CONFIG="configs/test_${RUNID}_${RATE}.yaml"
  cat > $CONFIG <<EOF
runId: ${RUNID}
dataset: 1KGP
chromosome: 22
chromosomes: [22]
segLen: ${SEGLEN}
overlap: ${OVERLAP}
population: ALL
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
bins: [0.0, 0.005, 0.01, 0.05, 0.40, 0.5]
missing: [$R_FRAC]
testRandStates: [42]
EOF
  # Append extra YAML lines (e.g., bio encoding parameters)
  if [ -n "$EXTRA" ]; then
    printf "%b\n" "$EXTRA" >> $CONFIG
  fi
done
echo "Wrote ${RUNID} inference configs for rates: ${RATES[@]}"

INF_JIDS=()
for RATE in "${RATES[@]}"; do
  cat > /tmp/test_${RUNID}_${RATE}.batch <<EOF
#!/bin/bash
#SBATCH --output=logs/test_${RUNID}_${RATE}_%j.log
#SBATCH --time=02:00:00
#SBATCH --job-name=${RUNID}_${RATE}
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
export MASTER_PORT=$((13000 + RATE + RANDOM % 100))
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo
export TRITON_CACHE_DIR=/tmp/\$USER/triton_cache
mkdir -p \$TRITON_CACHE_DIR

cd "${SLURM_SUBMIT_DIR:-.}"
srun --export=ALL --gpu-bind=closest python -u test.py --configFile configs/test_${RUNID}_${RATE}.yaml
EOF
  RAW=$(/usr/local/bin/sbatch /tmp/test_${RUNID}_${RATE}.batch 2>&1)
  echo "[debug] sbatch (${RUNID}_${RATE}): $RAW"
  JID=$(echo "$RAW" | awk '/Submitted batch job/ {print $NF}')
  INF_JIDS+=($JID)
done
echo "${RUNID} inference jobs: ${INF_JIDS[@]}"

DEP=$(IFS=:; echo "${INF_JIDS[*]}")
cat > /tmp/benchmark_${RUNID}.batch <<EOF
#!/bin/bash
#SBATCH --output=logs/benchmark_${RUNID}_%j.log
#SBATCH --time=02:00:00
#SBATCH --job-name=${RUNID}_bnch
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --dependency=afterany:${DEP}

source configs/credentials.sh
cd "${SLURM_SUBMIT_DIR:-.}"
echo "${RUNID} benchmark started at \$(date)"

for RATE in 5 15 25 50 95; do
  echo "=== rate \$RATE ==="
  python -u benchmark.py --configFile configs/test_${RUNID}_\${RATE}.yaml --randState 42 --missingLevelIdx 0 2>&1 | grep -E "Overall|MAF"
done

echo "${RUNID} benchmark done at \$(date)"
EOF
/usr/local/bin/sbatch /tmp/benchmark_${RUNID}.batch
echo "${RUNID} benchmark queued with dep=$DEP"
