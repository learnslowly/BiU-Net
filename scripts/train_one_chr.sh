#!/bin/bash
# One chromosome's fine-tuning, run as a job step inside a larger allocation.
#
# MASTER_ADDR comes from the step's own nodelist so that concurrent steps on
# different nodes do not share a rendezvous. The epoch_0 checkpoint is prebaked
# by the driver before dispatch, so this script only trains.
set -uo pipefail
source configs/credentials.sh

: "${FT_DS:?FT_DS not set}"
: "${FT_C:?FT_C not set}"
: "${FT_PORT:?FT_PORT not set}"

# SLURM_NODELIST names the whole job allocation, not this step, so every step
# would rendezvous on the job's first node instead of its own. Only
# SLURM_STEP_NODELIST is step-scoped; fall back for a plain srun job.
STEP_NODES="${SLURM_STEP_NODELIST:-$SLURM_NODELIST}"
export MASTER_ADDR=$(scontrol show hostname "$STEP_NODES" | head -n 1)
export MASTER_PORT="$FT_PORT"
export NCCL_SOCKET_IFNAME=^lo
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR=/tmp/$USER/triton_${SLURM_JOB_ID}_${FT_DS}_${FT_C}
export TORCHINDUCTOR_CACHE_DIR=/tmp/$USER/inductor_${SLURM_JOB_ID}_${FT_DS}_${FT_C}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# FT_TAG selects the campaign: chrft = the sporadic per-chromosome runs,
# chipft = the same fine-tuning under array-like column masking.
CFG="configs/train_${FT_DS}_${FT_TAG:-chrft}_${FT_C}.yaml"
[ -s "$CFG" ] || { echo "ERROR: missing $CFG"; exit 1; }

"$PYBIN" -u train.py --configFile "$CFG"
RC=$?
echo "train step done: $FT_DS chr$FT_C rc=$RC"
exit $RC
