#!/bin/bash
# One chromosome's array-to-WGS evaluation, run as a job step inside a larger
# allocation. EVAL_TAG picks which model is being scored on the array test set:
#   chipft      the model fine-tuned under array-like masking
#   chipscratch the model trained from random initialisation under that masking
#   chrft_chip  the existing per-chromosome model, trained under sporadic masking
#
# Both tasks of the step take part in the distributed imputation; the scoring
# that follows is single-process, so only rank 0 runs it.
set -uo pipefail
source configs/credentials.sh

: "${EVAL_DS:?EVAL_DS not set}"
: "${EVAL_C:?EVAL_C not set}"
: "${EVAL_PORT:?EVAL_PORT not set}"
TAG="${EVAL_TAG:-chipft}"

STEP_NODES="${SLURM_STEP_NODELIST:-$SLURM_NODELIST}"
export MASTER_ADDR=$(scontrol show hostname "$STEP_NODES" | head -n 1)
export MASTER_PORT="$EVAL_PORT"
export NCCL_SOCKET_IFNAME=^lo
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

# The tag is the middle of the config name, so chipft, chipscratch and
# chrft_chip all resolve without a case for each.
CFG="configs/test_${EVAL_DS}_${TAG}_${EVAL_C}.yaml"
[ -s "$CFG" ] || { echo "ERROR: missing $CFG"; exit 1; }

"$PYBIN" -u test.py --configFile "$CFG"
RC=$?
[ $RC -ne 0 ] && { echo "test.py failed for $EVAL_DS chr$EVAL_C tag=$TAG (rc=$RC)"; exit $RC; }

[ "${SLURM_PROCID:-0}" != "0" ] && exit 0

# benchmark.py reads the rates and seeds from the config, so one call covers the
# whole cell set for this chromosome.
"$PYBIN" -u benchmark.py --configFile "$CFG"
RC=$?
echo "done $EVAL_DS chr$EVAL_C tag=$TAG (rc=$RC)"
exit $RC
