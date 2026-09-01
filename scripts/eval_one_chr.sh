#!/bin/bash
# One chromosome's evaluation, run as a job step inside a larger allocation.
#
# Both tasks of the step take part in the distributed imputation; the scoring
# that follows is single-process, so only rank 0 runs it while rank 1 exits.
# MASTER_ADDR is resolved from the step's own nodelist, not the job's, so
# concurrent steps on different nodes do not collide.
set -uo pipefail
source configs/credentials.sh

: "${EVAL_DS:?EVAL_DS not set}"
: "${EVAL_C:?EVAL_C not set}"
: "${EVAL_PORT:?EVAL_PORT not set}"

# SLURM_NODELIST names the whole job allocation, not this step, so every step
# would rendezvous on the job's first node instead of its own. Only
# SLURM_STEP_NODELIST is step-scoped; fall back for a plain srun job.
STEP_NODES="${SLURM_STEP_NODELIST:-$SLURM_NODELIST}"
export MASTER_ADDR=$(scontrol show hostname "$STEP_NODES" | head -n 1)
export MASTER_PORT="$EVAL_PORT"
export NCCL_SOCKET_IFNAME=^lo
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

CFG="configs/test_${EVAL_DS}_chrft_${EVAL_C}.yaml"
[ -s "$CFG" ] || { echo "ERROR: missing $CFG"; exit 1; }

"$PYBIN" -u test.py --configFile "$CFG"
RC=$?
[ $RC -ne 0 ] && { echo "test.py failed for $EVAL_DS chr$EVAL_C (rc=$RC)"; exit $RC; }

[ "${SLURM_PROCID:-0}" != "0" ] && exit 0

FAIL=0
for RS in 0 42 1024; do
  for IDX in 0 1 2; do
    "$PYBIN" -u benchmark.py --configFile "$CFG" --randState "$RS" --missingLevelIdx "$IDX" \
      || { echo "FAILED $EVAL_DS chr$EVAL_C rand=$RS idx=$IDX"; FAIL=1; }
  done
done
echo "done $EVAL_DS chr$EVAL_C (fail=$FAIL)"
exit $FAIL
