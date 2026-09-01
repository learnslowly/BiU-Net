#!/usr/bin/env bash
# Cluster-side job submitter. The batches in job/ deliberately omit the private
# SLURM directives (--account, --mail-user); this wrapper injects them from
# configs/credentials.sh. Run from the project root on the cluster:
#
#     ./submit.sh job/train_SGDP_chr19_ALL.batch [extra sbatch args]
#
set -euo pipefail
cd "$(dirname "$0")"
source configs/credentials.sh
exec sbatch --account="$SLURM_ACCOUNT" --mail-user="$NOTIFY_EMAIL" "$@"
