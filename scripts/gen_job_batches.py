#!/usr/bin/env python3
"""Generate the uniform job/*.batch set for the 7 reported datasets.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = os.path.join(ROOT, "job")
os.makedirs(JOB, exist_ok=True)

# (label, overlap) -- label = <DATASET>_chr<CHR>_<POP>
DATASETS = [
    ("1KGP_chr22_ALL", 16),
    ("LOS_chr22_ALL", 16),
    ("SGDP_chr22_ALL", 16),
    ("HLA_chr6_ALL", 64),
    ("SGDP_chr19_ALL", 16),
    ("LOS_chr22_AA", 16),
    ("LOS_chr22_CA", 16),
]

HEADER = """#!/bin/bash
#SBATCH --job-name=@@JOB@@
#SBATCH --output=logs/%x_%j.log
#SBATCH --open-mode=append
@@SBATCH@@

# Project root + central credentials (REMOTE_*, PYBIN, MASTER_PORT, NCCL_*, etc.).
cd "${SLURM_SUBMIT_DIR:-.}"
source configs/credentials.sh
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=@@OMP@@
unset SLURM_TRES_PER_TASK
"""

SB_SEG = """#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10GB
#SBATCH --time=1-00:00:00"""

SB_TRAIN = """#SBATCH --partition=gpu4
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=2-23:59:00"""

SB_TRAIN_1GPU = """#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:59:00"""

SB_TEST = """#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00"""

SB_BENCH = """#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00"""

SB_REPORT = """#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00"""

DDP = ('export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)\n'
       '# MASTER_PORT, NCCL_SOCKET_IFNAME come from credentials.sh; CUDA_VISIBLE_DEVICES is pinned in the .py.\n')


def header(job, sbatch, omp):
    return (HEADER.replace("@@JOB@@", job).replace("@@SBATCH@@", sbatch).replace("@@OMP@@", str(omp)))


def write(name, text):
    with open(os.path.join(JOB, name), "w") as f:
        f.write(text)


def seg(label, conf, job=None, sbatch=SB_SEG):
    job = job or f"seg_{label}"
    return header(job, sbatch, 1) + (
        f'\n# Segment into seg HDF5. --createFilteredSNPs False: training does not need the\n'
        f'# per-MAF-bin filtered files (they OOM at scale).\n'
        f'echo "Start: $(date)"\n'
        f'srun "$PYBIN" -u segmenting.py --configFile {conf} --createFilteredSNPs False\n'
        f'echo "End:   $(date)"\n')


def train(label, conf, job=None, sbatch=SB_TRAIN):
    job = job or f"train_{label}"
    return header(job, sbatch, 1) + "\n" + DDP + (
        f'\nCONF={conf}\n'
        f'echo "Start: $(date)   Nodes: $SLURM_JOB_NODELIST   World: $SLURM_NTASKS"\n'
        f'srun --gpu-bind=closest "$PYBIN" -u train.py --configFile "$CONF"\n'
        f'echo "End:   $(date)"\n')


def test(label, conf, job=None):
    job = job or f"test_{label}"
    return header(job, SB_TEST, 1) + "\n" + DDP + (
        f'\nCONF={conf}\n'
        f'echo "Start: $(date)   World: $SLURM_NTASKS"\n'
        f'srun --gpu-bind=closest "$PYBIN" -u test.py --configFile "$CONF"\n'
        f'echo "End:   $(date)"\n')


def benchmark(label, conf, method=None, job=None):
    tag = "benchmark" if method is None else f"benchmark_{method}"
    job = job or f"{tag}_{label}"
    impflag = f" --impMethod {method}" if method else ""
    return header(job, SB_BENCH, 8) + (
        f'\nCFG={conf}\n'
        f'# Missing rates and random states come from the config; one pass covers them all.\n'
        f'"$PYBIN" -u benchmark.py --configFile "$CFG"{impflag}\n'
        f'echo "{tag} {label} done -> analysis/"\n')


def report(label, conf, scda_conf=None, job=None):
    """CPU job: pivot the per-bin benchmark CSVs into a paste-ready comparison table.
    Run AFTER the benchmark_* jobs for this dataset have written to analysis/."""
    job = job or f"report_{label}"
    scdaflag = f" --reportScda {scda_conf}" if scda_conf else ""
    methods = "Beagle + BiU-Net" + (" + SCDA" if scda_conf else "")
    return header(job, SB_REPORT, 2) + (
        f'\n# Aggregate per-bin CSVs ({methods}) over random states into a\n'
        f'# tab-separated comparison table (metric x method by MAF bin, one block per rate).\n'
        f'mkdir -p analysis\n'
        f'"$PYBIN" -u benchmark.py --report --configFile {conf}{scdaflag} | tee "analysis/report_{label}.tsv"\n'
        f'echo "report {label} -> analysis/report_{label}.tsv"\n')


# ---- per-dataset BiU-Net pipeline + Beagle benchmark ----
for label, ov in DATASETS:
    seg_conf = f"configs/{label}_seg128_overlap{ov}.yaml"
    train_conf = f"configs/train_{label}_seg128.yaml"
    eval_conf = f"configs/test_seg128_{label}.yaml"
    scda_eval = f"configs/test_scda_{label}.yaml"
    write(f"seg_{label}.batch", seg(label, seg_conf))
    write(f"train_{label}.batch", train(label, train_conf))
    write(f"test_{label}.batch", test(label, eval_conf))
    write(f"benchmark_{label}.batch", benchmark(label, eval_conf, method=None))
    write(f"benchmark_beagle_{label}.batch", benchmark(label, eval_conf, method="beagle"))
    write(f"report_{label}.batch", report(label, eval_conf, scda_conf=scda_eval))

# ---- SCDA baseline: full-length input, so its own segmentation and configs ----
for label, _ in DATASETS:
    write(f"scda_seg_{label}.batch",
          seg(label, f"configs/{label}_seg-1_overlap0.yaml", job=f"scda_seg_{label}"))
    write(f"scda_train_{label}.batch",
          train(label, f"configs/train_scda_{label}.yaml",
                job=f"scda_train_{label}", sbatch=SB_TRAIN_1GPU))  # full-length SCDA: 1 GPU
    write(f"scda_test_{label}.batch",
          test(label, f"configs/test_scda_{label}.yaml", job=f"scda_test_{label}"))
    write(f"benchmark_scda_{label}.batch",
          benchmark(label, f"configs/test_scda_{label}.yaml", method=None,
                    job=f"benchmark_scda_{label}"))

print("generated", len(os.listdir(JOB)), "batches in job/")
