# data_prep/

BiUNet-local mirror of dataset preprocessing pipelines. Each subdirectory holds the scripts that produce the contents of the corresponding `BiUNet/data/{dataset}/` tree.

Sourced 2026-05-21 from `$GENOTYPE_DATA_DIR/{dataset}/`, and held here because a cohort does not have one preprocessing pipeline: it has one per project that uses it. Masking scheme, split definition, encoding and chunk size all follow from the model being trained, so two projects reading the same VCFs need different code to get from those VCFs to their training tensors.

Keeping that code in a shared `proj/data/{dataset}/` directory encoded the opposite assumption, that a dataset has a single canonical preprocessing. Under it, one project's change to a shared script silently altered another project's inputs, which is what happened on 2025-02-07 between BiUNet and bsBSLMM. The fork removes the shared mutable state rather than trying to coordinate access to it: each project owns the path from raw data to its own tensors, and the cohort directory holds data.

## Layout

| Dataset | Scripts present |
|---|---|
| `LOS/` | `vcf2csv.py`, `masking.py`, `chunking.py`, `maskedCSV2VCF.py`, `0?_*.batch` SLURM wrappers |
| `1KGP/` | full pipeline incl. `masking_chip.py` / `masking_block.py` / `masking_phase2`, `PIPELINE.md`, `ref/` |
| `SGDP/` | `vcf2csv.py`, `chunking.py`, `mirror_HLA.py`, `0?_*.batch` |
| `HLA/` | *(empty — HLA is derived from chr6 of SGDP via `SGDP/mirror_HLA.py`; no own pipeline)* |

## Invocation convention

The Python scripts (e.g. `vcf2csv.py`, `masking.py`) use **CWD-relative paths** (`./split/...`, `./masked/...`). Run them with the working directory set to the matching `BiUNet/data/{dataset}/`:

```bash
cd "${SLURM_SUBMIT_DIR:-.}"/data/LOS
python ../../data_prep/LOS/vcf2csv.py 22 AA test
python ../../data_prep/LOS/masking.py --dataset LOS --chromosome 22 \
    --population AA --random-state 42 --missing-ratio 0.05
```

The `*.batch` SLURM wrappers expect to be submitted from `BiUNet/data/{dataset}/` so their relative paths resolve correctly.

## Rule going forward

When a new dataset is added, mirror its preprocessing scripts under `BiUNet/data_prep/{newDS}/` rather than referencing a shared `${DATA_DIR}` directory. Keeps each project self-contained.

## Divergence since the fork

The two copies have evolved independently, which is what the fork was for. Where
they differ, and why:

| File | Difference |
|---|---|
| `1KGP/masking.py` | The run copy treats chromosome 6 at full length as a target in its own right and takes the HLA region only when `--suffix _HLA` is given. The copy here still folds an unsuffixed chr6 into the HLA subset, which leaves the full chromosome unreachable. |
| `SGDP/vcf2csv.py` | The run copy takes the dataset as an argument. This copy hard-codes `dataset = 'HLA'`, inherited from the file it was branched from. |
| `SGDP/01_preprocess.batch` and the other SLURM wrappers | The run copies target chromosome 19 and are roughly twice as long, having grown QC steps this copy predates. |
| `LOS/vcf2csv.py` | This copy is the newer of the two: it was rewritten here to convert variant batches across a worker pool, with output identical to the sequential original. |

A script here is therefore not automatically the authoritative one. Before
rerunning any stage, compare the file with its counterpart beside the data and
take whichever matches the run being reproduced.
