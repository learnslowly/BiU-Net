# BiU-Net

A bio-aware 1D U-Net for SNP genotype imputation. BiU-Net reconstructs masked
genotypes from a segmented haplotype context, optionally conditioned on a
per-variant biological prior, and is benchmarked against
[Beagle](https://faculty.washington.edu/browning/beagle/beagle.html) and an SCDA
autoencoder baseline.

> Reference implementation for *BiU-Net: a Biological-informed U-Net for Genotype Imputation* ([preprint](https://doi.org/10.21203/rs.3.rs-6797863/v1)).

## Installation

Python 3.10, PyTorch 2.5 (CUDA 12.1).

```bash
conda create -n biunet python=3.10 -y && conda activate biunet
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Configuration

Cluster and private settings live in one git-ignored file:

```bash
cp configs/credentials.sh.example configs/credentials.sh
$EDITOR configs/credentials.sh
```

It exports the remote host and account, the conda interpreter, SLURM partitions,
the DDP variables, `WANDB_API_KEY`, `BIO_FILE` and `GENOTYPE_DATA_DIR`. Nothing
private is hard-coded elsewhere.

Everything else comes from the YAML passed to `--configFile`. Defaults are the
dataclass fields of `config/modelconfig.py`, and the shipped configs pin what each
run needs, so start a new one from an existing config.

## Pipeline

```bash
python segmenting.py --configFile configs/SGDP_chr19_ALL_seg128_overlap16.yaml
python train.py      --configFile configs/train_SGDP_chr19_ALL_seg128.yaml
python test.py       --configFile configs/test_seg128_SGDP_chr19_ALL.yaml
python benchmark.py  --configFile configs/test_seg128_SGDP_chr19_ALL.yaml
python benchmark.py  --report --configFile configs/test_seg128_SGDP_chr19_ALL.yaml \
                     --reportScda configs/test_scda_SGDP_chr19_ALL.yaml
```

Add `--impMethod beagle` to the benchmark step to score Beagle instead. The
evaluation config names the missing rates, masking seeds, marker set and
checkpoint, so one config is one evaluation.

## Datasets

1000 Genomes Project (1KGP), Louisiana Osteoporosis Study (LOS) and Simons Genome
Diversity Project (SGDP), at three scopes: region-specific models, genome-wide
models of 1KGP and SGDP, and array-to-WGS on chromosome 22.

| Region | Segmentation | Train | Eval |
|---|---|---|---|
| 1KGP chr22 ALL | `1KGP_chr22_ALL_seg128_overlap16` | `train_1KGP_chr22_ALL_seg128` | `test_seg128_1KGP_chr22_ALL` |
| LOS chr22 ALL  | `LOS_chr22_ALL_seg128_overlap16`  | `train_LOS_chr22_ALL_seg128`  | `test_seg128_LOS_chr22_ALL`  |
| LOS chr22 AA   | `LOS_chr22_AA_seg128_overlap16`   | `train_LOS_chr22_AA_seg128`   | `test_seg128_LOS_chr22_AA`   |
| LOS chr22 CA   | `LOS_chr22_CA_seg128_overlap16`   | `train_LOS_chr22_CA_seg128`   | `test_seg128_LOS_chr22_CA`   |
| SGDP chr22 ALL | `SGDP_chr22_ALL_seg128_overlap16` | `train_SGDP_chr22_ALL_seg128` | `test_seg128_SGDP_chr22_ALL` |
| SGDP chr19 ALL | `SGDP_chr19_ALL_seg128_overlap16` | `train_SGDP_chr19_ALL_seg128` | `test_seg128_SGDP_chr19_ALL` |
| SGDP HLA chr6  | `HLA_chr6_ALL_seg128_overlap64`   | `train_HLA_chr6_ALL_seg128`   | `test_seg128_HLA_chr6_ALL`   |

The SCDA baseline of each region uses `seg-1_overlap0` segmentation and
`train_scda_` / `test_scda_` configs.

## Genome-wide models

One model per dataset over all 22 autosomes, then fine-tuned per chromosome. Both
train on the target merged with the reference panel reindexed onto its axis.
Substitute `1KGP` for `SGDP` for the other family.

```bash
./submit.sh job/align_reference_SGDP.batch
./submit.sh job/train_SGDP_genomewide.batch
./submit.sh job/test_SGDP_genomewide.batch
./submit.sh job/benchmark_SGDP_genomewide.batch
./submit.sh job/finetune_SGDP_chromosomes.batch
./submit.sh job/test_SGDP_finetuned.batch
./submit.sh job/benchmark_SGDP_finetuned.batch
./submit.sh job/tables_genomewide.batch
./submit.sh job/figures_genomewide.batch
```

Each job walks the 22 chromosomes inside one allocation.
`job/train_SGDP_ablation_*.batch` run the reference-merge arms and
`job/beagle_excluded_markers.batch` counts the markers Beagle cannot match.
`python scripts/make_finetune_configs.py` writes the per-chromosome configs.

## Array-to-WGS

Chromosome 22 with only the positions an array types observed. `maskMode: chip`
trains on that pattern at 4,096 markers per segment.

```bash
./submit.sh job/rev2_chip_mask.batch          # test sets, target VCF, site list
./submit.sh job/rev2_seg_wide_chr22.batch     # segment 1KGP chr22
./submit.sh job/rev2_seg_wide_sgdp_all.batch  # segment SGDP
./submit.sh job/rev2_train_chip_scratch.batch # 1KGP, from random initialisation
./submit.sh job/rev2_train_sgdp_gw_chip.batch # SGDP, genome-wide
./submit.sh job/rev2_sgdp_chip_ft.batch       # specialise on one chromosome
./submit.sh job/rev2_beagle_chip.batch        # Beagle on the same positions
./submit.sh job/rev2_score_dosage_chip.batch  # dosage r squared for both
```

`python scripts/make_chip_configs.py` writes the configs.

## Running on SLURM

`job/` holds one batch script per stage and dataset. Each sources
`configs/credentials.sh` and writes to `logs/`; `submit.sh` injects the private
`--account` and `--mail-user` at submit time.

```bash
./submit.sh job/train_SGDP_chr19_ALL.batch
```

## Repository layout

```
config/     ModelConfig dataclass and YAML loader
configs/    segmentation / train / eval YAMLs, credentials template
data/       dataset, segmentation, metrics, masking utilities
model/      U-Net and SCDA architectures
job/        SLURM batch scripts
scripts/    config and job generators, reference alignment, table and figure builders
Results/    the reported tables and figures, and the code that draws them
*.py        segmenting / train / test / benchmark entry points
```

## Citation

Please cite the preprint. A peer-reviewed version is in revision.

> Huang L, Su K-J, Song M, Qiu C, Gragert L, Deng J, Luo Z, Tian Q, Gong P, Shen H, Zhang C, Deng H-W.
> *BiU-Net: a Biological-informed U-Net for Genotype Imputation.*
> Research Square (2025). https://doi.org/10.21203/rs.3.rs-6797863/v1

```bibtex
@article{biunet2025,
  author  = {Huang, Lei and Su, Kuan-Jui and Song, Meng and Qiu, Chuan and Gragert, Loren
             and Deng, Jeffrey and Luo, Zhe and Tian, Qing and Gong, Ping and Shen, Hui
             and Zhang, Chaoyang and Deng, Hong-Wen},
  title   = {BiU-Net: a Biological-informed U-Net for Genotype Imputation},
  journal = {Research Square (preprint)},
  year    = {2025},
  doi     = {10.21203/rs.3.rs-6797863/v1},
  url     = {https://doi.org/10.21203/rs.3.rs-6797863/v1}
}
```

## License

[MIT](LICENSE)
