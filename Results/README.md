# Results

The result tables and figures the paper reports, and the code that draws them.

Every number in a table or a figure is read from a file the pipeline wrote. No
value is transcribed into a plotting or table script, so re-running a stage and
re-running its renderer is enough to update what the paper shows.

## What builds what

### Main text

| Artifact | Built by | Reads |
|---|---|---|
| Figure 1 | `Results/Figure_S4_9.py` | `Results/exp1_1KGP.csv` |
| Figure 2 | `Results/Figure_S10.py` | the test splits and imputed matrices named by `configs/test_seg128_LOS_chr22_{AA,CA,ALL}.yaml` |
| Figure 3 | `Results/Figure_3.py` | `Results/LOS_demographic.csv` (see *Data not redistributed*) |
| Figures 4, 5 | schematics, drawn by hand | |
| Table 1 | `scripts/build_dataset_table.py` | the train, validation and test splits of each dataset |
| Table 2 | `scripts/complexity_report.py` | `configs/complexity_reference.yaml`, the one architecture both models are profiled at |
| Table 3 | the phased-state cost matrix: a definition, not a measurement | |

### Supplement

| Artifact | Built by | Reads |
|---|---|---|
| Tables S1–S21 | `scripts/build_region_tables.py`, typeset through `Tables/final_exp1.xlsx` and `Tables/final_exp2.xlsx` | benchmark cells in `analysis/` |
| Tables S22–S24 | `scripts/build_ignored_marker_tables.py` | the `Num_SNPs_total` and `Num_SNPs_removed` columns of the Beagle cells |
| Tables S25, S26 | `scripts/build_main_tables.py` | benchmark cells in `analysis/` |
| Tables S27, S32 | `scripts/build_genomewide_vs_region_tables.py`, `scripts/build_ablation_tables.py` | benchmark cells, plus the submitted supplement for the region-specific columns |
| Table S28 | `scripts/build_finetuning_tables.py` | benchmark cells in `analysis/` |
| Table S29 | `scripts/build_ablation_tables.py` | benchmark cells in `analysis/` |
| Table S30 | `scripts/beagle_excluded_markers.py` | the target and panel VCFs |
| Table S31 | `scripts/build_excluded_recovery_table.py` | the cells `benchmark.py` wrote with `excludedOnly` set |
| Figures S1, S2 | `Results/Figure_S1.py`, `Results/Figure_S2.py` | `Results/S1.csv` |
| Figure S3 | `benchmark.py` writes a confusion matrix beside every cell it scores; `job/figures.batch` copies the pair out under their figure names | the LOS Caucasian cells at 5% masking, Beagle and BiU-Net |
| Figure S4 | the same render as Figure 1 | |
| Figures S5–S9 | `Results/Figure_S4_9.py` | `Results/exp1_*.csv`, `Results/exp2_*.csv` |
| Figure S10 | the same render as Figure 2 | |
| Figure S11 | `scripts/plot_accuracy_by_frequency.py` | benchmark cells in `analysis/` |
| Figure S12 | `scripts/plot_accuracy_by_chromosome.py` | benchmark cells in `analysis/` |
| Figure S13 | `scripts/plot_beagle_excluded_markers.py` | `Results/beagle_excluded_markers_SGDP.csv` |

## Rebuilding

```bash
./submit.sh job/tables_region.batch         # Results/exp1_*.csv, exp2_*.csv and the excluded-marker counts
./submit.sh job/figures.batch               # Figures 1, 2, 3, S1-S10
./submit.sh job/tables_genomewide.batch     # the genome-wide tables
./submit.sh job/figures_genomewide.batch    # Figures S11-S13
./submit.sh job/complexity_report.batch     # Tables 1 and 2
```

The renderers take a few minutes and need no GPU: they read the benchmark cells
and the imputed matrices the training and scoring stages already wrote.

## Marker counts

On the datasets where the reference panel is missing markers the target carries,
the three methods are not counted over the same set. Beagle scores only the
markers it matched, and the reference-free models score the whole target axis,
which is why `Num_SNPs` differs by method in `exp1_SGDP.csv` and `exp1_HLA.csv`
and is identical in the others. Each evaluation config states its own scope
through `overlappedOnly`, and `scripts/build_region_tables.py` follows it.

## Data not redistributed

`Results/LOS_demographic.csv` and the LOS genotype matrices are individual-level
records of Louisiana Osteoporosis Study participants and are not part of this
repository. `Figure_3.py` and `Figure_S10.py` therefore run only where those
files are available; every other renderer here runs from the committed CSVs.
