CHR=6
QC_DIR=qc
SPLIT_DIR=split
SUBSET_DIR=subsets
DATASET=LOS

# Remove temp files for this chromosome
rm -f ${QC_DIR}/temp/*chr${CHR}*
rm -f ${QC_DIR}/temp/*_${CHR}.*

# Remove signal file (important!)
rm -f ${QC_DIR}/temp/Done_chr${CHR}.signal
