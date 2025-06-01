import re
import numpy as np
import os
import pandas as pd
from typing import List, Tuple, Optional
from data.metrics import calculate_maf
import h5py
from config.modelconfig import ModelConfig

def prepare_target_snps(config: ModelConfig, bin_range: str):
    print("Calculating MAF from complete training set...")
    train_file = f"../data/{config.dataset}/split/{config.dataset}_chr{config.chromosome}_{config.population}_train.csv.gz"

    df_train = pd.read_csv(train_file, compression='gzip', index_col=0)
    df_train.columns = df_train.columns.astype(int)
    # pattern = re.compile(r'\d+$')
    # df_train.index = df_train.index.map(lambda x: int(pattern.search(x).group()) if pattern.search(x) else None).astype(int)

    maf = calculate_maf(df_train)
    maf_bins = pd.cut(maf, bins=config.bins, labels=config.binRanges, include_lowest=True, right=True)
    target_snps = maf_bins[maf_bins == bin_range].index
    return target_snps

def segmentation(config: ModelConfig, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int]:

    depth = config.depth
    segLen = config.segLen
    overlap = config.overlap
    padId = config.padId
    padLabel = config.padLabel

    divisor = 2 ** (depth - 1)
    if segLen == -1:
        segLen = df.shape[0]
        overlap = 0
        if segLen % divisor == 0:
            snps = df.values.transpose()
            padded_df = df.transpose()
            #segLen = df.shape[0]
            padding_per_col = 0
        else:
            padding_per_col = divisor - (segLen % divisor)
            padded_df = pd.DataFrame(np.pad(df.values, ((0, padding_per_col), (0, 0)), constant_values=padId))
            padded_df.columns = df.columns
            padded_df.index = list(df.index) + [padLabel] * padding_per_col
            snps = padded_df.values.transpose()
            segLen = df.shape[0] + padding_per_col
        snpsIndex = np.array([[(col, row_idx) for col in padded_df.index] for row_idx in padded_df.columns])
        return snps, snpsIndex, padding_per_col

    effective_seg_len = segLen - overlap
    num_segments = int(np.ceil((df.shape[0] - overlap) / effective_seg_len))
    total_length_needed = effective_seg_len * (num_segments - 1) + segLen
    padding_per_col = max(0, total_length_needed - df.shape[0])

    if padding_per_col > 0:
        padded = np.pad(df.values, ((0, padding_per_col), (0, 0)), constant_values=padId)
        padded_index = list(df.index) + [padLabel] * padding_per_col
    else:
        padded = df.values
        padded_index = list(df.index)

    start_indices = np.arange(0, df.shape[0] - overlap, effective_seg_len)
    snps = np.array([padded[start:start + segLen, :] for start in start_indices]).transpose(0, 2, 1).reshape(-1, segLen)

    loci = [padded_index[start:start + segLen] for start in start_indices]
    sampleid = df.columns.astype(int)
    snpsIndex = np.array([np.array([[(col, idx) for col in loci_list] for idx in sampleid]) for loci_list in loci])
    snpsIndex = snpsIndex.reshape(snpsIndex.shape[0] * snpsIndex.shape[1], snpsIndex.shape[2], snpsIndex.shape[3])
    return snps, snpsIndex, padding_per_col


def de_segmentation(config: ModelConfig, segments_predictions, segments_indices, segments_confidences):
    """
    De-segment and merge predictions using confidence scores for SCDA model.

    Args:
        config: Configuration object with attributes:
            - test_csv_gz: Path to the compressed test dataset.
            - overlap: Number of overlapping elements between segments.
            - padId: ID representing padding tokens.
            - vocabSize: Vocabulary size for confidence scores.
        segments_predictions (np.ndarray): Predicted values for segments.
        segments_indices (np.ndarray): Indices corresponding to test_df.
        segments_confidences (np.ndarray): Confidence scores per token.

    Returns:
        pd.DataFrame: Restored predictions after merging.
    """

    # Load test DataFrame
    test_df = pd.read_csv(config.test_csv_gz, compression='gzip', index_col=0)
    index_to_row = {idx: i for i, idx in enumerate(test_df.index)}  # Map test index to row position
    column_to_col = {col: i for i, col in enumerate(test_df.columns.astype(int))}  # Map test columns to array columns

    # Initialize NumPy arrays to hold restored values and confidence scores
    restored_array = np.full((len(test_df.index), len(test_df.columns)), config.padId, dtype=np.int64)
    confidence_array = np.full_like(restored_array, -1, dtype=np.float64)  # Track confidence scores

    # Process each segment in a batch
    for batch, predictions, confidences in zip(segments_indices, segments_predictions, segments_confidences):
        col_name = batch[0, 1]  # Column name, sampleid
        indices = batch[:, 0]  # Extract row indices, loci or say SNPs of this segment

        # Mask valid indices (excluding -3 as config.padLabel)
        valid_mask = (indices != config.padLabel)

        # Get valid indices, predictions, and confidences
        valid_indices = indices[valid_mask]
        valid_values = predictions[valid_mask]
        valid_confidences = confidences[valid_mask]  # Shape: (num_valid_indices, config.vocabSize)

        # Get the highest confidence per token
        best_confidences = np.max(valid_confidences, axis=1)  # Best confidence per token

        # Convert indices to row positions in `restored_array`
        row_positions = np.array([index_to_row[idx] for idx in valid_indices])

        # Convert column to position
        col_position = column_to_col[col_name]

        # Resolve conflicts: Only update where confidence is higher than stored confidence
        mask = best_confidences > confidence_array[row_positions, col_position]
        restored_array[row_positions[mask], col_position] = valid_values[mask]
        confidence_array[row_positions[mask], col_position] = best_confidences[mask]

    # Convert NumPy array to DataFrame
    restored_df = pd.DataFrame(restored_array, index=test_df.index, columns=test_df.columns)

    return restored_df
