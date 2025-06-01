import os
import gzip
import torch
import torch.nn as nn
import re
import pandas as pd
import numpy as np
import h5py
from data.dataset import SNPsDataset_HDF5
from data.segmentation import segmentation
from config.modelconfig import ModelConfig
import torch.nn.functional as F
from typing import List, Tuple, Optional
import glob
from pdb import set_trace

def get_dataset_paths(config: ModelConfig, segmentation: bool = False) -> Tuple[List[str], List[str], List[str], List[str]]:

    dataset_dir = f"../data/{config.dataset}/chunked/"

    # Get CSV files and sort them
    train_csv_gz_files = glob.glob(os.path.join(dataset_dir, f"{config.dataset}_chr{config.chromosome}_{config.population}_train_chunk*.csv.gz"))
    val_csv_gz_files = glob.glob(os.path.join(dataset_dir, f"{config.dataset}_chr{config.chromosome}_{config.population}_val_chunk*.csv.gz"))
    train_csv_gz_files.sort()
    val_csv_gz_files.sort()

    # Create base HDF5 file paths by replacing .csv.gz extension with .hdf5
    base_train_hdf5_files = [
        os.path.join(config.resDir, os.path.basename(csv_file).replace('.csv.gz', f'_seg{config.segLen}_overlap{config.overlap}.hdf5'))
        for csv_file in train_csv_gz_files
    ]

    base_val_hdf5_files = [
        os.path.join(config.resDir, os.path.basename(csv_file).replace('.csv.gz', f'_seg{config.segLen}_overlap{config.overlap}.hdf5'))
        for csv_file in val_csv_gz_files
    ]
    if segmentation: # For segmenation
        train_file_pairs = [
            (csv_file, hdf5_file)
            for csv_file, hdf5_file in zip(train_csv_gz_files, base_train_hdf5_files)
        ]
        val_file_pairs = [
            (csv_file, hdf5_file)
            for csv_file, hdf5_file in zip(val_csv_gz_files, base_val_hdf5_files)
        ]
        return train_file_pairs, val_file_pairs

    else: # For training
        # If finetuning is enabled and targetBin is specified, use filtered files
        if hasattr(config, 'finetuning') and config.finetuning and hasattr(config, 'targetBin') and config.targetBin:
            train_hdf5_files = [
                file.replace('.hdf5', f'_filtered_{config.targetBin}.hdf5')
                for file in base_train_hdf5_files
            ]

            val_hdf5_files = [
                file.replace('.hdf5', f'_filtered_{config.targetBin}.hdf5')
                for file in base_val_hdf5_files
            ]

            print(f"Using filtered files for target bin: {config.targetBin}")
        else:
            # Use unfiltered files
            train_hdf5_files = base_train_hdf5_files
            val_hdf5_files = base_val_hdf5_files

        return train_hdf5_files, val_hdf5_files

def save_checkpoint(state, epoch, config, filename_prefix="checkpoint"):
    checkpointName = os.path.join(config.modelDir, f"{filename_prefix}_{config.run}_epoch_{epoch}.pth")
    torch.save(state, checkpointName)

def find_latest_checkpoint(config, filename_prefix="checkpoint"):
    # Construct the glob pattern to match files based on the provided filename pattern
    pattern = os.path.join(config.modelDir, f"{filename_prefix}_{config.run}_epoch_*.pth")
    list_of_files = glob.glob(pattern)

    if not list_of_files:
        return None

    # Initialize variables to track the latest file and highest epoch
    latest_file = None
    highest_epoch = -1

    # Loop through each file, extract the epoch number, and determine if it is the latest
    for file in list_of_files:
        match = re.search(fr"epoch_(\d+)\.pth", file)
        if match:
            epoch_number = int(match.group(1))
            if epoch_number > highest_epoch:
                highest_epoch = epoch_number
                latest_file = file

    return latest_file


class F1Loss(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.epsilon = config.f1Epsilon

    def forward(self, y_pred, y_true):
        # y_pred: (batch_size, num_classes)
        # y_true: (batch_size)
        y_true = F.one_hot(y_true, num_classes=y_pred.size(1)).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        # avoid extremely small or large values
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class FocalLoss(nn.Module):
    def __init__(self, config: ModelConfig):
        super(FocalLoss, self).__init__()
        self.alpha = config.focalAlpha
        self.gamma = config.focalGamma

    def forward(self, inputs, targets):
        bce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # prevents nans if probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class HybridFocalLoss(nn.Module):
    def __init__(self, config: ModelConfig):
        super(HybridFocalLoss, self).__init__()
        # Parameters - unchanged
        self.gamma = getattr(config, 'focalGamma', 2.0)
        self.lambda_dosage = getattr(config, 'dosageLossLambda', 0.3)
        self.use_label_smoothing = getattr(config, 'useLabelSmoothing', False)
        self.label_smoothing_gamma = getattr(config, 'labelSmoothingGamma', 0.1)

        # Class proportions (as provided)
        class_proportions = {
            1: 0.60,  # 60%
            2: 0.13,  # 13%
            3: 0.13,  # 13%
            4: 0.14,  # 14%
        }

        # Set alpha values based on inverse class frequency
        alphas_inverse = {cls: 1.0 / prop for cls, prop in class_proportions.items()}
        alphas_sqrt = {cls: 1.0 / np.sqrt(prop) for cls, prop in class_proportions.items()}

        # Choose which alpha method to use
        alpha_method = getattr(config, 'alphaMethod', 'sqrt')
        selected_alphas = alphas_sqrt if alpha_method != 'inverse' else alphas_inverse

        # Normalize alphas to have mean = 0.25
        mean_alpha = sum(selected_alphas.values()) / len(selected_alphas)
        normalized_alphas = {cls: (alpha * 0.25 / mean_alpha)
                             for cls, alpha in selected_alphas.items()}

        # Register alpha values as buffer
        self.register_buffer('alphas', torch.tensor(
            [normalized_alphas.get(i, 0.25) for i in range(max(normalized_alphas.keys()) + 1)],
            dtype=torch.float32
        ))

    def compute_loss(self, inputs, targets, use_label_smoothing=None):
        """
        Optimized compute_loss function to avoid CPU-GPU transfers
        """
        # Determine whether to use label smoothing
        if use_label_smoothing is None:
            use_label_smoothing = self.use_label_smoothing

        # Create indices tensor on same device as inputs
        batch_indices = torch.arange(targets.size(0), device=inputs.device)

        if use_label_smoothing:
            # Get probabilities
            probs = F.softmax(inputs, dim=1)

            # Extract probability of true class for each sample - vectorized
            p_y_given_x = probs[batch_indices, targets]

            # Label smoothing parameter
            gamma = self.label_smoothing_gamma

            # Modified cross-entropy with label smoothing
            bce_loss = -torch.log(gamma + (1-gamma) * p_y_given_x) / (1-gamma)

            # Focal weighting (using smoothed probabilities)
            pt = gamma + (1-gamma) * p_y_given_x
        else:
            # Standard cross entropy for basic loss calculation
            bce_loss = F.cross_entropy(inputs, targets, reduction='none')

            # Probability of the target class
            pt = torch.exp(-bce_loss)

        # Get alpha value for each target - ensure same device
        alpha_t = self.alphas[targets]  # alphas should automatically be on correct device

        # Apply class-specific alpha weighting
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        classification_loss = focal_loss.mean()

        return classification_loss

    def forward(self, inputs, targets):
        # Compute the classification loss
        classification_loss = self.compute_loss(inputs, targets)

        # Convert logits to probabilities for dosage calculation
        probs = F.softmax(inputs, dim=1)

        # Create dosage targets - vectorized without repeated where calls
        dosage_targets = torch.zeros_like(targets, dtype=torch.float)
        mask_2 = (targets == 2)
        mask_3 = (targets == 3)
        mask_4 = (targets == 4)

        dosage_targets[mask_2] = 1.0  # 0|1
        dosage_targets[mask_3] = 1.0  # 1|0
        dosage_targets[mask_4] = 2.0  # 1|1

        # Calculate predicted dosage
        predicted_dosage = probs[:, 2] + probs[:, 3] + 2 * probs[:, 4]

        # Standard MSE loss
        dosage_loss = F.mse_loss(predicted_dosage, dosage_targets)

        # Combine the losses
        total_loss = (1 - self.lambda_dosage) * classification_loss + self.lambda_dosage * dosage_loss

        return total_loss

class HybridWeightedFocalLoss(nn.Module):
    def __init__(self, config: ModelConfig):
        super(HybridWeightedFocalLoss, self).__init__()

        # Configuration settings - unchanged
        self.gamma = getattr(config, 'weightedFocalGamma', 2.0)
        self.lambda_dosage = getattr(config, 'dosageLossLambda', 0.3)
        self.use_label_smoothing = getattr(config, 'useLabelSmoothing', False)
        self.label_smoothing_gamma = getattr(config, 'labelSmoothingGamma', 0.1)
        self.normalize_penalties = getattr(config, 'normalizePenalties', False)

        # Create penalty matrix for different misclassifications
        penalty_matrix = torch.ones(config.vocabSize, config.vocabSize)
        # Define penalties based on dosage misclassification severity
        severe_penalty = 3.0
        mild_penalty = 1.0
        # Set up penalty matrix - unchanged logic
        penalty_matrix[1, 2] = severe_penalty
        penalty_matrix[1, 3] = severe_penalty
        penalty_matrix[1, 4] = severe_penalty
        penalty_matrix[2, 1] = severe_penalty
        penalty_matrix[2, 4] = severe_penalty
        penalty_matrix[2, 3] = mild_penalty
        penalty_matrix[3, 1] = severe_penalty
        penalty_matrix[3, 4] = severe_penalty
        penalty_matrix[3, 2] = mild_penalty
        penalty_matrix[4, 1] = severe_penalty
        penalty_matrix[4, 2] = severe_penalty
        penalty_matrix[4, 3] = severe_penalty
        self.register_buffer('penalty_matrix', penalty_matrix)

    def compute_loss(self, inputs, targets, use_label_smoothing=None):
        """
        Compute classification loss with vectorized operations to avoid CPU transfers
        """
        # Determine whether to use label smoothing
        if use_label_smoothing is None:
            use_label_smoothing = self.use_label_smoothing

        # Create index tensor on same device as inputs
        batch_indices = torch.arange(targets.size(0), device=inputs.device)

        # Make sure penalty matrix is on the same device as inputs
        # This line shouldn't be needed if register_buffer works correctly,
        # but let's add it to be safe
        penalty_matrix = self.penalty_matrix.to(inputs.device)

        if use_label_smoothing:
            # Get probabilities
            probs = F.softmax(inputs, dim=1)

            # Extract probability of true class for each sample - vectorized
            p_y_given_x = probs[batch_indices, targets]

            # Label smoothing parameter
            gamma = self.label_smoothing_gamma

            # Modified cross-entropy with label smoothing
            ce_loss = -torch.log(gamma + (1-gamma) * p_y_given_x) / (1-gamma)

            # Focal weighting (using smoothed probabilities)
            pt = gamma + (1-gamma) * p_y_given_x
        else:
            # Standard cross entropy for basic loss calculation
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

            # Probability of the target class
            pt = torch.exp(-ce_loss)

        # Focal weighting
        focal_weight = (1 - pt) ** self.gamma

        # Get predicted classes for penalty application
        pred_classes = inputs.argmax(dim=1)

        # Apply penalty based on true class and predicted class - VECTORIZED
        # Make sure all tensors are on the same device
        penalties = penalty_matrix[targets, pred_classes]

        # Normalize penalties if configured - without .max().item()
        if self.normalize_penalties:
            penalties_max = penalties.max()
            if penalties_max > 0:
                penalties = penalties / penalties_max

        # Weighted focal loss with penalties
        classification_loss = (penalties * focal_weight * ce_loss).mean()

        return classification_loss

    def forward(self, inputs, targets):
        # Compute the classification loss
        classification_loss = self.compute_loss(inputs, targets)

        # Convert logits to probabilities for dosage calculation
        probs = F.softmax(inputs, dim=1)

        # Create dosage targets - VECTORIZED without .item() calls
        dosage_targets = torch.zeros_like(targets, dtype=torch.float)
        # Vectorized assignment instead of where()
        mask_2 = (targets == 2)
        mask_3 = (targets == 3)
        mask_4 = (targets == 4)
        dosage_targets[mask_2] = 1.0  # 0|1
        dosage_targets[mask_3] = 1.0  # 1|0
        dosage_targets[mask_4] = 2.0  # 1|1

        # Calculate predicted dosage
        predicted_dosage = probs[:, 2] + probs[:, 3] + 2 * probs[:, 4]

        # Apply a weighted MSE loss for dosage prediction
        dosage_diff = torch.abs(predicted_dosage - dosage_targets)
        # Apply weights that grow with error magnitude
        weighting_factor = 1.0 + dosage_diff  # Linear growth with error
        weighted_dosage_loss = (weighting_factor * dosage_diff**2).mean()

        # Combine the losses
        total_loss = (1 - self.lambda_dosage) * classification_loss + self.lambda_dosage * weighted_dosage_loss

        return total_loss

class WeightedFocalLoss(nn.Module):
    def __init__(self, config: ModelConfig):
        super(WeightedFocalLoss, self).__init__()
        self.gamma = config.weightedFocalGamma

        # Create a full penalty matrix to match vocabSize=6
        penalty_matrix = torch.ones(config.vocabSize, config.vocabSize)

        # Setting the penalties according to your requirements
        severe_penalty = 3.0
        mild_penalty = 1.0

        # Set penalty values - unchanged
        penalty_matrix[1, 2] = severe_penalty
        penalty_matrix[1, 3] = severe_penalty
        penalty_matrix[1, 4] = severe_penalty
        penalty_matrix[2, 1] = severe_penalty
        penalty_matrix[2, 4] = severe_penalty
        penalty_matrix[2, 3] = mild_penalty
        penalty_matrix[3, 1] = severe_penalty
        penalty_matrix[3, 4] = severe_penalty
        penalty_matrix[3, 2] = mild_penalty
        penalty_matrix[4, 1] = severe_penalty
        penalty_matrix[4, 2] = severe_penalty
        penalty_matrix[4, 3] = severe_penalty

        self.register_buffer('penalty_matrix', penalty_matrix)

    def forward(self, inputs, targets):
        # Standard cross entropy for basic loss calculation
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Focal weighting
        pt = torch.exp(-ce_loss)  # probability of the target class
        focal_weight = (1 - pt) ** self.gamma

        # Get predicted classes - use argmax instead of max to avoid unpacking
        pred_classes = inputs.argmax(dim=1)

        # Apply penalty based on true class and predicted class - VECTORIZED
        # This is the key optimization - replace the loop with direct indexing
        penalties = self.penalty_matrix[targets, pred_classes]

        # Final loss with penalties and focal weighting
        loss = penalties * focal_weight * ce_loss
        return loss.mean()

def filter_segments_by_targetSNPs(batch_snpsIndex, target_snps):
    device = batch_snpsIndex.device

    # Extract sites from batch_snpsIndex
    col_idx = batch_snpsIndex[:, :, 0]  # Shape: (batch_size, seq_len)

    # Identify columns corresponding to target SNPs
    is_in_target = torch.isin(col_idx, target_snps)

    # Determine which samples have at least one target SNP
    filtered_row_indices = is_in_target.any(dim=1)  # Shape: (batch_size,)

    return filtered_row_indices

def mask_random_positions(config, data_batch, bert_strategy=False):
    """Simplified masking that's more performance-focused"""
    # Create mask tensor with target percentage of 1s
    mask = torch.rand_like(data_batch, dtype=torch.float) < config.missingRatio

    # Only apply to valid positions
    valid_positions = (data_batch >= 1) & (data_batch < config.vocabSize - 1)
    mask = mask & valid_positions

    # Create masked data
    masked_data = data_batch.clone()
    masked_data[mask] = config.missingId

    return masked_data, mask

# Original version, too delicated, too slow
def mask_random_positions_slow(config, data_batch, bert_strategy=False):
    """
    Fully vectorized implementation of mask_random_positions without for-loops.
    """
    # Find maskable tokens
    maskable_values = (data_batch >= 1) & (data_batch < (config.vocabSize - 1))
    batch_size, seq_len = data_batch.shape

    # Compute number of tokens to mask per row
    if config.dynamicRatio:
        random_missing_ratios = torch.rand(batch_size, device=data_batch.device) * config.missingRatio
        num_to_mask_per_row = (maskable_values.sum(dim=1) * random_missing_ratios).int()
    else:
        num_to_mask_per_row = (maskable_values.sum(dim=1) * config.missingRatio).int()

    # Create probability mask for selecting tokens to mask
    rand_tensor = torch.rand(batch_size, seq_len, device=data_batch.device)
    # Set probabilities to 0 for non-maskable positions
    rand_tensor = rand_tensor * maskable_values.float()

    # Create a mask tensor to track masked positions
    mask_tensor = torch.zeros_like(data_batch, dtype=torch.bool, device=data_batch.device)

    # Clone data for masking
    masked_data = data_batch.clone()

    # Create a sorting threshold for each row
    max_num_to_mask = num_to_mask_per_row.max().item()
    if max_num_to_mask == 0:
        return masked_data, mask_tensor

    # Instead of iterating through rows, we'll use top-k to get indices
    # First, get topk values and their indices across the batch
    _, topk_indices = torch.topk(rand_tensor, k=min(max_num_to_mask, seq_len), dim=1, sorted=True)

    # Create a mask that we can use to selectively apply different masks
    index_mask = torch.arange(max_num_to_mask, device=data_batch.device).expand(batch_size, -1)
    valid_mask = index_mask < num_to_mask_per_row.unsqueeze(1)

    # For simple masking (non-BERT strategy)
    if not bert_strategy:
        # Get batch indices and valid topk indices
        batch_indices = torch.arange(batch_size, device=data_batch.device).unsqueeze(1).expand(-1, max_num_to_mask)
        valid_topk = topk_indices.masked_select(valid_mask).view(-1)
        batch_indices = batch_indices.masked_select(valid_mask).view(-1)

        # Set masked values
        masked_data[batch_indices, valid_topk] = config.missingId
        mask_tensor[batch_indices, valid_topk] = True
    else:
        # For BERT strategy, we need to handle different mask types
        # Compute indices for each mask type (mask token, random token, unchanged)
        mask_ratio = 0.8
        random_ratio = 0.1
        # unchanged_ratio = 0.1 (remainder)

        num_mask_tokens = (num_to_mask_per_row * mask_ratio).long()
        num_random_tokens = (num_to_mask_per_row * random_ratio).long()

        # Create mask for each type
        mask_indices_mask = index_mask < num_mask_tokens.unsqueeze(1)
        random_indices_mask = (index_mask >= num_mask_tokens.unsqueeze(1)) & (index_mask < (num_mask_tokens + num_random_tokens).unsqueeze(1))
        unchanged_indices_mask = (index_mask >= (num_mask_tokens + num_random_tokens).unsqueeze(1)) & valid_mask

        # Apply mask token
        batch_indices = torch.arange(batch_size, device=data_batch.device).unsqueeze(1).expand(-1, max_num_to_mask)
        mask_topk = topk_indices.masked_select(mask_indices_mask).view(-1)
        mask_batch_indices = batch_indices.masked_select(mask_indices_mask).view(-1)
        masked_data[mask_batch_indices, mask_topk] = config.missingId

        # Apply random tokens
        random_topk = topk_indices.masked_select(random_indices_mask).view(-1)
        random_batch_indices = batch_indices.masked_select(random_indices_mask).view(-1)
        random_tokens = torch.randint(1, config.vocabSize - 1,
                                     size=(len(random_topk),),
                                     device=data_batch.device)
        masked_data[random_batch_indices, random_topk] = random_tokens

        # Mark all positions as masked in the mask tensor
        # Combine all indices
        all_masked_topk = topk_indices.masked_select(valid_mask).view(-1)
        all_masked_batch = batch_indices.masked_select(valid_mask).view(-1)
        mask_tensor[all_masked_batch, all_masked_topk] = True

    return masked_data, mask_tensor

def mask_random_positions_bias(config, data_batch, bert_strategy=False):
    """
    Mask tokens in `data_batch` using:
      - First prioritize masking `2`, `3` or `4`
      - If `2`, `3` or `4` are insufficient, fill the rest with `1`
      - If `bert_strategy=True`, follows 80/10/10 masking (like BERT)
      - If `bert_strategy=False`, simply replaces chosen tokens with `config.missingId`

    Args:
        config: Configuration object containing missingRatio, upsamplingRatio, and missingId.
        data_batch (torch.Tensor): Input tensor with tokens.
        bert_strategy (bool): If True, applies BERT-style 80/10/10 masking; otherwise, simple masking.

    Returns:
        masked_data (torch.Tensor): Masked input tensor.
        mask_tensor (torch.Tensor): Boolean mask indicating which tokens were masked.
    """

    # Ensure `data_batch` is integer type for bitwise operations
    data_batch = data_batch.to(torch.int)

    # Find maskable tokens (only `1`, `2`, `3` or `4`)
    maskable = (data_batch >= 1) & (data_batch < config.vocabSize - 1)
    maskable_234 = (data_batch >= 2) & (data_batch < config.vocabSize - 1)
    maskable_1 = data_batch == 1

    # Total number of tokens to mask in the batch
    num_to_mask = int(config.missingRatio * maskable.sum().item())

    if num_to_mask <= 0:
        return None, None  # Skip batch if no tokens need to be masked

    # Compute how many `2/3` tokens we ideally want to mask
    num_to_mask_234 = int(num_to_mask * config.upsamplingRatio)

    # First try to fill `num_to_mask_234` positions from `2/3/4`
    maskable_234_indices = maskable_234.nonzero(as_tuple=True)[0]  # Get 1D indices

    if len(maskable_234_indices) >= num_to_mask_234:
        # If we have enough `2/3/4`, select randomly from them
        chosen_indices = maskable_234_indices[torch.randperm(len(maskable_234_indices))[:num_to_mask_234]]
    else:
        # Otherwise, take all `2/3/4` and fill the rest with `1`
        chosen_indices = maskable_234_indices  # Take all available `2/3/4`
        remaining_mask = num_to_mask - len(chosen_indices)

        # Now pick `remaining_mask` tokens from `1`
        if remaining_mask > 0:
            maskable_1_indices = maskable_1.nonzero(as_tuple=True)[0]  # Get 1D indices
            if len(maskable_1_indices) > 0:
                extra_indices = maskable_1_indices[torch.randperm(len(maskable_1_indices))[:remaining_mask]]
                chosen_indices = torch.cat((chosen_indices, extra_indices))

    # Create tensors to store mask indices
    mask_tensor = torch.zeros_like(data_batch, dtype=torch.bool, device=data_batch.device)
    mask_tensor.view(-1)[chosen_indices] = True  # Mark chosen positions as masked
    masked_data = data_batch.clone()

    if bert_strategy:
        # Split chosen indices into 80% masked, 10% random, and 10% unchanged
        num_mask_tokens = int(0.8 * len(chosen_indices))
        num_random_tokens = int(0.1 * len(chosen_indices))
        num_unchanged_tokens = len(chosen_indices) - num_mask_tokens - num_random_tokens

        # Shuffle chosen indices to randomize assignment
        shuffled_indices = torch.randperm(len(chosen_indices))

        # Apply `[MASK]` (set to `config.missingId`)
        mask_indices = chosen_indices[shuffled_indices[:num_mask_tokens]]
        masked_data.view(-1)[mask_indices] = config.missingId

        # Apply random tokens
        if num_random_tokens > 0:
            random_indices = chosen_indices[shuffled_indices[num_mask_tokens:num_mask_tokens + num_random_tokens]]
            random_tokens = torch.randint(1, config.vocabSize - 1, size=(len(random_indices),), device=data_batch.device)
            masked_data.view(-1)[random_indices] = random_tokens.to(masked_data.dtype)

        # The remaining 10% are left unchanged

    else:
        # Simple masking: Directly replace all chosen tokens with `config.missingId`
        masked_data.view(-1)[chosen_indices] = config.missingId

    return masked_data, mask_tensor

def save_to_csv_gz(df, filename):
    # Create the directory if it doesn't exist
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)  # Create any missing directories along the path

    # Save the DataFrame to a compressed CSV file with index as 'SampleID'
    with gzip.open(filename, 'wt') as f:
        df.reset_index().rename(columns={'index': 'SampleID'}).to_csv(f, index=False)
    print(f"File saved at: {filename}")


def one_hot_encode(data, num_categories=6, device=torch.device('cpu')): #config.vocabSize = 6
    data = data.long()
    encoded = torch.zeros(data.shape[0], data.shape[1], num_categories, dtype=torch.float32, device=data.device)
    encoded.scatter_(2, data.unsqueeze(-1), 1)
    return encoded


def cleanup_memory(datasets=None, dataloaders=None, samplers=None, tensors=None, models=None, optimizer=None, scheduler=None, force_os_release=False, scaler=None):
    """
    Comprehensive memory cleanup function for both training and testing.

    Args:
        datasets: List of dataset objects to clean up
        dataloaders: List of dataloader objects to clean up
        samplers: List of sampler objects to clean up
        tensors: List of tensors to delete
        models: Models to clear from memory
        optimizer: Optimizer to clear
        scheduler: Scheduler to clear
        force_os_release: Whether to attempt to release memory back to OS (CPU only)
        scaler: GradScaler for mixed precision training
    """
    # First cleanup dataloaders since they hold references to datasets
    if dataloaders:
        for loader in dataloaders:
            # Close dataset if it has a close method
            if hasattr(loader.dataset, 'close'):
                loader.dataset.close()
            # Try to clean up workers
            if hasattr(loader, '_iterator'):
                del loader._iterator
        # Delete loader references
        for loader in dataloaders:
            del loader

    # Cleanup samplers
    if samplers:
        for sampler in samplers:
            del sampler

    # Clean up datasets
    if datasets:
        for dataset in datasets:
            # Close HDF5 files properly
            if hasattr(dataset, 'data_file') and dataset.data_file is not None:
                dataset.data_file.close()
                dataset.data_file = None
            # Clean preloaded data
            if hasattr(dataset, 'clear_preloaded_data'):
                dataset.clear_preloaded_data()
            # Delete any large tensors in the dataset
            if hasattr(dataset, 'snps'):
                del dataset.snps
            if hasattr(dataset, 'snpsIndex'):
                del dataset.snpsIndex
        # Now delete the dataset references
        for dataset in datasets:
            del dataset

    # Clean up models
    if models:
        for model in models:
            if hasattr(model, 'zero_grad'):
                model.zero_grad(set_to_none=True)

    # Clean up optimizer
    if optimizer:
        optimizer.zero_grad(set_to_none=True)
        del optimizer

    # Clean up scheduler
    if scheduler:
        del scheduler

    # Clean up mixed precision scaler
    if scaler:
        del scaler

    # Clean up tensors
    if tensors:
        for tensor in tensors:
            del tensor

    # GPU-specific cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force garbage collection
    import gc
    gc.collect()
    gc.collect()

    # CPU-specific memory release
    if not torch.cuda.is_available() and force_os_release:
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            # malloc_trim releases memory back to the system
            libc.malloc_trim(0)
        except Exception as e:
            pass # Skip if not supported
