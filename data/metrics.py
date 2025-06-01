
import pandas as pd
import numpy as np

def calculate_maf(df):
    """
    Calculate Minor Allele Frequency (MAF) for each row (SNP) in a genotype dataframe
    where rows are SNPs and columns are samples.

    Parameters:
        df (pd.DataFrame): Genotype dataframe where rows are SNPs and columns are samples.

    Returns:
        pd.Series: Minor Allele Frequencies for each SNP (row).
    """
    # Count occurrences of 2 (heterozygous) and 3 (homozygous minor allele) across rows
    counts_23 = ((df == 2) | (df == 3)).sum(axis=1)  # 1 minor allele per occurrence
    counts_4 = (df == 4).sum(axis=1) * 2  # 2 minor alleles per occurrence

    # Count total alleles, accounting for diploid genotypes (2 alleles per individual)
    total_alleles = df.notna().sum(axis=1) * 2

    # Compute total minor alleles
    total_minor_alleles = counts_23 + counts_4

    # Calculate minor allele frequency (avoid division by zero)
    minor_allele_freq = total_minor_alleles / total_alleles
    minor_allele_freq[total_alleles == 0] = 0  # Set MAF to 0 for rows with no data

    return minor_allele_freq

def calculate_accuracy(orig, imputed, mask=None):
    # orig/imputed/mask are all numpy arrays with same shapes
    # Apply mask if provided
    if mask is not None:
        G = orig[mask]
        D = imputed[mask]
    else:
        G = orig
        D = imputed

    # Validate lengths
    if len(G) != len(D):
        raise ValueError("orig and imputed must have the same length")

    # Flatten arrays if they are multi-dimensional
    if G.ndim > 1:
        # Columns as SNPs:
        #G = G.flatten(order='F')
        #D = D.flatten(order='F')

        # Rows as SNPs:
        G = G.flatten()
        D = D.flatten()

    # Calculate accuracy
    correct = (G == D).sum()
    total = len(G)

    return correct / total if total > 0 else 0

def calculate_r2(orig, imputed, mask=None):

    if mask is not None:
        G = orig[mask]
        D = imputed[mask]
    else:
        G = orig
        D = imputed

    if len(G) != len(D):
        raise ValueError("orig and imputed must have the same length")

    # Flatten arrays if they are multi-dimensional
    if G.ndim > 1:
        # Columns as SNPs:
        #G = G.flatten(order='F')
        #D = D.flatten(order='F')

        # Rows as SNPs:
        G = G.flatten()
        D = D.flatten()

    if len(G) < 2:
        return None  # Not enough data to compute R²

    # Subtract 1 first (to account for the shift)
    G = G - 1
    D = D - 1

    G_mean = np.mean(G)
    D_mean = np.mean(D)

    numerator = np.sum((G - G_mean) * (D - D_mean)) ** 2
    denominator = np.sum((G - G_mean) ** 2) * np.sum((D - D_mean) ** 2)

    if denominator == 0:
        return None  # R² undefined due to zero variance

    return numerator / denominator


def calculate_iqs(orig, imputed, mask=None):
    if len(orig) != len(imputed):
        raise ValueError("orig and imputed must have the same length")

    if mask is not None:
        G = orig[mask]
        D = imputed[mask]
    else:
        G = orig
        D = imputed

    # Flatten arrays if they are multi-dimensional
    if G.ndim > 1:
        G = G.flatten()
        D = D.flatten()

    # Subtract 1 first (to account for the shift)
    G = G - 1
    D = D - 1

    # Convert phased genotype format to dosage format
    mapping = {0: 0, 1: 1, 2: 1, 3: 2}  # Mapping phased genotype to dosage
    G = np.vectorize(mapping.get)(G)
    D = np.vectorize(mapping.get)(D)

    contingency_table = np.zeros((3, 3))

    for o, i in zip(G, D):
        if 0 <= o <= 2 and 0 <= i <= 2:
            contingency_table[int(round(o)), int(round(i))] += 1

    n = np.sum(contingency_table)

    if n == 0:
        return 0

    p_o = np.trace(contingency_table) / n

    row_sums = np.sum(contingency_table, axis=1)
    col_sums = np.sum(contingency_table, axis=0)

    p_c = np.sum(row_sums * col_sums) / (n ** 2)

    iqs = (p_o - p_c) / (1 - p_c) if (1 - p_c) != 0 else 0

    return iqs

