import pandas as pd
import numpy as np

# horizontally split version
def calculate_maf(df):
    """
    Calculate Minor Allele Frequency (MAF ≤ 0.5) for each row (SNP) in a genotype dataframe,
    where rows are SNPs and columns are samples. Genotype encoding:
    1 = A|A (0 minor alleles), 2 = A|a (1 minor allele), 3 = a|A (1 minor allele), 4 = a|a (2 minor alleles)

    Parameters:
        df (pd.DataFrame): Genotype dataframe with rows as SNPs and columns as samples.

    Returns:
        pd.Series: Minor Allele Frequencies for each SNP (row), guaranteed to be ≤ 0.5.
    """
    # Count 1 minor allele from heterozygous (2, 3)
    counts_23 = ((df == 2) | (df == 3)).sum(axis=1)

    # Count 2 minor alleles from homozygous alternate (4)
    counts_4 = (df == 4).sum(axis=1) * 2

    # Total observed alleles (2 per non-missing genotype)
    total_alleles = df.notna().sum(axis=1) * 2

    # Total alternate allele count (as currently encoded)
    alt_freq = (counts_23 + counts_4) / total_alleles
    alt_freq[total_alleles == 0] = 0

    # Convert to minor allele frequency (MAF ≤ 0.5)
    maf = alt_freq.apply(lambda x: min(x, 1 - x))

    return maf



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

        # Rows as SNPs:
        G = G.flatten()
        D = D.flatten()

    # Calculate accuracy
    correct = (G == D).sum()
    total = len(G)

    return correct / total if total > 0 else 0

def _pearson_r2_1d(G, D):
    """Squared Pearson correlation on two 1D arrays. Returns None when undefined."""
    if len(G) < 2:
        return None
    G_mean = G.mean()
    D_mean = D.mean()
    num = (np.sum((G - G_mean) * (D - D_mean))) ** 2
    den = np.sum((G - G_mean) ** 2) * np.sum((D - D_mean) ** 2)
    if den == 0:
        return None
    return float(num / den)


def calculate_r2_per_variant(orig_2d, imputed_2d, mask_2d=None, variants_only=False):
    """Per-variant Pearson r² (literature standard for imputation accuracy).

    For each variant (row of orig_2d, imputed_2d), compute Pearson r² across samples,
    optionally filtering samples via mask_2d (e.g., masked-only) and/or variants_only
    (drop sample positions where truth class == 1 i.e., 0|0 homo-ref). The function
    operates on the 4-class phased encoding {1,2,3,4} with shift to {0,1,2,3} internally.

    Returns mean of valid per-variant r²s (variants with undefined r² are excluded).
    """
    if orig_2d.ndim != 2:
        raise ValueError("expected 2D arrays")
    n_snps = orig_2d.shape[0]
    r2s = []
    for i in range(n_snps):
        g_row = np.asarray(orig_2d[i]).astype(float)
        d_row = np.asarray(imputed_2d[i]).astype(float)
        if mask_2d is not None:
            m_row = np.asarray(mask_2d[i]).astype(bool)
            g_row = g_row[m_row]
            d_row = d_row[m_row]
        if variants_only:
            vm = (g_row != 1)
            g_row = g_row[vm]
            d_row = d_row[vm]
        # shift to {0,1,2,3}
        g_row = g_row - 1
        d_row = d_row - 1
        r2 = _pearson_r2_1d(g_row, d_row)
        if r2 is not None:
            r2s.append(r2)
    if not r2s:
        return None
    return float(np.mean(r2s))




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

