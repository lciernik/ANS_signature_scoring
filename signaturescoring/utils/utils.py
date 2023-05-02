import logging
import os
import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt
from pandas import DataFrame
from scanpy.preprocessing._utils import _get_mean_var
from scanpy.tools._score_genes import _sparse_nanmean
from scipy.sparse import issparse
from sklearn.metrics import r2_score
from skmisc.loess import loess


def check_signature_genes(var_names: List[str], gene_list: List[str], return_type: Any = list):
    """
    The method checks the availability of signature genes in the list of available genes (var_names). Genes not present
    in var_names are expluded.
    Args:
        var_names: List of available genes in the dataset.
        gene_list: List of genes to score for (signature).
        return_type: Indicates whether to return a list or a set.

    Returns:
        Filtered gene list.

    Raises:
        ValueError
            if return_type is not of type list or set.
            if gene_list gets empty.
    """
    # assume that if single string is passed it represents a single gene that should be scored
    if isinstance(gene_list, str):
        gene_list = [gene_list]

    if type(gene_list) not in [list, set]:
        raise ValueError(f"gene_list needs to be either list or set.")

    if len(gene_list) == 0:
        raise ValueError(f'gene_list must contain at least 1 gene name in adata.var_names')

    if len(gene_list) != len(set(gene_list)):
        seen = set()
        duplicates = [x for x in gene_list if x in seen or seen.add(x)]
        warnings.warn(f'The passed gene_list contains duplicated genes: {duplicates}')

    if return_type not in [list, set]:
        raise ValueError(f"return_type needs to be either list or set.")

    if not isinstance(var_names, list):
        var_names = var_names.tolist()

    genes_to_ignore = (set(gene_list)).difference(set(var_names))
    gene_list = (set(var_names)).intersection(set(gene_list))

    if len(genes_to_ignore) > 0:
        sc.logging.warning(f"genes are not in var_names and ignored: {genes_to_ignore}")
    if len(gene_list) == 0:
        raise ValueError("No valid genes were passed for scoring.")

    return return_type(gene_list)


def checks_ctrl_size(ctrl_size: int, gene_pool_size: int, gene_list_size: int):
    """
    Applies input checks on the control set size and if valid control sets can be built with the desired size.
    Args:
        ctrl_size: The number of control genes selected for each gene in the gene_list.
        gene_pool_size: The number of genes in the allowed control genes pool.
        gene_list_size: The number of genes in of a gene expression signature.

    Raise:
        Value Error if checks on ctrl size fail.
    """
    if isinstance(ctrl_size, float):
        ctrl_size = int(np.round(ctrl_size))
    if not isinstance(ctrl_size, int) or ctrl_size < 1:
        raise ValueError(f'ctrl_size needs to be a positive integer larger than 0.')

    if (gene_pool_size - gene_list_size) < ctrl_size:
        raise ValueError(f'Not enough genes in gene_pool (len(gene_pool) - len(gene_list) < ctrl_size) to compute '
                         f'scoring control sets. Decrease ctrl_size and/or siganture length and/or gene pool.')


def get_bins_wrt_avg_gene_expression(gene_means: Any, n_bins: int, verbose: int = 0):
    """
    Method to compute the average gene expression bins.
    Args:
        gene_means: Average gene expression vector.
        n_bins: Number of desired bins.
        verbose: Show print statements if larger than 0.

    Returns:
        Series containing gene to expression bin assignment.
    """
    # correct computations of cuts
    ranked_gene_means = gene_means.rank(method="min")
    gene_bins = pd.cut(ranked_gene_means, n_bins, labels=False)
    if verbose > 0:
        print(f"Got {len(np.unique(gene_bins))} bins.")
    return gene_bins


def get_data_for_gene_pool(adata: AnnData, gene_pool: List[str], gene_list: List[str], ctrl_size: Optional[int] = None,
                           check_gene_list: bool = True):
    """
    The method to filter dataset for gene pool and genes in gene_list.
    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        gene_pool: List of genes from which the control genes can be selected.
        gene_list: List of genes (signature) scoring methods want to score for.
        check_gene_list: Indicates whether gene list should be checked.
    Returns:
        Eventually, filtered adata subset and new gene_pool.
    """
    var_names = list(adata.var_names)
    if gene_pool is not None and type(gene_pool) not in [list, set]:
        raise ValueError(f'gene_pool needs to be a list or set of Gene names (i.e., strings)')

    if gene_pool is not None and len(set(gene_pool).difference(set(var_names))) > 0:
        warnings.warn(f'Passed gene_pool contains genes not available in adata.var_names. The following genes are '
                      f'ignored: {set(gene_pool).difference(set(var_names))}')
    gene_pool = (
        var_names if (gene_pool is None) else [x for x in gene_pool if x in var_names]
    )

    if not gene_pool:
        raise ValueError("No valid genes were passed for reference set.")

    if check_gene_list:
        gene_list = check_signature_genes(var_names, gene_list)

    if ctrl_size is not None and isinstance(ctrl_size, int) and (len(gene_pool) - len(gene_list)) < ctrl_size:
        raise ValueError(f'Not enough genes in gene_pool (len(gene_pool) - len(gene_list) < ctrl_size) to compute '
                         f'scoring control sets. Decrease ctrl_size and/or signature length and/or gene pool.')

    gene_pool = list(set(gene_list).union(set(gene_pool)))

    # need to include gene_list to
    if len(gene_pool) < len(var_names):
        return adata[:, gene_pool], gene_pool
    else:
        return adata, gene_pool


def get_gene_list_real_data(
        adata: AnnData,
        dge_method: str = "wilcoxon",
        dge_key: str = "wilcoxon",
        dge_pval_cutoff: float = 0.01,
        dge_log2fc_min: float = 0,
        nr_de_genes: int = 100,
        mode: str = "most_diff_expressed",
        label_col: str = 'healthy',
        label_of_interest: str = 'unhealthy',
        random_state: Optional[int] = 0,
        log: Optional[str] = None,
        copy: bool = False,
        verbose: int = 0,
) -> List[str]:
    """
    This function returns the signature genes for a given dataset. It first gets all differentially expressed genes for
    a group of interest and then selects a signature based on the defined mode.
    mode).
    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        dge_method: Method for DGEX in Scanpy. Available methods: 'logreg', 't-test', 'wilcoxon', 't-test_overestim_var'
        dge_key: Name of key that is added in '.uns'
        dge_pval_cutoff: Cutoff value of adjusted p-value, i.e., max adjusted p-value
        dge_log2fc_min: Cutoff minimum value of log fold change.
        nr_de_genes: Number of genes in signature.
        mode: Select most highly, least or random differentially expressed genes.
        label_col: Name of column containing cell type labels for DGEX.
        label_of_interest: Name of class we want to get the signature for.
        random_state: Seed for random state.
        log: Name of logger.
        copy: Whether to do the DGEX computation inplace or on a copy of adata.
        verbose: Show print statements if verbose larger than 0.

    Returns:
        Gene expression signature, i.e., list of genes.

    """
    adata = adata.copy() if copy else adata

    if mode not in ["most_diff_expressed", "least_diff_expressed", "random"]:
        raise ValueError(
            f"Unknown mode {mode}. You can choose between ['most_diff_expressed', 'least_diff_expressed','random']"
        )

    # logging
    if log is None:
        log = logging
    else:
        log = logging.getLogger(log)

    if label_col not in adata.obs:
        raise ValueError(
            f"data does not contain any labels in a column called healthy."
        )

    sc.tl.rank_genes_groups(
        adata, label_col, method=dge_method, key_added=dge_key, tie_correct=True
    )
    wc = sc.get.rank_genes_groups_df(
        adata,
        group=label_of_interest,
        key=dge_key,
        pval_cutoff=dge_pval_cutoff,
        log2fc_min=dge_log2fc_min,
    )
    if mode == "most_diff_expressed":
        diffexp_genes = wc.nlargest(nr_de_genes, columns="logfoldchanges")
    elif mode == "least_diff_expressed":
        diffexp_genes = wc.nsmallest(nr_de_genes, columns="logfoldchanges")
    else:
        diffexp_genes = wc.sample(nr_de_genes, random_state=random_state)

    if verbose > 0:
        diffexp_genes["logfoldchanges"].hist(bins=50, figsize=(8, 5))
        plt.xlabel("logfoldchanges", fontsize=10)
        plt.title("logfoldchanges for signature genes", fontsize=12)
        plt.show()

    diffexp_genes = diffexp_genes["names"].tolist()

    return list(diffexp_genes)


def get_least_variable_genes_per_bin_v1(
        adata: AnnData, cuts: Any, ctrl_size: int, method: str = "seurat", gene_pool: Optional[List[str]] = None
) -> dict:
    """
    This method implements v1 of the least variable control genes selection for a given dataset. The method uses
    provided expression bins to select from each bin the genes with the smallest dispersion.
    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        cuts: Assignment of genes to expression bins.
        ctrl_size: The number of control genes selected for each expression bin.
        method: Indicates which method should be used to compute the least variable genes. We can use
            'seurat' or 'cell_ranger'. See reference
            https://scanpy.readthedocs.io/en/latest/generated/scanpy.pp.highly_variable_genes.html#scanpy-pp-highly-variable-genes
        gene_pool: The pool of genes out of which control genes can be selected.

    Returns:
        A dictionary mapping to each expression bin (i.e., distinct values in cuts) a set of genes with the least
        variation.
    """
    gene_pool = (
        adata.var_names
        if (gene_pool is None)
        else [x for x in gene_pool if x in adata.var_names]
    )
    if len(gene_pool) < len(adata.var_names):
        adata_subset = adata[:, gene_pool].copy()
        cuts_subset = cuts[gene_pool].copy()
    else:
        adata_subset = adata
        cuts_subset = cuts

    # get with method indicated dispersion values
    if method == "seurat":
        sc.pp.highly_variable_genes(adata_subset, flavor="seurat")
    elif method == "cell_ranger":
        sc.pp.highly_variable_genes(adata_subset, flavor="cell_ranger")
    else:
        raise ValueError(
            f"Unknown method {method} to compute the least variables genes per bin."
        )

    lvg_per_bin = {}
    for cut in np.unique(cuts_subset):
        r_genes_current_bin = (
            adata_subset.var["dispersions"][cuts_subset == cut]
            .nsmallest(ctrl_size)
            .index
        )
        r_genes_current_bin = r_genes_current_bin.tolist()
        lvg_per_bin[cut] = r_genes_current_bin

    return lvg_per_bin


def get_least_variable_genes_per_bin_v2(
        adata: AnnData, cuts: Any, ctrl_size: int, method: str = "seurat", gene_pool: Optional[List[str]] = None,
        nr_norm_bins: int = 5
) -> dict:
    """
    This method implements v2 of the least variable control genes selection for a given dataset. This method computes for
    each of the expression bins the least variable genes, during which the average expression of the expression bins are
    binned a second time and normalized dispersion scores are computed. The method then selects for each expression bin
    the genes with smallest normalized dispersion.
    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        cuts: Assignment of genes to expression bins.
        ctrl_size: The number of control genes selected for each expression bin.
        method: Indicates which method should be used to compute the least variable genes. We can use
            'seurat' or 'cell_ranger'. See reference
            https://scanpy.readthedocs.io/en/latest/generated/scanpy.pp.highly_variable_genes.html#scanpy-pp-highly-variable-genes
        gene_pool: The pool of genes out of which control genes can be selected.
        nr_norm_bins: The number of bins required for the highly variable genes computation for each expression bin.

    Returns:
        A dictionary mapping to each expression bin (i.e., distinct values in cuts) a set of genes with least variation.
    """
    gene_pool = (
        adata.var_names
        if (gene_pool is None)
        else [x for x in gene_pool if x in adata.var_names]
    )
    if len(gene_pool) < len(adata.var_names):
        adata_subset = adata[:, gene_pool].copy()
        cuts_subset = cuts[gene_pool].copy()
    else:
        adata_subset = adata
        cuts_subset = cuts

    method = method.lower()
    if method not in ["seurat", "cell_ranger"]:
        raise ValueError(
            f"Unknown method {method} to compute the least variables genes per bin."
        )

    lvg_per_bin = {}
    for cut in np.unique(cuts_subset):
        curr_bin_data = adata_subset[:, cuts_subset == cut].copy()
        sc.pp.highly_variable_genes(curr_bin_data, flavor=method, n_bins=nr_norm_bins)
        r_genes_current_bin = (
            curr_bin_data.var["dispersions_norm"].nsmallest(ctrl_size).index
        )
        r_genes_current_bin = r_genes_current_bin.tolist()
        lvg_per_bin[cut] = r_genes_current_bin

    return lvg_per_bin


def get_mean_and_variance_gene_expression(adata: AnnData,
                                          estim_var: bool = False,
                                          loess_span: float = 0.3,
                                          loess_degree: int = 2,
                                          show_plots: bool = False,
                                          store_path: Optional[str] = None,
                                          store_data_prefix: str = '') -> DataFrame:
    """
    This function computes for the passed data the average gene expression and the variance of genes. Additionally, one
    can compute the estimated variance and standard deviation by regression the mean out. The estimation of the variance
    is computed by fitting a loess curve on the log10 mean and log10 variance.
    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        estim_var: Indicates whether to compute the estimated variance or not
        loess_span: Span parameter of loess see https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess.html
        loess_degree: Span parameter of loess see https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess.html
        show_plots: Indicates whether to show the plot or not.
        store_path: Path where to store the computed data and the plots. If the path does not exist, no data or plots
            are stored.
        store_data_prefix: Indicates a prefix to the data/ plot file names.

    Returns:
        Dataframe containing the average gene expression and the variance of each gene. If estim_var=True, it
        additionally contains the estimated variance and standard variation of each gene and the loess r2-score
        (in data.attrs).
    """

    store_data = True
    if store_path is not None:
        if not os.path.exists(store_path):
            store_data = False
            warnings.warn(f'The passed store path {store_path} does not exists. Data won\'t be stored.')
    else:
        store_data = False
        #print('No store_path indicated, thus no data stored.')

    X = adata.X
    df = pd.DataFrame()
    # compute mean and variance
    df['mean'], df['var'] = _get_mean_var(X)
    df = df.set_index(adata.var_names)
    df = df.sort_values(by='mean')

    # can compute the estimated variance based on the mean by regression.
    if estim_var:
        not_const = df['var'] > 0
        estimate_var = np.zeros(X.shape[1], dtype=np.float64)

        y = np.log10(df['var'][not_const])
        x = np.log10(df['mean'][not_const])
        model = loess(x, y, span=loess_span, degree=loess_degree)
        model.fit()
        estimate_var[not_const] = model.outputs.fitted_values

        df['estimate_var'] = 10 ** estimate_var
        df['estimate_var_1p'] = (10 ** estimate_var) + 1

        df['estimate_std'] = np.sqrt(df['estimate_var'])
        df['estimate_std_1p'] = np.sqrt(df['estimate_var_1p'])

        loess_r2_score = r2_score(y, model.outputs.fitted_values)
        df.attrs['loess_r2_score'] = loess_r2_score

        # Create plot
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(x, y, label='log10(mean) vs. log10(var)')
        plt.scatter(x, estimate_var[not_const], marker='.', label='log10(mean) vs. log10(estimated var)')
        plt.legend(fontsize=14)
        plt.title(f'Relationship mean vs. var/ estimated var, R2-score:{str(np.round(loess_r2_score, decimals=3))}',
                  fontsize=16)
        if store_data:
            new_fn = os.path.join(store_path, (store_data_prefix + '_logmean_vs_logvar.png'))
            new_fn = nextnonexistent(new_fn)
            fig.savefig(new_fn, format='png')
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    if store_data:
        new_fn = os.path.join(store_path, (store_data_prefix + '_mean_var_estim_var_data.pkl'))
        new_fn = nextnonexistent(new_fn)
        df.to_pickle(new_fn)

    return df


def nanmean(x: Any, axis: int, dtype=None):
    """
    Sparse equivalent to numpy.nanmean using the sparse nanmean implementation of Scanpy
    https://github.com/scverse/scanpy/blob/034ca2823804645e0d4874c9b16ba2eb8c13ac0f/scanpy/tools/_score_genes.py
    Args:
        x: Data matrix.
        axis: Axis along which to compute mean (0: column-wise, 1: row-wise).
        dtype: Desired type of the mean vector.

    Returns:
        Mean vector of desired dimension

    """
    if issparse(x):
        x_mean = np.array(_sparse_nanmean(x, axis=axis)).flatten()
    else:
        if dtype:
            x_mean = np.nanmean(x, axis=axis, dtype=dtype)
        else:
            x_mean = np.nanmean(x, axis=axis)

    return x_mean


def nextnonexistent(f: str) -> str:
    """
    Method to get next filename if original filename does already exist.
    Args:
        f: Filename.

    Returns:
        New unique filename.
    """
    fnew = f
    root, ext = os.path.splitext(f)
    i = 0
    while os.path.exists(fnew):
        i += 1
        fnew = '%s_%i%s' % (root, i, ext)
    return fnew


# Following two methods are contributed by Rachit Belwariar.
# https://www.geeksforgeeks.org/longest-common-prefix-using-divide-and-conquer-algorithm/?ref=lbp
# A Python3 Program to find the longest common prefix

# A Utility Function to find the common
# prefix between strings- str1 and str2
def commonPrefixUtil(str1, str2):
    result = ""
    n1, n2 = len(str1), len(str2)
    i, j = 0, 0

    while i <= n1 - 1 and j <= n2 - 1:

        if str1[i] != str2[j]:
            break
        result += str1[i]
        i, j = i + 1, j + 1

    return result


# A Divide and Conquer based function to
# find the longest common prefix. This is
# similar to the merge sort technique
def commonPrefix(arr, low, high):
    if low == high:
        return arr[low]

    if high > low:
        # Same as (low + high)/2, but avoids
        # overflow for large low and high
        mid = low + (high - low) // 2

        str1 = commonPrefix(arr, low, mid)
        str2 = commonPrefix(arr, mid + 1, high)

        return commonPrefixUtil(str1, str2)
