from typing import Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from pandas import DataFrame
from scanpy._utils import _check_use_raw

from signaturescoring.scoring_methods.compute_signature_score import compute_signature_score
from signaturescoring.utils.utils import (
    check_signature_genes,
    get_bins_wrt_avg_gene_expression,
    get_data_for_gene_pool,
    get_least_variable_genes_per_bin_v1,
    get_least_variable_genes_per_bin_v2,
    get_mean_and_variance_gene_expression,
)


def score_genes(
        adata: AnnData,
        gene_list: List[str],
        ctrl_size: int = 100,
        n_bins: int = 25,
        gene_pool: Optional[List[str]] = None,
        lvg_computation_version: str = "v1",
        lvg_computation_method: str = "seurat",
        nr_norm_bins: int = 5,
        df_mean_var: Optional[DataFrame] = None,
        adjust_for_gene_std: bool = False,
        adjust_for_all_genes: bool = False,
        adjust_for_gene_std_var_1p: bool = False,
        adjust_for_gene_std_show_plots: bool = False,
        store_path_mean_var_data: Optional[str] = None,
        score_name: str = "LVCG_score",
        random_state: Optional[int] = None,
        copy: bool = False,
        return_control_genes: bool = False,
        return_gene_list: bool = False,
        use_raw: Optional[bool] = None,
        verbose: int = 0,
) -> Optional[AnnData]:
    """
    Least variable genes as control genes scoring method (LVCGS) scores each cell in the dataset for a passed signature
    (gene_list) and stores the scores in the data object.
    Implementation is inspired by score_genes method of Scanpy
    (https://scanpy.readthedocs.io/en/latest/generated/scanpy.tl.score_genes.html#scanpy.tl.score_genes)

    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        gene_list: A list of genes for which the cells are scored for.
        ctrl_size: The number of control genes selected for each gene in the gene_list.
        n_bins: The number of average gene expression bins to use.
        gene_pool: The pool of genes out of which control genes can be selected.
        lvg_computation_version: The version of the least variable genes selection defines if the genes with the
            smallest dispersion are chosen directly from an expression bin (v1) or whether the expressions are binned
            a second round (v2).
        lvg_computation_method: Indicates which method should be used to compute the least variable genes. We can use
            'seurat' or 'cell_ranger'. See reference
            https://scanpy.readthedocs.io/en/latest/generated/scanpy.pp.highly_variable_genes.html#scanpy-pp-highly-variable-genes
        nr_norm_bins: If `lvg_computation_version="v2"`, we need to define the number of subbins used.
        df_mean_var: A pandas DataFrame containing the average expression (and variance) for each gene in the dataset.
            If df_mean_var is None, the average gene expression and variance is computed during gene signature scoring.
        adjust_for_gene_std: Apply gene signature scoring with standard deviation adjustment. Divide the difference
            between a signature gene's expression and the mean expression of the control genes by the estimated
            standard deviation of the signature gene.
        adjust_for_all_genes: Apply gene signature scoring with standard deviation adjustment for each occurring gene.
            Divide each gene's expression (signature and control genes) by estimated standard deviation of the gene.
        adjust_for_gene_std_var_1p: Apply gene signature scoring with standard deviation adjustment. Divide the
            difference between a signature gene's expression and the mean expression of the control genes by the
            estimated standard deviation + 1  of the signature gene.
        adjust_for_gene_std_show_plots: Indicates whether plots should be shown during average expression computation.
        store_path_mean_var_data: Path to store data and visualizations created during average gene expression
            computation. If it is None, data and visualizations are not stored.
        score_name: Column name for scores added in `.obs` of data.
        random_state: Seed for random state
        copy: Indicates whether original or a copy of `adata` is modified.
        return_control_genes: Indicated if method returns selected control genes.
        return_gene_list: Indicates if method returns the possibly reduced gene list.
        use_raw: Whether to compute gene signature score on raw data stored in `.raw` attribute of `adata`
        verbose: If verbose is larger than 0, print statements are shown.

    Returns:
        If copy=True, the method returns a copy of the original data with stored LVCG scores in `.obs`, otherwise None
        is returned.
    """
    start = sc.logging.info(f"computing score {score_name!r}")
    if verbose > 0:
        print(f"computing score {score_name!r}")

    # set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # copy original data if copy=True
    adata = adata.copy() if copy else adata
    # work on raw data if desired
    use_raw = _check_use_raw(adata, use_raw)
    _adata = adata.raw if use_raw else adata

    # remove genes from gene_list not available in the data
    var_names = _adata.var_names.tolist()
    gene_list = check_signature_genes(var_names, gene_list)

    # get data for gene pool
    _adata_subset, gene_pool = get_data_for_gene_pool(_adata, gene_pool, gene_list)

    # bin according to average gene expression on the gene_pool
    if df_mean_var is None:
        df_mean_var = get_mean_and_variance_gene_expression(_adata_subset,
                                                            estim_var=adjust_for_gene_std,
                                                            show_plots=adjust_for_gene_std_show_plots,
                                                            store_path=store_path_mean_var_data,
                                                            store_data_prefix=score_name)

    if len(set(df_mean_var.index).difference(set(gene_pool))) > 0:
        df_mean_var = df_mean_var.loc[gene_pool, :]

    gene_bins = get_bins_wrt_avg_gene_expression(df_mean_var['mean'], n_bins)

    # get for each bin the ctrl_size number of the least variable genes
    ref_genes = list(set(gene_pool).difference(set(gene_list)))
    lvg_computation_method = lvg_computation_method.lower()
    if lvg_computation_method not in ["seurat", "cell_ranger"]:
        raise ValueError(
            f"Unknown method {lvg_computation_method} for computation of the least variable genes per bin"
        )

    if lvg_computation_version == "v1":
        least_variable_genes_per_bin = get_least_variable_genes_per_bin_v1(
            _adata_subset,
            gene_bins,
            ctrl_size,
            method=lvg_computation_method,
            gene_pool=ref_genes,
        )
    elif lvg_computation_version == "v2":
        least_variable_genes_per_bin = get_least_variable_genes_per_bin_v2(
            _adata_subset,
            gene_bins,
            ctrl_size,
            method=lvg_computation_method,
            gene_pool=ref_genes,
            nr_norm_bins=nr_norm_bins,
        )
    else:
        raise ValueError(
            f"Unknown version {lvg_computation_version} for computation of the least variable genes per bin"
        )
    if verbose > 0:
        print(f"least_variable_genes_per_bin {least_variable_genes_per_bin}")

    # average expression of all control get all control genes.
    signature_bins = gene_bins.loc[gene_list]
    nr_control_genes = 0

    # compute set of control genes
    control_genes = []
    for curr_bin in signature_bins:
        r_genes = least_variable_genes_per_bin[curr_bin]
        nr_control_genes += len(r_genes)
        control_genes.append(r_genes)

    if adjust_for_gene_std:
        if adjust_for_gene_std_var_1p:
            estim_std = df_mean_var['estimate_std_1p']
        else:
            estim_std = df_mean_var['estimate_std']
        score = compute_signature_score(_adata_subset, gene_list, control_genes,
                                        estim_std, adjust_for_all_genes)
    else:
        score = compute_signature_score(_adata_subset, gene_list, control_genes)

    adata.obs[score_name] = pd.Series(
        np.array(score).ravel(), index=adata.obs_names, dtype="float64"
    )

    sc.logging.info(
        "    finished",
        time=start,
        deep=(
            "added\n"
            f"    {score_name!r}, score of gene set (adata.obs).\n"
            f"    {nr_control_genes} total control genes are used."
        ),
    )
    if return_control_genes and return_gene_list:
        return (adata, control_genes, gene_list) if copy else (control_genes, gene_list)
    elif return_control_genes:
        return (adata, control_genes) if copy else control_genes
    elif return_gene_list:
        return (adata, gene_list) if copy else gene_list
    else:
        return adata if copy else None
