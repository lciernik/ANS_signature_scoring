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
    get_data_for_gene_pool,
    get_bins_wrt_avg_gene_expression,
    get_mean_and_variance_gene_expression,
)


def score_genes(
        adata: AnnData,
        gene_list: List[str],
        n_bins: int = 25,
        gene_pool: Optional[List[str]] = None,
        df_mean_var: Optional[DataFrame] = None,
        store_path_mean_var_data: Optional[str] = None,
        score_name: str = "Seurat_AG_score",
        copy: bool = False,
        return_control_genes: bool = False,
        return_gene_list: bool = False,
        use_raw: Optional[bool] = None,
) -> Optional[AnnData]:
    """
    All genes as control genes scoring method (Seurat_AG) scores each cell in the dataset for a passed signature
    (gene_list) and stores the scores in the data object.
    Implementation is inspired by score_genes method of Scanpy
    (https://scanpy.readthedocs.io/en/latest/generated/scanpy.tl.score_genes.html#scanpy.tl.score_genes)

    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        gene_list: A list of genes for which the cells are scored for.
        n_bins: The number of average gene expression bins to use.
        gene_pool: The pool of genes out of which control genes can be selected.
        df_mean_var: A pandas DataFrame containing the average expression (and variance) for each gene in the dataset.
            If df_mean_var is None, the average gene expression and variance is computed during gene signature scoring
        store_path_mean_var_data: Path to store data and visualizations created during average gene expression
            computation. If it is None, data and visualizations are not stored.
        score_name: Column name for scores added in `.obs` of data.
        copy: Indicates whether original or a copy of `adata` is modified.
        return_control_genes: Indicated if method returns selected control genes.
        return_gene_list: Indicates if method returns the possibly reduced gene list.
        use_raw: Whether to compute gene signature score on raw data stored in `.raw` attribute of `adata`.

    Returns:
        If copy=True, the method returns a copy of the original data with stored AGCGS scores in `.obs`, otherwise None
        is returned.
    """
    start = sc.logging.info(f"computing score {score_name!r}")

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

    # compute average expression of genes and remove missing data
    if df_mean_var is None:
        df_mean_var = get_mean_and_variance_gene_expression(_adata_subset,
                                                            store_path=store_path_mean_var_data,
                                                            store_data_prefix=score_name)
    if len(set(df_mean_var.index).difference(set(gene_pool))) > 0:
        df_mean_var = df_mean_var.loc[gene_pool, :]

    # bin according to average gene expression on the gene_pool
    gene_bins = get_bins_wrt_avg_gene_expression(df_mean_var['mean'], n_bins)

    # get the bin ids of the genes in gene_list.
    signature_bins = gene_bins.loc[gene_list]

    # compute set of control genes
    nr_control_genes = 0
    control_genes = []
    for curr_bin in signature_bins:
        r_genes = np.array(gene_bins[gene_bins == curr_bin].index)
        r_genes = list(set(r_genes).difference(set(gene_list)))
        nr_control_genes += len(r_genes)
        control_genes.append(r_genes)

    # compute final scores
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
