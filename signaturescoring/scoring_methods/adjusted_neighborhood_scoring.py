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
    checks_ctrl_size,
    get_data_for_gene_pool,
    get_mean_and_variance_gene_expression
)


def score_genes(
        adata: AnnData,
        gene_list: List[str],
        ctrl_size: int = 100,
        gene_pool: Optional[List[str]] = None,
        df_mean_var: Optional[DataFrame] = None,
        remove_genes_with_invalid_control_set: bool = True,
        store_path_mean_var_data: Optional[str] = None,
        score_name: str = "ANS_score",
        copy: bool = False,
        return_control_genes: bool = False,
        return_gene_list: bool = False,
        use_raw: Optional[bool] = None,
) -> Optional[AnnData]:
    """
    Adjusted neighborhood gene signature scoring method (ANS) scores each cell in the dataset for a passed signature
    (gene_list) and stores the scores in the data object.
    Implementation is inspired by score_genes method of Scanpy
    (https://scanpy.readthedocs.io/en/latest/generated/scanpy.tl.score_genes.html#scanpy.tl.score_genes)

    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        gene_list: A list of genes for which the cells are scored for.
        ctrl_size: The number of control genes selected for each gene in the gene_list.
        gene_pool: The pool of genes out of which control genes can be selected.
        df_mean_var: A pandas DataFrame containing the average expression (and variance) for each gene in the dataset.
            If df_mean_var is None, the average gene expression and variance is computed during gene signature scoring.
        remove_genes_with_invalid_control_set: If true, the scoring method removes genes from the gene_list for which
            no optimal control set can be computed, i.e., if a gene belongs to the ctrl_size/2 genes with the largest
            average expression.
        store_path_mean_var_data: Path to store data and visualizations created during average gene expression
            computation. If it is None, data and visualizations are not stored.
        score_name: Column name for scores added in `.obs` of data.
        copy: Indicates whether original or a copy of `adata` is modified.
        return_control_genes: Indicates if method returns selected control genes.
        return_gene_list: Indicates if method returns the possibly reduced gene list.
        use_raw: Whether to compute gene signature score on raw data stored in `.raw` attribute of `adata`

    Returns:
        If copy=True, the method returns a copy of the original data with stored ANS scores in `.obs`, otherwise None
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
    _adata_subset, gene_pool = get_data_for_gene_pool(_adata, gene_pool, gene_list, check_gene_list=False)

    # checks on ctrl_size, i.e., number of control genes
    checks_ctrl_size(ctrl_size, len(gene_pool), len(gene_list))

    # compute average expression of genes and remove missing data
    if df_mean_var is None:
        df_mean_var = get_mean_and_variance_gene_expression(_adata_subset,
                                                            store_path=store_path_mean_var_data,
                                                            store_data_prefix=score_name)
    if len(set(df_mean_var.index).difference(set(gene_pool))) > 0:
        df_mean_var = df_mean_var.loc[gene_pool, :]

    gene_means = df_mean_var['mean'].copy()

    # computation of neighboring genes around each signature gene
    sorted_gene_means = gene_means.sort_values()
    ref_genes_means = sorted_gene_means[sorted_gene_means.index.isin(gene_list) == False]

    # use sliding window to compute for each window the mean
    rolled = ref_genes_means.rolling(ctrl_size, closed='right').mean()

    # remove signature genes belonging to the ctrl_size/2 genes with the highest average expressions
    if remove_genes_with_invalid_control_set:
        gene_list = [x for x in gene_list if
                     (np.where(sorted_gene_means.index == x)[0]) < (sorted_gene_means.shape[0] - ctrl_size // 2)]
        if len(gene_list) == 0:
            raise ValueError(f'After removing signature genes for which no valid control was found, no signature '
                             f'genes are remaining, i.e., empty signature. Control your signature and control size.')

    # compute for each signature gene its control set
    control_genes = []
    for sig_gene in gene_list:
        curr_sig_avg = sorted_gene_means.loc[sig_gene]
        min_val_idx = np.argmin(((rolled - curr_sig_avg).abs()))
        sig_gene_ctrl_genes = rolled.iloc[(min_val_idx - ctrl_size + 1):min_val_idx + 1]
        control_genes.append(list(sig_gene_ctrl_genes.index))

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
            f"    {len(control_genes) * ctrl_size} total control genes are used."
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
