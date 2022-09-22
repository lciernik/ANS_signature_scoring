import sys
from typing import Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame
import scanpy as sc
from anndata import AnnData
from scanpy._utils import _check_use_raw

sys.path.append("../..")

from src.scoring_methods.compute_signature_score import compute_signature_score
from src.utils.utils import (
    check_signature_genes,
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
        adjust_for_gene_std: bool = False,
        adjust_for_all_genes: bool = False,
        adjust_for_gene_std_var_1p: bool = False,
        adjust_for_gene_std_show_plots: bool = False,
        store_path_mean_var_data: Optional[str] = None,
        score_name: str = "ANS_score",
        random_state: Optional[int] = None,
        copy: bool = False,
        return_control_genes: bool = False,
        use_raw: Optional[bool] = None,
        verbose: int = 0,
) -> Optional[AnnData]:
    """
    Adjusted neighborhood gene signature scoring method (ANS) scores each cell in the dataset for a passed signature
     (gene_list) and stores the scores in the data object.

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
        use_raw: Whether to compute gene signature score on raw data stored in `.raw` attribute of `adata`
        verbose: If verbose is larger than 0, print statements are shown.

    Returns:
        If copy=True, the method returns a copy of the original data with stored ANS scores in `.obs`, otherwise None
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

    # compute average expression of genes and remove missing data
    if df_mean_var is None:
        df_mean_var = get_mean_and_variance_gene_expression(_adata_subset,
                                                            estim_var=adjust_for_gene_std,
                                                            show_plots=adjust_for_gene_std_show_plots,
                                                            store_path=store_path_mean_var_data,
                                                            store_data_prefix=score_name)
    if len(set(df_mean_var.index).difference(set(gene_pool))) > 0:
        df_mean_var = df_mean_var.loc[gene_pool, :]

    gene_means = df_mean_var['mean'].copy()

    # remove signature genes belonging to the ctrl_size/2 genes with the highest average expressions
    if remove_genes_with_invalid_control_set:
        gene_list = [x for x in gene_list if
                     (np.where(gene_means.index == x)[0]) < (gene_means.shape[0] - ctrl_size // 2)]

    # computation of neighboring genes around each signature gene
    sorted_gene_means = gene_means.sort_values()
    ref_genes_means = sorted_gene_means[sorted_gene_means.index.isin(gene_list) == False]

    # use sliding window to compute for each window the mean
    rolled = ref_genes_means.rolling(ctrl_size, closed='right').mean()

    control_genes = []
    for sig_gene in gene_list:
        curr_sig_avg = sorted_gene_means.loc[sig_gene]
        min_val_idx = np.argmin(((rolled - curr_sig_avg).abs()))
        sig_gene_ctrl_genes = rolled.iloc[(min_val_idx - ctrl_size + 1):min_val_idx + 1]
        control_genes.append(list(sig_gene_ctrl_genes.index))

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
            f"    {len(control_genes) * ctrl_size} total control genes are used."
        ),
    )
    if return_control_genes:
        return adata, control_genes if copy else control_genes
    else:
        return adata if copy else None
