import multiprocessing
import sys
from typing import Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from joblib import Parallel, delayed
from scanpy._utils import _check_use_raw
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix

sys.path.append("../..")

from src.utils.utils import check_signature_genes


def rank_calculation(cell_data, genes):
    """
    Compute the ranks of the gene expressions for a cell
    Args:
        cell_data: gene expression data for a given cell
        genes: signature genes

    Returns:
        average rank of signature genes for a given cell
    """
    subdata = cell_data[cell_data != 0]
    cell_data_ranked = subdata.rank(na_option='bottom')
    sig_data_ranked = cell_data_ranked[cell_data_ranked.index.isin(genes)]
    if len(sig_data_ranked) > 0:
        cumsum = sig_data_ranked.mean(skipna=True)
    else:
        cumsum = 0
    return cumsum / len(subdata)


def compute_avg_ranks_sig_subset(X_data, index, columns, gene_list, X_indices=None, X_indptr=None, X_shape=None):
    """
    Compute the average ranks for a given signature for each cell.
    Args:
        X_data: Gene expression data.
        index: Index of cells.
        columns: Names of genes.
        gene_list: Signature genes.
        X_indices: For sparse matrix reconstruction indices. If None, method assumes `X_data` to be a dense matrix.
        X_indptr: For sparse matrix reconstruction index pointers. If None, method assumes `X_data` to be a dense matrix.
        X_shape: For sparse matrix reconstruction shape of original matrix. If None, method assumes `X_data` to be a
            dense matrix.

    Returns:
        For each cell in X_data the method returns the average ranks.
    """
    if any([x is None for x in [X_indices, X_indptr, X_shape]]):
        data_df = pd.DataFrame(
            X_data, index=index, columns=columns
        )
    else:
        data_df = pd.DataFrame(
            csr_matrix((X_data, X_indices, X_indptr), X_shape, copy=False).todense(), index=index, columns=columns
        )

    return data_df.apply(func=(lambda x: rank_calculation(x, gene_list)), axis=1)


def compute_avg_ranks_signature(adata, sparse_X, gene_list, bs, joblib_kwargs):
    """
    Create groups of managable sizes. For each group compute for each cell the ranks of the genes and select the
    ranks that belong to the signature genes
    Args:
        adata: AnnData object with gene expression data.
        sparse_X: Indicates if data is sparse.
        gene_list: Signature genes.
        bs: The number of bins.
        joblib_kwargs: Keyword argument for parallel execution with joblib.

    Returns:
        For each cell in adata the method returns the average ranks
    """
    # create groups of managable sizes
    bss = pd.cut(np.arange(adata.obs.shape[0]), (adata.obs.shape[0] // bs + 1), labels=False)

    num_cores = multiprocessing.cpu_count()

    avg_sig_ranks = Parallel(**joblib_kwargs)(
        delayed(compute_avg_ranks_sig_subset)(
            X_data=adata[group[1].index,].X.data if sparse_X else adata[group[1].index,].X,
            X_indices=adata[group[1].index,].X.indices if sparse_X else None,
            X_indptr=adata[group[1].index,].X.indptr if sparse_X else None,
            X_shape=adata[group[1].index,].X.shape if sparse_X else None,
            index=group[1].index,
            columns=adata.var_names,
            gene_list=gene_list) for group in adata.obs.groupby(bss))
    avg_sig_ranks = pd.concat(avg_sig_ranks, axis=0)
    return avg_sig_ranks


def preparation(adata, genes):
    """
    Preparation for the computation of the expression value in JASMINE scoring.
    Args:
        adata: AnnData object with gene expression data.
        genes: Signature genes.

    Returns:
        The method returns the number of signature genes expressed, signature genes not expressed, non-signature genes
        expressed, and non-signature genes not expressed
    """
    sg_x = adata[:, genes].X
    nsg_x = adata[:, adata.var_names.isin(genes) == False].X

    nsg = list(set(adata.var_names).difference(set(genes)))

    if issparse(adata.X):
        ge = pd.DataFrame.sparse.from_spmatrix(sg_x, index=adata.obs_names, columns=genes)
        nge = pd.DataFrame.sparse.from_spmatrix(nsg_x, index=adata.obs_names, columns=nsg)
    else:
        ge = pd.DataFrame(sg_x, index=adata.obs_names, columns=genes)
        nge = pd.DataFrame(nsg_x, index=adata.obs_names, columns=nsg)

    sig_genes_exp = ge.astype(bool).sum(axis=1)
    n_sig_genes_exp = nge.astype(bool).sum(axis=1)

    sig_genes_ne = ge.shape[1] - sig_genes_exp
    sig_genes_ne = sig_genes_ne.replace(0, 1)

    n_sig_genes_exp = n_sig_genes_exp.replace(0, 1)

    n_sig_genes_ne = nge.shape[1] - (sig_genes_exp + n_sig_genes_exp)
    n_sig_genes_ne = n_sig_genes_ne - sig_genes_ne

    return sig_genes_exp, sig_genes_ne, n_sig_genes_exp, n_sig_genes_ne


def or_calculation(adata, genes):
    """
    Computation of enrichment value based on the Odds Ratio of the values returned in preparation
    Args:
        adata: AnnData object with gene expression data
        genes: Signature genes

    Returns:
        Enrichment score based on Odds Ratio
    """
    sig_genes_exp, sig_genes_ne, n_sig_genes_exp, n_sig_genes_ne = preparation(adata, genes)

    or_score = (sig_genes_exp * n_sig_genes_ne) / (sig_genes_ne * n_sig_genes_exp)

    return or_score


def likelihood_calculation(adata, genes):
    """
    Computation of enrichment value based on the Likelihood of the values returned in preparation
    Args:
        adata: AnnData object with gene expression data
        genes: Signature genes

    Returns:
        Enrichment score based on Likelihood
    """
    sig_genes_exp, sig_genes_ne, n_sig_genes_exp, n_sig_genes_ne = preparation(adata, genes)

    lr_one = sig_genes_exp * (n_sig_genes_exp + n_sig_genes_ne)
    lr_two = n_sig_genes_exp * (sig_genes_exp + sig_genes_ne)
    lr_score = lr_one / lr_two

    return lr_score


def score_genes(
        adata: AnnData,
        gene_list: List[str],
        score_method: str = 'likelihood',
        bs: int = 500,
        score_name: str = "JASMINE_score",
        random_state: Optional[int] = None,
        copy: bool = False,
        use_raw: Optional[bool] = None,
        verbose: int = 0,
        joblib_kwargs: dict = {'n_jobs': 4}
) -> Optional[AnnData]:
    """
    JASMINE signature scoring method is a Python implementation of the scoring method proposed by Noureen et al. 2022.

    Nighat Noureen, Zhenqing Ye, Yidong Chen, Xiaojing Wang, and Siyuan Zheng.
    „Signature-scoring methods developed for bulk samples are not adequate for cancer
    single-cell RNA sequencing data“. In: Elife 11 (Feb. 2022), e71994 (cit. on pp. iii, 2,
    9, 15–17).

    Args:
        adata: AnnData object containing the gene expression data.
        gene_list: A list of genes (signature) for which the cells are scored for.
        score_method: The method describes, which submethod of enrichment value computation should be used ('oddsratio',
            'likelihood').
        bs: The number of cells in a processing batch.
        score_name: Column name for scores added in `.obs` of data.
        random_state: Seed for random state.
        copy: Indicates whether original or a copy of `adata` is modified.
        use_raw: Whether to compute gene signature score on raw data stored in `.raw` attribute of `adata`
        verbose: If verbose is larger than 0, print statements are shown.
        joblib_kwargs: Keyword argument for parallel execution with joblib.

    Returns:
        If copy=True, the method returns a copy of the original data with stored JASMINE scores in `.obs`, otherwise
        None is returned.
    """
    start = sc.logging.info(f"computing score {score_name!r}")
    if verbose > 0:
        print(f"computing score {score_name!r}")

    adata = adata.copy() if copy else adata

    use_raw = _check_use_raw(adata, use_raw)

    _adata = adata.raw if use_raw else adata

    if random_state is not None:
        np.random.seed(random_state)

    # remove genes from gene_list not available in the data
    gene_list = check_signature_genes(_adata.var_names, gene_list)

    # check type of rank
    if score_method not in ['oddsratio', 'likelihood']:
        raise ValueError(f"method {score_method} must be one of the obptions ['oddsratio','likelihood']")
    elif score_method == 'oddsratio':
        f_score_method = or_calculation
    else:
        f_score_method = likelihood_calculation

    sparse_X = issparse(_adata.X)
    if sparse_X and not isspmatrix_csr(_adata.X):
        _adata.X = _adata.X.tocsr()

    avg_sig_ranks = compute_avg_ranks_signature(_adata, sparse_X, gene_list, bs, joblib_kwargs)
    scores = f_score_method(_adata, gene_list)
    # if not sparse_X:
    #    avg_sig_ranks = compute_avg_ranks_signature(_adata, sparse_X, gene_list, bs, joblib_kwargs)
    #    scores = f_score_method(_adata, gene_list)
    # elif sparse_X and isspmatrix_csc(_adata.X):
    #    scores = f_score_method(_adata, gene_list)
    #    _adata.X = _adata.X.tocsr()
    #    avg_sig_ranks = compute_avg_ranks_signature(_adata, sparse_X, gene_list, bs, joblib_kwargs)
    #    warnings.warn(f'Changed sparse format to CSR for performance reasons')
    # elif sparse_X and isspmatrix_csr(_adata.X):
    #    avg_sig_ranks = compute_avg_ranks_signature(_adata, sparse_X, gene_list, bs, joblib_kwargs)
    #    _adata.X = _adata.X.tocsc()
    #    scores = f_score_method(_adata, gene_list)
    #    warnings.warn(f'Changed sparse format to CSC for performance reasons')
    # else:
    #    raise ValueError('Unknown sparse matrix format. Allowd are CSR and CSC')

    avg_sig_ranks = (avg_sig_ranks - avg_sig_ranks.min()) / (avg_sig_ranks.max() - avg_sig_ranks.min())
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    score = (scores + avg_sig_ranks) / 2

    adata.obs[score_name] = score

    sc.logging.info(
        "    finished",
        time=start,
        deep=("added\n" f"    {score_name!r}, score of gene set (adata.obs)."),
    )
    return adata if copy else None
