import warnings
from typing import Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from joblib import Parallel, delayed
from scanpy._utils import _check_use_raw
from scipy.sparse import issparse, csr_matrix, isspmatrix_csr
from scipy.stats import rankdata

from signaturescoring.utils.utils import check_signature_genes


def u_stat(rank_value, maxRank=1500):
    """
    The method computes the U statistic on signature gene ranks.
    Args:
        rank_value: Ranks of the signature genes.
        maxRank: Cutoff for maximum rank allowed.

    Returns:
        The U statistic for given signature gene ranks.
    """
    insig = rank_value > maxRank
    if all(insig):
        return 0
    else:
        rank_value[insig] = maxRank + 1
        rank_sum = rank_value.sum()
        len_sig = len(rank_value)
        u_value = rank_sum - (len_sig * (len_sig + 1)) / 2
        auc = 1 - u_value / (len_sig * maxRank)
        return auc


def compute_ranks_and_ustat(X_data, index, columns, gene_list, X_indices=None, X_indptr=None, X_shape=None,
                            maxRank=1500):
    """
    The following method computes for each cell in `X_data` the UCell score.
    Args:
        X_data: Current batch of gene expression data.
        index: Index of cells.
        columns: Names of genes.
        gene_list: Signature genes.
        X_indices: For sparse matrix reconstruction indices. If None, method assumes `X_data` to be a dense matrix.
        X_indptr: For sparse matrix reconstruction index pointers. If None, method assumes `X_data` to be a dense matrix.
        X_shape: For sparse matrix reconstruction shape of original matrix. If None, method assumes `X_data` to be
            a dense matrix.
        maxRank:  Cutoff for maximum rank allowed.

    Returns:
        For each cell in X_data the method returns the UCell score.
    """
    if any([x is None for x in [X_indices, X_indptr, X_shape]]):
        data_df = pd.DataFrame(
            X_data, index=index, columns=columns
        )
    else:
        data_df = pd.DataFrame(
            csr_matrix((X_data, X_indices, X_indptr), X_shape, copy=False).todense(), index=index, columns=columns
        )

    res = (data_df.apply(
        lambda x: rankdata(-x),
        axis=1,
        raw=True
    ))[gene_list]

    del data_df

    score = res.apply(
        func=(lambda x: u_stat(x, maxRank=maxRank)),
        axis=1,
        raw=True
    )
    return score


def score_genes(
        adata: AnnData,
        gene_list: List[str],
        maxRank: int = 1500,
        bs: int = 500,
        score_name: str = "UCell_score",
        random_state: Optional[int] = None,
        copy: bool = False,
        use_raw: Optional[bool] = None,
        verbose: int = 0,
        joblib_kwargs: dict = {'n_jobs': 4}
) -> Optional[AnnData]:
    """

    UCell signature scoring method is a Python implementation of the scoring method proposed by Andreatta et al. 2021.

    Massimo Andreatta and Santiago J Carmona. „UCell: Robust and scalable single-cell
    gene signature scoring“. en. In: Comput. Struct. Biotechnol. J. 19 (June 2021),
    pp. 3796–3798 (cit. on pp. iii, 2, 9, 15, 16).

    Implementation is inspired by score_genes method of Scanpy
    (https://scanpy.readthedocs.io/en/latest/generated/scanpy.tl.score_genes.html#scanpy.tl.score_genes)

    Args:
        adata: AnnData object containing the gene expression data.
        gene_list: A list of genes (signature) for which the cells are scored for.
        maxRank: Cutoff for maximum rank allowed.
        bs: The number of cells in a processing batch.
        score_name: Column name for scores added in `.obs` of data.
        random_state: Seed for random state.
        copy: Indicates whether original or a copy of `adata` is modified.
        use_raw: Whether to compute gene signature score on raw data stored in `.raw` attribute of `adata`
        verbose: If verbose is larger than 0, print statements are shown.
        joblib_kwargs: Keyword argument for parallel execution with joblib.

    Returns:
        If copy=True, the method returns a copy of the original data with stored UCell scores in `.obs`, otherwise
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
    if not isinstance(maxRank, int):
        raise ValueError(f"maxRank {maxRank} must be of type int")

    # check maxRank is not larger than available nr. of genes
    if maxRank > len(_adata.var_names):
        print(
            f"Provided maxRank is larger than the number of available genes. Set maxRank=len(adata.var_names)"
        )
        maxRank = len(_adata.var_names)

    # check that signature is not longer than maxRank
    if maxRank < len(gene_list) <= len(_adata.var_names):
        warnings.warn(
            f"The provided signature contains more genes than the maxRank parameter. maxRank is increased to "
            f"signature length "
        )
        maxRank = len(gene_list)

    sparse_X = issparse(_adata.X)
    if sparse_X and not isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()
        warnings.warn("Chaning adata.X format to CSR format")
        # create groups of managable sizes
    bss = pd.cut(np.arange(_adata.obs.shape[0]), (_adata.obs.shape[0] // bs + 1), labels=False)

    scores = Parallel(**joblib_kwargs)(
        delayed(compute_ranks_and_ustat)(
            X_data=_adata[group[1].index,].X.data if sparse_X else _adata[group[1].index,].X,
            X_indices=_adata[group[1].index,].X.indices if sparse_X else None,
            X_indptr=_adata[group[1].index,].X.indptr if sparse_X else None,
            X_shape=_adata[group[1].index,].X.shape if sparse_X else None,
            index=group[1].index,
            columns=_adata.var_names,
            gene_list=gene_list,
            maxRank=maxRank) for group in _adata.obs.groupby(bss))
    scores = pd.concat(scores)

    adata.obs[score_name] = scores

    sc.logging.info(
        "    finished",
        time=start,
        deep=("added\n" f"    {score_name!r}, score of gene set (adata.obs)."),
    )
    return adata if copy else None
