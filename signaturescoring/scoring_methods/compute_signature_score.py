import itertools
import warnings
from typing import List

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse, isspmatrix_csr

from signaturescoring.utils.utils import nanmean


def compute_signature_score(adata: AnnData,
                            sig_genes: List[str],
                            ctrl_genes: List[List[str]],
                            max_block: int = 10000,
                            max_nr_ctrl: int = 10000,
                            ):
    """
    The method computes for all Tirosh-based scoring methods the cell scores given the signature genes and the control
    gene sets for each signature gene.

    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        sig_genes: A list of genes for which the cells are scored for.
        ctrl_genes: A list of control gene lists. The length of the outer list must correspond to the length of the
            sig_genes
        max_block: Maximum number of cell before changing to parallel score computation for grouped cells.
        max_nr_ctrl: Maximum number of total control gene number before changing to parallel score computation for
            grouped cells.

    Returns:
        Returns vector of scores for each cell.
    """
    assert isinstance(sig_genes, list) and isinstance(ctrl_genes, list), \
        f'Signature genes and control genes must be of type list.'

    assert all([isinstance(ctrl, list) for ctrl in ctrl_genes]), \
        f'Control genes needs to be a list of lists containing the control genes.'

    sparse_X = issparse(adata.X)

    control_genes = list(itertools.chain(*ctrl_genes))

    control_genes_with_weights = pd.Series(control_genes).value_counts().sort_index()

    splits_of_cells = []

    if len(adata.obs_names) > max_block and len(control_genes_with_weights) > max_nr_ctrl:
        splits_of_cells = np.array_split(adata.obs_names, len(adata.obs_names) // max_block)
        if sparse_X and not isspmatrix_csr(adata.X):
            adata.X = adata.X.tocsr()
            warnings.warn("Chaning adata.X format to CSR format")

    if len(splits_of_cells) > 1:
        signature_scores_for_blocks = []
        control_scores_for_blocks = []
        for curr_cells in splits_of_cells:
            x_list = adata[curr_cells, sig_genes].X
            x_list = nanmean(x_list, axis=1, dtype="float64")
            signature_scores_for_blocks.append(x_list)

            if sparse_X:
                curr_score = np.average(adata[curr_cells, control_genes_with_weights.index].X.todense(),
                                        axis=1,
                                        weights=control_genes_with_weights.values)
            else:
                curr_score = np.average(adata[curr_cells, control_genes_with_weights.index].X,
                                        axis=1,
                                        weights=control_genes_with_weights.values)

            curr_score = np.array(curr_score).reshape(-1)
            control_scores_for_blocks.append(curr_score)

        x_list = np.concatenate(signature_scores_for_blocks)
        x_control = np.concatenate(control_scores_for_blocks)
    else:
        x_list = adata[:, sig_genes].X
        x_list = nanmean(x_list, axis=1, dtype="float64")
        if sparse_X:
            x_control = np.average(adata[:, control_genes_with_weights.index].X.todense(),
                                   axis=1,
                                   weights=control_genes_with_weights.values)
        else:
            x_control = np.average(adata[:, control_genes_with_weights.index].X,
                                   axis=1,
                                   weights=control_genes_with_weights.values)

    x_control = np.array(x_control).reshape(-1)

    score = x_list - x_control

    if score.shape != (adata.shape[0],):
        raise ValueError(
            f'Computation of scores did not result in correct shape of the scores vector. ',
            f'Got score.shape={score.shape}, but expected {(adata.shape[0],)}')
    return score
