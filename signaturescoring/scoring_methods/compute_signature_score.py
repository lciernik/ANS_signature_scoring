import itertools
import multiprocessing
import warnings
from typing import List

import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.sparse import issparse, isspmatrix_csr

from signaturescoring.utils.utils import nanmean


def compute_signature_score(adata: AnnData,
                            sig_genes: List[str],
                            ctrl_genes: List[List[str]],
                            std=None,
                            adjust_all_genes: bool = False,
                            max_block: int = 10000,
                            max_nr_cntrl: int = 10000,
                            ):
    """
    The method computes for all Tirosh-based scoring methods the cell scores given the signature genes and the control
    gene sets for each signature gene.

    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        sig_genes: A list of genes for which the cells are scored for.
        ctrl_genes: A list of control gene lists. The length of the outer list must correspond to the length of the
            sig_genes
        std: A pandas Series containing the (estimated) standard deviation for each gene in `adata`. Passing the
            standard deviation vector implies score computation with std adjustment.
        adjust_all_genes: Apply gene signature scoring with standard deviation adjustment for each occurring gene.
            Divide each gene's expression (signature and control genes) by estimated standard deviation of the gene.
        max_block: Maximum number of cell before changing to parallel score computation for grouped cells.
        max_nr_cntrl: Maximum number of total control gene number before changing to parallel score computation for
            grouped cells.

    Returns:
        Returns vector of scores for each cell.
    """
    assert isinstance(sig_genes, list) and isinstance(ctrl_genes, list), \
        f'Signature genes and control genes must be of type list.'

    assert all([isinstance(ctrl, list) for ctrl in ctrl_genes]), \
        f'Control genes needs to be a list of lists containing the control genes.'
    sparse_X = issparse(adata.X)
    if std is not None:
        assert all(isinstance(elem, list) for elem in ctrl_genes) and \
               len(sig_genes) == len(ctrl_genes), \
            f'Control genes must be of type list of lists. ' \
            f'For scoring with variance adjustment we need as many control gene set ' \
            f'as signature genes (i.e., len(sig_genes) == len(ctrl_genes)).'

        assert all(x in std.index for x in sig_genes)

        if adjust_all_genes:
            x_list = adata[:, sig_genes].X
            x_list = x_list / std.loc[sig_genes].values[None, :]
            x_list = nanmean(x_list, axis=1, dtype="float64")

            control_genes = list(itertools.chain(*ctrl_genes))
            control_genes_with_weights = pd.Series(control_genes).value_counts().sort_index()
            sig_stds = std.loc[control_genes_with_weights.index].values[None, :]
            x_controls_unique = adata[:, control_genes_with_weights.index].X / sig_stds
            if sparse_X:
                x_control = np.average(x_controls_unique.todense(), axis=1, weights=control_genes_with_weights.values)
            else:
                x_control = np.average(x_controls_unique, axis=1, weights=control_genes_with_weights.values)
            x_control = np.array(x_control).reshape(-1)

            score = x_list - x_control

        else:

            def compute_score_per_sig_gene(sig_gene, curr_ctrl_genes):
                if sparse_X:
                    obs = np.array((adata[:, sig_gene].X.todense())).reshape(-1)
                else:
                    obs = np.array((adata[:, sig_gene].X)).reshape(-1)
                exp = nanmean(adata[:, curr_ctrl_genes].X, axis=1, dtype="float64")
                sig_score = (obs - exp) / std.loc[sig_gene]
                return sig_score

            num_cores = multiprocessing.cpu_count()
            scores_per_sig_gene = Parallel(n_jobs=num_cores)(
                delayed(compute_score_per_sig_gene)(sig_gene, ctrl_gene) for sig_gene, ctrl_gene in
                zip(sig_genes, ctrl_genes))
            score = nanmean(np.array(scores_per_sig_gene).T, axis=1, dtype="float64")

    else:
        # x_list = adata[:, sig_genes].X
        # x_list = nanmean(x_list, axis=1, dtype="float64")

        control_genes = list(itertools.chain(*ctrl_genes))
        #         x_list = adata[:, sig_genes].X
        #         if issparse(x_list):
        #             x_list = np.array(_sparse_nanmean(x_list, axis=1)).flatten()
        #         else:
        #             x_list = np.nanmean(x_list, axis=1, dtype='float64')

        #         x_control = adata[:, control_genes].X
        #         if issparse(x_control):
        #             x_control = np.array(_sparse_nanmean(x_control, axis=1)).flatten()
        #         else:
        #             x_control = np.nanmean(x_control, axis=1, dtype='float64')
        control_genes_with_weights = pd.Series(control_genes).value_counts().sort_index()
        splits_of_cells = []
        if len(adata.obs_names) > max_block and len(control_genes_with_weights) > max_nr_cntrl:
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
    #         score =  pd.Series(np.array(score).ravel(), index=adata.obs_names, dtype='float64')

    if score.shape != (adata.shape[0],):
        raise ValueError(
            f'Computation of scores did not result in correct shape of the scores vector. ',
            f'Got score.shape={score.shape}, but expected {(adata.shape[0],)}')
    return score
