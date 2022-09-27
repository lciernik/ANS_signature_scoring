from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scanpy._utils import AnyRandom, _check_use_raw

from signaturescoring.utils.utils import nanmean


def score_genes(
        adata: AnnData,
        gene_list: Sequence[str],
        ctrl_size: int = 100,
        gene_pool: Optional[Sequence[str]] = None,
        n_bins: int = 25,
        score_name: str = "corrected_scanpy_score",
        random_state: AnyRandom = 0,
        copy: bool = False,
        use_raw: Optional[bool] = None,
        verbose: int = 0,
) -> Optional[AnnData]:
    """
    The following scoring method is very similar to Scanpy's scoring method
    (https://scanpy.readthedocs.io/en/latest/generated/scanpy.tl.score_genes.html, inkl. code). Scanpy's scoring method
    does not allow sampling from an expression bin more than once, even if more than one signature gene land into the same
    expression bin. This behaviour is corrected in the following scoring method.

    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        gene_list: A list of genes for which the cells are scored for.
        ctrl_size: The number of control genes selected for each gene in the gene_list.
        gene_pool: The pool of genes out of which control genes can be selected.
        n_bins: The number of average gene expression bins to use.
        score_name: Column name for scores added in `.obs` of data.
        random_state: Seed for random state
        copy: Indicates whether original or a copy of `adata` is modified.
        use_raw: Whether to compute gene signature score on raw data stored in `.raw` attribute of `adata`
        verbose: If verbose is larger than 0, print statements are shown.

    Returns:
        If copy=True, the method returns a copy of the original data with stored ANS scores in `.obs`, otherwise None
        is returned.
    """

    start = sc.logging.info(f"computing score {score_name!r}")
    if verbose > 0:
        print(f"computing score {score_name!r}")

    adata = adata.copy() if copy else adata
    use_raw = _check_use_raw(adata, use_raw)

    if random_state is not None:
        np.random.seed(random_state)

    # remove genes from gene_list not available in the data
    var_names = adata.raw.var_names if use_raw else adata.var_names
    var_names = var_names.tolist()
    gene_list = (set(var_names)).intersection(set(gene_list))
    genes_to_ignore = (set(gene_list)).difference(set(var_names))
    if len(genes_to_ignore) > 0:
        sc.logging.warning(f"genes are not in var_names and ignored: {genes_to_ignore}")
    if len(gene_list) == 0:
        raise ValueError("No valid genes were passed for scoring.")

    # compute the correct gene pool
    gene_pool = (
        var_names if (gene_pool is None) else [x for x in gene_pool if x in var_names]
    )
    if not gene_pool:
        raise ValueError("No valid genes were passed for reference set.")

    # get data and correct subset for gene_pool
    _adata = adata.raw if use_raw else adata
    _adata_subset = (
        _adata[:, gene_pool] if len(gene_pool) < len(_adata.var_names) else _adata
    )

    # compute average expression of genes and remove missing data
    obs_avg = pd.Series(nanmean(_adata_subset.X, axis=0), index=gene_pool)
    obs_avg = obs_avg[np.isfinite(obs_avg)]
    if verbose > 0:
        print(f"obs_avg.shape: {obs_avg.shape}")

    # correct computations of cuts
    ranked_obs_avg = obs_avg.rank(method="min")
    obs_cut = pd.cut(ranked_obs_avg, n_bins, labels=False)
    if verbose > 0:
        print(f"Got {len(np.unique(obs_cut))} bins.")

    if random_state is not None:
        np.random.seed(random_state)
    # get the bin ids of the genes in gene_list.
    cuts = obs_cut.loc[list(gene_list)]
    control_genes = set()
    for cut in cuts:
        r_genes = np.array(obs_cut[obs_cut == cut].index)
        np.random.shuffle(r_genes)
        control_genes.update(set(r_genes[:ctrl_size]))

    # To index, we need a list – indexing implies an order.
    control_genes = list(control_genes - gene_list)
    gene_list = list(gene_list)

    x_list = _adata[:, gene_list].X
    x_list = nanmean(x_list, axis=1, dtype="float64")

    x_control = _adata[:, control_genes].X
    x_control = nanmean(x_control, axis=1, dtype="float64")

    score = x_list - x_control

    adata.obs[score_name] = pd.Series(
        np.array(score).ravel(), index=adata.obs_names, dtype="float64"
    )

    sc.logging.info(
        "    finished",
        time=start,
        deep=(
            "added\n"
            f"    {score_name!r}, score of gene set (adata.obs).\n"
            f"    {len(control_genes)} total control genes are used."
        ),
    )
    return adata if copy else None
