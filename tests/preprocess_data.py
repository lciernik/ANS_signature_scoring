import logging

from scanpy.preprocessing import filter_cells, filter_genes, normalize_total, log1p


def preprocess(adata, min_genes=200, min_cells=3, target_sum=1e4, copy=False, verbose=0, log=None):
    """
    This function preprocesses raw counts based on the tutorials of Scanpy
    (https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html#Preprocessing-and-clustering-3k-PBMCs).

    Args:
        adata: gene expression matrix in AnnData object.
        min_genes: Minimum number of genes required to be expressed in a cell.
        min_cells:  Minimum number of cell required to be expressed for a gene.
        target_sum: Unit of expression values after normalization of counts per cell.
                    E.g., target_sum=1e4 results in Counts per 10K.
        copy: Indicates whether the function returns a preprocessed copy of the original data.
        verbose: Indicates whether print statements should be displayed. If verbose=0, then minimal print statements are
                 displayed
        log: Ability to pass a logger

    Returns: If copy=True, then the method returns a preprocessed copy of the original data. Otherwise, it returns None.

    """
    if log is None:
        log = logging
    else:
        log = logging.getLogger(log)

    if not isinstance(min_genes, int) or not isinstance(min_cells, int):
        raise TypeError(f'The arguments min_genes and min_cells need be of type int')
    if not (isinstance(target_sum, int) or isinstance(target_sum, float)):
        raise TypeError(f'The argument target_sum an min_cells need be of type int or float')

    adata = adata.copy() if copy else adata

    if min_genes is not None:
        filter_cells(adata, min_genes=min_genes)
        if verbose > 0:
            log.info(f"filter_cells with min_genes={min_genes}")
    else:
        log.info(f'no cell filtering')

    if min_cells is not None:
        filter_genes(adata, min_cells=min_cells)
        if verbose > 0:
            log.info(f"filter_genes with min_cells={min_cells}")
    else:
        log.info(f'no gene filtering')

    if target_sum is not None:
        normalize_total(adata, target_sum=target_sum)
        if verbose > 0:
            log.info(f"normalize counts per cell to CP{target_sum}")
    else:
        log.info(f"data not normalized")

    if verbose > 0:
        log.info(f"apply log1p")
    log1p(adata)

    return adata if copy else None
