from typing import List, Optional

from anndata import AnnData
from scanpy.tools import score_genes as scanpy_scoring

from signaturescoring.scoring_methods.adjusted_neighborhood_scoring import \
    score_genes as adjusted_neighborhood_scoring
from signaturescoring.scoring_methods.corrected_scanpy_scoring import \
    score_genes as corrected_scanpy_scoring
from signaturescoring.scoring_methods.jasmine_scoring import \
    score_genes as jasmine_scoring
from signaturescoring.scoring_methods.neighborhood_scoring import \
    score_genes as neighborhood_scoring
from signaturescoring.scoring_methods.tirosh_ag_scoring import \
    score_genes as tirosh_ag_scoring
from signaturescoring.scoring_methods.tirosh_lv_scoring import \
    score_genes as tirosh_lv_scoring
from signaturescoring.scoring_methods.tirosh_scoring import \
    score_genes as tirosh_scoring
from signaturescoring.scoring_methods.ucell_signature_scoring import \
    score_genes as ucell_scoring


def score_signature(adata: AnnData,
                    gene_list: List[str],
                    method: str = 'adjusted_neighborhood_scoring',
                    **kwarg) -> Optional[AnnData]:
    """
    Wrapper method to call one of the available gene expression signature scoring methods (ANS, Tirosh, Triosh_AG,
    Tirosh_LV, Scanpy, Jamine, UCell).
    Args:
        method: Scoring method to use.
        adata: AnnData object containing the gene expression data.
        gene_list: A list of genes,i.e., gene expression signature, for which the cells are scored for.
        **kwarg: Other keyword arguments specific for the scoring method. See individual scoring methods for available
            keyword arguments.

    Returns:
        If copy=True, the method returns a copy of the original data with stored ANS scores in `.obs`, otherwise None
        is returned.
    """
    if method == "adjusted_neighborhood_scoring":
        return adjusted_neighborhood_scoring(adata, gene_list, **kwarg)
    elif method == "tirosh_scoring":
        return tirosh_scoring(adata, gene_list, **kwarg)
    elif method == "tirosh_ag_scoring":
        return tirosh_ag_scoring(adata, gene_list, **kwarg)
    elif method == "tirosh_lv_scoring":
        return tirosh_lv_scoring(adata, gene_list, **kwarg)
    elif method == "scanpy_scoring":
        return scanpy_scoring(adata, gene_list, **kwarg)
    elif method == "jasmine_scoring":
        return jasmine_scoring(adata, gene_list, **kwarg)
    elif method == "ucell_scoring":
        return ucell_scoring(adata, gene_list, **kwarg)
    elif method == "neighborhood_scoring":
        return neighborhood_scoring(adata, gene_list, **kwarg)
    elif method == "corrected_scanpy_scoring":
        return corrected_scanpy_scoring(adata, gene_list, **kwarg)
    else:
        msg = f""" 
        Unknown gene scoring method {method}. 
        Choose between ['adjusted_neighborhood_scoring', 'tirosh_scoring', 'tirosh_ag_scoring', 
                        'tirosh_lv_scoring', 'scanpy_scoring', 'jasmine_scoring', 'ucell_scoring', 
                        'neighborhood_scoring', 'corrected_scanpy_scoring']
        """
        raise ValueError(msg)
