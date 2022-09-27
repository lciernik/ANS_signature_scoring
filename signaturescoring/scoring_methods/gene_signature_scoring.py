from typing import List,Optional

from anndata import AnnData
from scanpy.tools import score_genes as original_scanpy_scoring

from signaturescoring.scoring_methods.AGCG_tirosh_signature_scoring import \
    score_genes as agcg_tirosh_scoring
from signaturescoring.scoring_methods.LVCG_tirosh_signature_scoring import \
    score_genes as lvcg_tirosh_scoring
from signaturescoring.scoring_methods.adjusted_neighborhood_signature_scoring import \
    score_genes as adjusted_neighborhood_signature_scoring
from signaturescoring.scoring_methods.corrected_scanpy_signature_scoring import \
    score_genes as corrected_scanpy_scoring
from signaturescoring.scoring_methods.jasmine_signature_scoring import \
    score_genes as jasmine_scoring
from signaturescoring.scoring_methods.neighborhood_signature_scoring import \
    score_genes as neighborhood_scoring
from signaturescoring.scoring_methods.tirosh_signature_scoring import \
    score_genes as tirosh_scoring
from signaturescoring.scoring_methods.ucell_signature_scoring import \
    score_genes as ucell_scoring


def score_signature(method: str, adata: AnnData, gene_list: List[str], **kwarg)-> Optional[AnnData]:
    """
    Wrapper method to call one of the available gene expression signature scoring methods (AGCGS, ANS, JASMINE, LVCGS,
    NS, Scanpy, Tirosh, UCell).
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
        return adjusted_neighborhood_signature_scoring(adata, gene_list, **kwarg)
    elif method == "agcg_tirosh_scoring":
        return agcg_tirosh_scoring(adata, gene_list, **kwarg)
    elif method == "corrected_scanpy_scoring":
        return corrected_scanpy_scoring(adata, gene_list, **kwarg)
    elif method == "jasmine_scoring":
        return jasmine_scoring(adata, gene_list, **kwarg)
    elif method == "lvcg_tirosh_scoring":
        return lvcg_tirosh_scoring(adata, gene_list, **kwarg)
    elif method == "neighborhood_scoring":
        return neighborhood_scoring(adata, gene_list, **kwarg)
    elif method == "original_scanpy_scoring":
        return original_scanpy_scoring(adata, gene_list, **kwarg)
    elif method == "tirosh_scoring":
        return tirosh_scoring(adata, gene_list, **kwarg)
    elif method == "ucell_scoring":
        return ucell_scoring(adata, gene_list, **kwarg)
    else:
        msg = f""" 
        Unknown gene scoring method {method}. 
        Choose between ['adjusted_neighborhood_scoring', 'agcg_tirosh_scoring', 'corrected_scanpy_scoring', 
                        'jasmine_scoring', 'lvcg_tirosh_scoring', 'neighborhood_scoring', 'original_scanpy_scoring', 
                        'tirosh_scoring', 'ucell_scoring']
        """
        raise ValueError(msg)
