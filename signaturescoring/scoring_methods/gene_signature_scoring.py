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
from signaturescoring.scoring_methods.ucell_scoring import \
    score_genes as ucell_scoring


def score_signature(adata: AnnData,
                    gene_list: List[str],
                    method: str = 'adjusted_neighborhood_scoring',
                    **kwarg) -> Optional[AnnData]:
    """
    Wrapper method to call one of the available gene expression signature scoring methods (ANS, Tirosh, Triosh_AG,
    Tirosh_LV, Scanpy, Jasmine, UCell).

    Args:
        adata: AnnData object containing the gene expression data.
        gene_list: A list of genes,i.e., gene expression signature, for which the cells are scored for.
        method: Scoring method to use. One of ['adjusted_neighborhood_scoring', 'tirosh_scoring', 'tirosh_ag_scoring',
                'tirosh_lv_scoring', 'scanpy_scoring', 'jasmine_scoring', 'ucell_scoring', 'neighborhood_scoring',
                'corrected_scanpy_scoring']
        **kwarg: Other keyword arguments specific for the scoring method. See below names of individual scoring methods
                 and their available keyword arguments.

    Returns:
        If copy=True, the method returns a copy of the original data with stored ANS scores in `.obs`, otherwise None
        is returned.

    Notes:
        ANS: Adujsted neighborhood signature scoring method.
            (see signaturescoring.scoring_methods.adjusted_neighborhood_scoring.score_genes)
        Tirosh: Scoring method based on approach suggested by Tirosh et al. 2016 (10.1126/science.aad0501).
            (see signaturescoring.scoring_methods.tirosh_scoring.score_genes)
        Tiosh_AG, Tirosh_LV: Modifications of above method. First selecting all genes in an expression bin as control
                             genes. Latter selecting the least variable genes of an expression bin as control genes.
                             (see signaturescoring.scoring_methods.tirosh_[ag/lv]_scoring.score_genes)
        Scanpy: Scoring method implemented in Scanpy
                (https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes.html)
        Jasmine: Rank-based signature scoring method by Noureen et al. 2022 (https://doi.org/10.7554/eLife.71994)
                (see signaturescoring.scoring_methods.jasmine_scoring.score_genes)
        UCell: Rank-based signature scoring method by Andretta et al. 2021 (https://doi.org/10.1016/j.csbj.2021.06.043)
                (see signaturescoring.scoring_methods.ucell_scoring.score_genes)
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
