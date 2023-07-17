# ANS: Adjusted Neighborhood Scoring to  improve assessment of gene signatures in single-cell RNA-seq data
A gene expression signature scoring Python package.  

This repository accompanies the work: Laure Ciernik, Agnieszka Kraft, Joséphine Yates, Florian Barkmann, and 
Valentina Boeva, “ANS: Adjusted Neighborhood Scoring to  improve assessment of gene signatures in single-cell RNA-seq data”.

*Note: This repository is in the experimental stage. Changes to the API may appear.*

## Installation 
We aim for Python versions 3.8+. Run:

```
pip install git+https://github.com/lciernik/ANS_signature_scoring.git
```

*Disclaimer*: The implementations of all Tirosh et al. 2016 based scoring methods are largely based on the implementation of the [`score_genes`](https://scanpy.readthedocs.io/en/latest/generated/scanpy.tl.score_genes.html) method in [Scanpy](https://scanpy.readthedocs.io).


#### Method implementation in R 
The repository contains an R implementation of the novel scoring method in the folder 
`src_R/adjusted_neighborhood_scoring.R`. The file can be downloaded, and the methods can be loaded for usage. 

*Disclaimer*: The code is largely based on the implementation of the [`AddModuleScore`](https://satijalab.org/seurat/reference/addmodulescore) 
method of [the Seurat package](https://satijalab.org/seurat/index.html). 


## Getting started
The package allows full compatibility with the Python scRNA-seq analysis toolbox [Scanpy](https://scanpy.readthedocs.io/en/stable/index.html).
The scoring methods are applied on preprocessed (log-normalized) scRNA-seq. 
```python
import signaturescoring as ssc

ssc.score_signature(
    adata=adata,                            # preprocessed (log-normalized) gene expression data in an AnnData object 
    gene_list=gene_signature,               # gene expression signature, type list
    method='adjusted_neighborhood_scoring',
    ctrl_size=100, 
    score_name='scores',                    # scores stored in adata.obs column defined by score_name
)

print(adata.obs['scores'].describe())
```
Other `method` values:
- `tirosh_scoring`, `tirosh_ag_scoring`, and `tirosh_lvg_scoring`: Python implementation of the scoring method proposed by [Tirosh et al. 2016](https://doi.org/10.1126/science.aad0501) and two alternatives (this paper). 
- `jasmine_scoring`: Python implementation of [JASMINE](https://github.com/NNoureen/JASMINE) by [Noureen et al. 2022](https://doi.org/10.7554/eLife.71994). Requires an additional argument `score_method` with the values `likelihood` or `oddsratio`. 
- `ucell_scoring`: Python implementation of [UCell](https://github.com/carmonalab/UCell) by [Andreatta et Carmona 2021](https://doi.org/10.1016/j.csbj.2021.06.043). 

