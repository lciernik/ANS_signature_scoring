# ANS: Adjusted Neighborhood Scoring to  improve assessment of gene signatures in single-cell RNA-seq data
A gene expression signature scoring package 

This repository accompanies the work: Laure Ciernik, Agnieszka Kraft, Joséphine Yates, Florian Barkmann,and 
Valentina Boeva, “ANS: Adjusted Neighborhood Scoring to  improve assessment of gene signatures in single-cell RNA-seq data”.

*Note: This repository is in the experimental stage. Changes to the API may appear.*

## Installation 

```
pip install git+ssh://git@github.com:lciernik/ANS_signature_scoring.git
```

#### Method implementation in R 
The repository contains an R implementation of the novel scoring method in the folder 
`src_R/adjusted_neighborhood_scoring.R`. The file can be downloaded and the methods can be loaded for usage. 

*Disclaimer*: The code is largely based on the implementation of the [`AddModuleScore`](https://satijalab.org/seurat/reference/addmodulescore) 
method of [the Seurat package](https://satijalab.org/seurat/index.html). 


## Getting started
The package allows full compatibility with the python scRNA-seq anal
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

