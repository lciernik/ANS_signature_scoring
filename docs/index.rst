.. ANS documentation master file, created by
   sphinx-quickstart on Thu Oct  5 10:24:37 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ANS: Adjusted Neighborhood Scoring
==================================
In the field of single-cell RNA sequencing (scRNA-seq), gene signature scoring is integral for pinpointing and characterizing distinct cell populations. However, challenges arise in ensuring the robustness and comparability of scores across various gene signatures and across different batches and conditions. Addressing these challenges, we evaluated the stability of established methods such as Scanpy, UCell, and JASMINE in the context of scoring cells of different types and states. Additionally, we introduced a new scoring method, the Adjusted Neighbourhood Scoring (ANS), that builds on the traditional Scanpy method and improves the handling of the control gene sets. We further exemplified the usability of ANS scoring in differentiating between cancer-associated fibroblasts and malignant cells undergoing epithelial-mesenchymal transition (EMT) in four cancer types and evidenced excellent classification performance (AUCPR train: 0.95-0.99, AUCPR test: 0.91-0.99). In summary, our research introduces the ANS as a robust and deterministic scoring approach that enables the comparison of diverse gene signatures. The results of our study contribute to the development of more accurate and reliable methods for analyzing scRNA-seq data.

.. note::
   A preprint describing ANS and showing the results of signature scoring methods benchmark is  `now available <https://doi.org/10.1101/2023.09.20.558114>`_.

Navigation
----------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials
   api


* :ref:`search`


Getting started
---------------

Installation
^^^^^^^^^^^^
We aim for Python versions 3.8+. Run:

.. code-block:: bash

   pip install git+https://github.com/lciernik/ANS_signature_scoring.git

*Disclaimer*: The implementations of all Tirosh et al. 2016 based scoring methods are largely based on the implementation of the :func:`score_genes` method in `Scanpy <https://scanpy.readthedocs.io>`_.


Basic usage in Python
^^^^^^^^^^^^^^^^^^^^^
The package allows full compatibility with the Python scRNA-seq analysis toolbox `Scanpy <https://scanpy.readthedocs.io>`_. The scoring methods are applied on preprocessed (log-normalized) scRNA-seq.

.. code-block:: python

    import signaturescoring as ssc

    ssc.score_signature(
        adata=adata,                            # preprocessed (log-normalized) gene expression data in an AnnData object
        gene_list=gene_signature,               # gene expression signature, type list
        method='adjusted_neighborhood_scoring',
        ctrl_size=100,
        score_name='scores',                    # scores stored in adata.obs column defined by score_name
    )

    print(adata.obs['scores'].describe())


Other `method` values:

- `seurat_scoring`, `seurat_ag_scoring`, and `seurat_lvg_scoring`: Python implementation of the scoring method `AddModuleScore <https://satijalab.org/seurat/reference/addmodulescore>`_ of the package `Seurat <https://satijalab.org/seurat/>`_ first proposed by `Tirosh et al. 2016 <https://doi.org/10.1126/science.aad0501>`_ and two alternatives (this paper).
- `jasmine_scoring`: Python implementation of `JASMINE <https://github.com/NNoureen/JASMINE>`_ by `Noureen et al. 2022 <https://github.com/NNoureen/JASMINE>`_. Requires an additional argument `score_method` with the values `likelihood` or `oddsratio`.
- `ucell_scoring`: Python implementation of `UCell <https://github.com/carmonalab/UCell>`_ by `Andreatta et Carmona 2021 <https://doi.org/10.1016/j.csbj.2021.06.043>`_.


Basic usage in R
^^^^^^^^^^^^^^^^
The repository contains an R implementation of the novel scoring method in the folder `src_R/adjusted_neighborhood_scoring.R`. The file can be downloaded, and the method can be loaded for usage.

*Disclaimer*: The code is largely based on the implementation of the `AddModuleScore` method of the `Seurat <https://satijalab.org/seurat/>`_ package.

Note: ANS for R should be used on Seurat objects. Source the file in your script and use it identically to `AddModuleScore <https://satijalab.org/seurat/reference/addmodulescore>`_.

Example:

.. code-block:: R

   source('MT/ANS_signature_scoring/src_R/adjusted_neighborhood_scoring.R')

   # Initialize the Seurat object with the log-normalized data.
   # e.g. Peripheral Blood Mononuclear Cells (PBMC) freely available from 10X Genomics
   pbmc <- "..."

   # List of signatures
   markers <- list(markers = gene_list)

   # score data
   pbmc <- AdjustedNeighborhoodScoring(pbmc, features = markers)


