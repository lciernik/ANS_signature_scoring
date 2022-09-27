from unittest import TestCase

import anndata
import numpy as np
import pytest

from signaturescoring.data.preprocess_data import preprocess


@pytest.fixture
def adata():
    X = np.zeros((20000, 20000))
    X[np.triu_indices_from(X)] = 1
    adata = anndata.AnnData(X, dtype='float64')
    return adata


def test_preprocess(adata):
    with pytest.raises(TypeError):
        preprocess(adata, '200', 3, target_sum=1e4)
    with pytest.raises(TypeError):
        preprocess(adata, 200, '3', target_sum=1e4)
    with pytest.raises(TypeError):
        preprocess(adata, 200, 3, target_sum=[1e4])
    assert isinstance(preprocess(adata, 200, 3, target_sum=1e4, copy=True), anndata.AnnData)

    assert preprocess(adata, 200, 3, target_sum=1e4) is None


def test_filter_cells(adata):
    preprocess(adata, 200, 0, target_sum=1e4)
    print(adata.X.shape[0])
    assert adata.X.shape[0] == 19801


def test_filter_genes(adata):
    preprocess(adata, 0, 3, target_sum=1e4)
    print(adata.X.shape[1])
    assert adata.X.shape[1] == 19998

