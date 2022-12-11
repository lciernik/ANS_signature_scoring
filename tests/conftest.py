import anndata as ad
import numpy as np
import pytest


@pytest.fixture
def adata():
    counts = np.ones((100, 100))
    adata = ad.AnnData(counts, dtype='float32')
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
    return adata


@pytest.fixture
def adata2():
    counts = np.zeros((1000, 1000))
    interesting_genes = np.random.choice(np.arange(1, 1000, 15), 10, replace=False)
    counts[0:500, interesting_genes] = 1
    counts[500:1000, interesting_genes] = 2
    adata2 = ad.AnnData(counts, dtype='float32')
    adata2.obs_names = [f"Cell_{i:d}" for i in range(adata2.n_obs)]
    adata2.var_names = [f"Gene_{i:d}" for i in range(adata2.n_vars)]
    return adata2


@pytest.fixture
def adata3():
    counts = np.ones((100, 5 * 9))
    list_val = [1, 5, 10, 15, 20]
    for i, j in enumerate(range(0, 5 * 9, 9)):
        counts[:, j:(j + 9)] = list_val[i]
    for i, j in enumerate(range(4, 5 * 9, 9)):
        counts[0:10, j] = list_val[i] + 1
    adata3 = ad.AnnData(counts, dtype='float32')
    adata3.obs_names = [f"Cell_{i:d}" for i in range(adata3.n_obs)]
    adata3.var_names = [f"Gene_{i:d}" for i in range(adata3.n_vars)]
    return adata3


@pytest.fixture
def gene_list(adata):
    sig_length = 10
    assert len(adata.var_names) > sig_length
    return adata.var_names.to_series().sample(sig_length).to_list()


@pytest.fixture
def gene_list2(adata2):
    gene_series = adata2.var_names[np.where(np.sum(adata2.X, axis=0))[0]]
    return gene_series.to_list()

@pytest.fixture
def gene_list3(adata3):
    gene_series = adata3.var_names[np.where(np.var(adata3.X, axis=0))[0]]
    return gene_series.to_list()


@pytest.fixture
def ctrl_size():
    return 10


@pytest.fixture
def ctrl_size2():
    return 2


@pytest.fixture
def ctrl_size3():
    return 4
