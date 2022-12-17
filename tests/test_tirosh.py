import numpy as np
import pytest

from signaturescoring import score_signature


@pytest.fixture
def scoring_method_1():
    return 'tirosh_scoring'


@pytest.fixture
def scoring_method_2():
    return 'tirosh_ag_scoring'


@pytest.fixture
def scoring_method_3():
    return 'tirosh_lv_scoring'


@pytest.fixture
def score_name_1():
    return 'TIROSH_score'


@pytest.fixture
def score_name_2():
    return 'TIROSH_AG_score'


@pytest.fixture
def score_name_3():
    return 'TIROSH_LV_score'


@pytest.fixture
def n_bins():
    return 5


def test_score_genes_tirosh_1(adata, gene_list, ctrl_size, n_bins, scoring_method_1, score_name_1):
    score_signature(adata, gene_list, method=scoring_method_1, ctrl_size=ctrl_size, n_bins=n_bins,
                    score_name=score_name_1)
    assert score_name_1 in adata.obs, f'No column {score_name_1} found in adata'
    assert np.all(adata.obs[score_name_1] == 0), f'Unexpected scores expected all equal to zero, got ' \
                                                 f'{adata.obs[score_name_1]}'


def test_score_genes_tirosh_2(adata3, gene_list3, ctrl_size3, n_bins, scoring_method_1, score_name_1):
    score_signature(adata3, gene_list3, method=scoring_method_1, ctrl_size=ctrl_size3, n_bins=n_bins,
                    score_name=score_name_1)
    assert score_name_1 in adata3.obs, f'No column {score_name_1} found in adata3'
    assert np.all(adata3.obs[score_name_1][0:10] == 1) and np.all(adata3.obs[score_name_1][10:] == 0), \
        f'Unexpected scores expected all equal to zero, got {adata3.obs[score_name_1]}'


def test_score_genes_tirosh_ag_1(adata, gene_list, n_bins, scoring_method_2, score_name_2):
    score_signature(adata, gene_list, method=scoring_method_2, n_bins=n_bins, score_name=score_name_2)
    assert score_name_2 in adata.obs, f'No column {score_name_2} found in adata'
    assert np.all(adata.obs[score_name_2] == 0), f'Unexpected scores expected all equal to zero, got ' \
                                                 f'{adata.obs[score_name_2]}'


def test_score_genes_tirosh_ag_2(adata3, gene_list3, n_bins, scoring_method_2, score_name_2):
    score_signature(adata3, gene_list3, method=scoring_method_2, score_name=score_name_2,
                    n_bins=n_bins)
    assert score_name_2 in adata3.obs, f'No column {score_name_2} found in adata3'
    assert np.all(adata3.obs[score_name_2][0:10] == 1) and np.all(adata3.obs[score_name_2][10:] == 0), \
        f'Unexpected scores expected all equal to zero, got {adata3.obs[score_name_2]}'


def test_score_genes_tirosh_lv_1(adata, gene_list, ctrl_size, n_bins, scoring_method_3, score_name_3):
    score_signature(adata, gene_list, method=scoring_method_3, ctrl_size=ctrl_size, n_bins=n_bins,
                    score_name=score_name_3)
    assert score_name_3 in adata.obs, f'No column {score_name_3} found in adata'
    assert np.all(adata.obs[score_name_3] == 0), f'Unexpected scores expected all equal to zero, got ' \
                                                 f'{adata.obs[score_name_3]}'


def test_score_genes_tirosh_lv_2(adata3, gene_list3, ctrl_size3, n_bins, scoring_method_3, score_name_3):
    score_signature(adata3, gene_list3, method=scoring_method_3, ctrl_size=ctrl_size3, n_bins=n_bins,
                    score_name=score_name_3)
    assert score_name_3 in adata3.obs, f'No column {score_name_3} found in adata3'
    assert np.all(adata3.obs[score_name_3][0:10] == 1) and np.all(adata3.obs[score_name_3][10:] == 0), \
        f'Unexpected scores expected all equal to zero, got {adata3.obs[score_name_3]}'


def test_gene_list(adata, gene_list, ctrl_size, n_bins, scoring_method_1, scoring_method_2, scoring_method_3,
                   score_name_1, score_name_2, score_name_3):
    sc_methods = [scoring_method_1, scoring_method_2, scoring_method_3]
    sc_names = [score_name_1, score_name_2, score_name_3]
    for method, name in zip(sc_methods, sc_names):
        # test correct types of signature
        try:
            if 'ag' in method:
                score_signature(adata, gene_list, method=method, n_bins=n_bins, score_name=name)
                score_signature(adata, set(gene_list), method=method, n_bins=n_bins, score_name=name)
            else:
                score_signature(adata, gene_list, method=method, ctrl_size=ctrl_size, n_bins=n_bins, score_name=name)
                score_signature(adata, set(gene_list), method=method, ctrl_size=ctrl_size, n_bins=n_bins,
                                score_name=name)
        except:
            pytest.fail(f'Unexpected Error when scoring with {method} for a valid gene list or set.')

        # test scoring one gene is possible
        try:
            if 'ag' in method:
                score_signature(adata, gene_list[0], method=method, n_bins=n_bins, score_name=name)
            else:
                score_signature(adata, gene_list[0], method=method, ctrl_size=ctrl_size, n_bins=n_bins, score_name=name)
        except:
            pytest.fail(f'Unexpected Error when scoring with {method} for single signature gene.')
        assert np.all(adata.obs[name] == 0), f'Unexpected scores, expected all equal to zero, got ' \
                                             f'{adata.obs[name]}'


def test_return_control_genes(adata, gene_list, ctrl_size, n_bins, scoring_method_1, scoring_method_2, scoring_method_3,
                              score_name_1, score_name_2, score_name_3):
    sc_methods = [scoring_method_1, scoring_method_2, scoring_method_3]
    sc_names = [score_name_1, score_name_2, score_name_3]
    for method, name in zip(sc_methods, sc_names):
        if 'ag' in method:
            control_genes, new_gene_list = score_signature(adata, gene_list, method=method,
                                                           n_bins=n_bins,
                                                           score_name=name,
                                                           return_control_genes=True,
                                                           return_gene_list=True)
        else:
            control_genes, new_gene_list = score_signature(adata, gene_list, method=method,
                                                           ctrl_size=ctrl_size,
                                                           n_bins=n_bins,
                                                           score_name=name,
                                                           return_control_genes=True,
                                                           return_gene_list=True)
        assert isinstance(control_genes, list) and len(control_genes) == len(new_gene_list)
        assert np.all([isinstance(x, list) and np.all([isinstance(y, str) for y in x]) for x in control_genes])
