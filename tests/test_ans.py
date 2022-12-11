import numpy as np
import pytest

from signaturescoring.scoring_methods.gene_signature_scoring import score_signature


@pytest.fixture
def scoring_method():
    return 'adjusted_neighborhood_scoring'


@pytest.fixture
def score_name():
    return 'ANS_score'


def test_score_genes(adata, gene_list, scoring_method, ctrl_size, score_name):
    score_signature(adata, gene_list, method=scoring_method, ctrl_size=ctrl_size, score_name=score_name)
    assert score_name in adata.obs, f'No column {score_name} found in adata'
    assert np.all(adata.obs[score_name] == 0), f'Unexpected scores expected all equal to zero, got ' \
                                               f'{adata.obs[score_name]}'


def test_score_genes_two(adata2, gene_list2, scoring_method, ctrl_size2, score_name):
    cg = score_signature(adata2, gene_list2, method=scoring_method, ctrl_size=ctrl_size2, score_name=score_name,
                         return_control_genes=True)
    assert score_name in adata2.obs, f'No column {score_name} found in adata2'
    assert np.all(adata2.obs[score_name][0:500] == 1) and np.all(adata2.obs[score_name][500:1000] == 2), \
        f'Unexpected scores expected all equal to zero, got {adata2.obs[score_name]}'

def test_score_genes_three(adata3, gene_list3, scoring_method, ctrl_size2, score_name):
    cg = score_signature(adata3, gene_list3, method=scoring_method, ctrl_size=ctrl_size2, score_name=score_name,
                         return_control_genes=True)
    assert score_name in adata3.obs, f'No column {score_name} found in adata2'
    assert np.all(adata3.obs[score_name][0:10] == 1) and np.all(adata3.obs[score_name][10:] == 0), \
        f'Unexpected scores expected all equal to zero, got {adata3.obs[score_name]}'


def test_gene_list(adata, gene_list, scoring_method, ctrl_size, score_name):
    # test correct types of signature
    try:
        score_signature(adata, gene_list, method=scoring_method, ctrl_size=ctrl_size, score_name=score_name)
        score_signature(adata, set(gene_list), method=scoring_method, ctrl_size=ctrl_size, score_name=score_name)
    except:
        pytest.fail('Unexpected Error when scoring for a valid gene list or set.')

    # test scoring one gene is possible
    try:
        score_signature(adata, gene_list[0], method=scoring_method, ctrl_size=ctrl_size, score_name=score_name)
    except:
        pytest.fail('Unexpected Error when scoring for single signature gene.')
    assert np.all(adata.obs[score_name] == 0), f'Unexpected scores expected all equal to zero, got ' \
                                               f'{adata.obs[score_name]}'


def test_valid_control_sets_possible(adata, gene_list, scoring_method, ctrl_size, score_name):
    # test ctrl_size larger than reference pool
    #  -> signature is so large no valid control sets can be created
    with pytest.raises(ValueError):
        score_signature(adata, adata.var_names.tolist(), method=scoring_method, ctrl_size=ctrl_size, score_name=score_name)
    #  -> ctrl_size too large
    with pytest.raises(ValueError):
        score_signature(adata, gene_list, method=scoring_method, ctrl_size=100, score_name=score_name)
    #  -> ctrl_size cannot be equal to zero
    with pytest.raises(ValueError):
        score_signature(adata, gene_list, method=scoring_method, ctrl_size=0, score_name=score_name)
    # test wrong input for ctrl_size
    with pytest.raises(Exception):
        score_signature(adata, gene_list, method=scoring_method, ctrl_size='0', score_name=score_name)
    with pytest.raises(Exception):
        score_signature(adata, gene_list, method=scoring_method, ctrl_size=0.1, score_name=score_name)

    adata[:, ['Gene_36', 'Gene_99']].X = 2
    # test signature genes removed from signature if too close to the right boundary
    new_gene_list = score_signature(adata, ['Gene_98', 'Gene_36', 'Gene_99'], method=scoring_method,
                                    ctrl_size=ctrl_size,
                                    score_name=score_name, remove_genes_with_invalid_control_set=True,
                                    return_gene_list=True)
    assert isinstance(new_gene_list, list) and len(new_gene_list) == 1 \
           and set(new_gene_list) != set(['Gene_98', 'Gene_36', 'Gene_99'])

    new_gene_list = score_signature(adata, ['Gene_98', 'Gene_36', 'Gene_99'], method=scoring_method,
                                    ctrl_size=ctrl_size,
                                    score_name=score_name, remove_genes_with_invalid_control_set=False,
                                    return_gene_list=True)
    assert isinstance(new_gene_list, list) and set(new_gene_list) == set(['Gene_98', 'Gene_36', 'Gene_99'])


def test_return_control_genes(adata, gene_list, scoring_method, ctrl_size, score_name):
    control_genes, new_gene_list = score_signature(adata, gene_list, method=scoring_method, ctrl_size=ctrl_size,
                                                   score_name=score_name, return_control_genes=True,
                                                   return_gene_list=True)
    assert isinstance(control_genes, list) and len(control_genes) == len(new_gene_list)
    assert np.all([isinstance(x, list) and np.all([isinstance(y, str) for y in x]) for x in control_genes])
