import anndata as ad
import numpy as np
import pytest
import warnings

from signaturescoring.scoring_methods.gene_signature_scoring import score_signature


@pytest.fixture
def adata():
    counts = np.ones((100, 100))
    adata = ad.AnnData(counts, dtype='float32')
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
    return adata

@pytest.fixture
def adata2():
    counts = np.zeros((100, 100))
    increasing_col = np.arange(0,100)
    interesting_genes = np.random.choice(100, 10)
    #TODO
    counts[:,interesting_genes] = (np.ones((3,1))*increasing_col)
    adata2 = ad.AnnData(counts, dtype='float32')
    adata2.obs_names = [f"Cell_{i:d}" for i in range(adata2.n_obs)]
    adata2.var_names = [f"Gene_{i:d}" for i in range(adata2.n_vars)]
    return adata2


@pytest.fixture
def gene_list(adata):
    sig_length = 10
    assert len(adata.var_names) > sig_length
    return adata.var_names.to_series().sample(sig_length).to_list()


@pytest.fixture
def scoring_method():
    return 'adjusted_neighborhood_scoring'


@pytest.fixture
def ctrl_size():
    return 10


@pytest.fixture
def score_name():
    return 'ANS_score'


def test_score_genes(adata, gene_list, scoring_method, ctrl_size, score_name):
    score_signature(scoring_method, adata, gene_list, ctrl_size=ctrl_size, score_name=score_name)
    assert score_name in adata.obs, f'No column {score_name} found in adata'
    assert np.all(adata.obs[score_name] == 0), f'Unexpected scores expected all equal to zero, got ' \
                                               f'{adata.obs[score_name]}'





def test_gene_list(adata, gene_list, scoring_method, ctrl_size, score_name):
    # test empty signature
    with pytest.raises(ValueError):
        score_signature(scoring_method, adata, [], ctrl_size=ctrl_size, score_name=score_name)
    with pytest.raises(ValueError):
        score_signature(scoring_method, adata, set(), ctrl_size=ctrl_size, score_name=score_name)
    # test correct types of signature
    try:
        score_signature(scoring_method, adata, gene_list, ctrl_size=ctrl_size, score_name=score_name)
        score_signature(scoring_method, adata, set(gene_list), ctrl_size=ctrl_size, score_name=score_name)
    except:
        pytest.fail('Unexpected Error when scoring for a valid gene list or set.')
    # test wrong type of signature
    with pytest.raises(ValueError):
        score_signature(scoring_method, adata, {'signature': 'this is not a signature'}, ctrl_size=ctrl_size,
                        score_name=score_name)
    # test scoring one gene is possible
    try:
        score_signature(scoring_method, adata, gene_list[0], ctrl_size=ctrl_size, score_name=score_name)
    except:
        pytest.fail('Unexpected Error when scoring for single signature gene.')
    assert np.all(adata.obs[score_name] == 0), f'Unexpected scores expected all equal to zero, got ' \
                                               f'{adata.obs[score_name]}'
    # test entering a single non existing gene
    with pytest.raises(ValueError):
        score_signature(scoring_method, adata, 'ThisGeneDoesNotExist', ctrl_size=ctrl_size,
                        score_name=score_name)

    # test if warning is thrown if signature contains duplicated values
    with pytest.warns(match=r'The passed gene_list contains duplicated genes: .*' + gene_list[0] + r'.*'):
        score_signature(scoring_method, adata, [gene_list[0]] + gene_list, ctrl_size=ctrl_size, score_name=score_name)


def test_valid_control_sets_possible(adata, gene_list, scoring_method, ctrl_size, score_name):
    # test ctrl_size larger than reference pool
    #  -> signature is so large no valid control sets can be created
    with pytest.raises(ValueError):
        score_signature(scoring_method, adata, adata.var_names.tolist(), ctrl_size=ctrl_size, score_name=score_name)
    #  -> ctrl_size too large
    with pytest.raises(ValueError):
        score_signature(scoring_method, adata, gene_list, ctrl_size=100, score_name=score_name)
    #  -> ctrl_size cannot be equal to zero
    with pytest.raises(ValueError):
        score_signature(scoring_method, adata, gene_list, ctrl_size=0, score_name=score_name)
    # test wrong input for ctrl_size
    with pytest.raises(Exception):
        score_signature(scoring_method, adata, gene_list, ctrl_size='0', score_name=score_name)
    with pytest.raises(Exception):
        score_signature(scoring_method, adata, gene_list, ctrl_size=0.1, score_name=score_name)

    adata[:, ['Gene_36', 'Gene_99']].X = 2
    # test signature genes removed from signature if too close to the right boundary
    new_gene_list = score_signature(scoring_method, adata, ['Gene_98', 'Gene_36', 'Gene_99'], ctrl_size=ctrl_size,
                                    score_name=score_name, remove_genes_with_invalid_control_set=True,
                                    return_gene_list=True)
    assert isinstance(new_gene_list, list) and len(new_gene_list) == 1 \
           and set(new_gene_list) != set(['Gene_98', 'Gene_36', 'Gene_99'])
    new_gene_list = score_signature(scoring_method, adata, ['Gene_98', 'Gene_36', 'Gene_99'], ctrl_size=ctrl_size,
                                    score_name=score_name, remove_genes_with_invalid_control_set=False,
                                    return_gene_list=True)
    assert isinstance(new_gene_list, list) and set(new_gene_list) == set(['Gene_98', 'Gene_36', 'Gene_99'])


def test_return_control_genes(adata, gene_list, scoring_method, ctrl_size, score_name):
    control_genes, new_gene_list = score_signature(scoring_method, adata, gene_list, ctrl_size=ctrl_size,
                                                   score_name=score_name, return_control_genes=True,
                                                   return_gene_list=True)
    assert isinstance(control_genes, list) and len(control_genes) == len(new_gene_list)
    assert np.all([isinstance(x, list) and np.all([isinstance(y, str) for y in x]) for x in control_genes])


def test_gene_pool(adata, gene_list, scoring_method, ctrl_size, score_name):
    # gene pool contains names not in data and are thus ignored
    with pytest.warns():
        score_signature(scoring_method, adata, gene_list, ctrl_size=ctrl_size, score_name=score_name,
                        gene_pool=(adata.var_names.tolist() + ['ThisGeneDoesNotExist1', 'ThisGeneDoesNotExist2']))
    with pytest.raises(Exception):
        score_signature(scoring_method, adata, gene_list, ctrl_size=ctrl_size, score_name=score_name,
                        gene_pool=adata.var_names.tolist()[0:9])
    # pass empty gene pool
    with pytest.raises(Exception):
        score_signature(scoring_method, adata, gene_list, ctrl_size=ctrl_size, score_name=score_name,
                        gene_pool=[])
