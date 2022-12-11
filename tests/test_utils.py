import anndata as ad
import pytest

from signaturescoring.utils.utils import check_signature_genes, get_data_for_gene_pool


def test_gene_list(adata, gene_list):
    # test empty signature
    with pytest.raises(ValueError):
        check_signature_genes(adata.var_names.tolist(), [])
    with pytest.raises(ValueError):
        check_signature_genes(adata.var_names.tolist(), set())

    # test wrong type of signature
    with pytest.raises(ValueError):
        check_signature_genes(adata.var_names.tolist(), {'signature': 'this is not a signature'})

    # test entering a single non existing gene
    with pytest.raises(ValueError):
        check_signature_genes(adata.var_names.tolist(), 'ThisGeneDoesNotExist')

    # test if warning is thrown if signature contains duplicated values
    with pytest.warns(match=r'The passed gene_list contains duplicated genes: .*' + gene_list[0] + r'.*'):
        check_signature_genes(adata.var_names.tolist(), [gene_list[0]] + gene_list)

    # test return type
    assert isinstance(check_signature_genes(adata.var_names.tolist(), gene_list, return_type=list), list)
    assert isinstance(check_signature_genes(adata.var_names.tolist(), gene_list, return_type=set), set)
    with pytest.raises(ValueError):
        check_signature_genes(adata.var_names.tolist(), gene_list, return_type=dict)


def test_gene_pool(adata, gene_list, ctrl_size):
    # gene pool contains names not in data and are thus ignored
    with pytest.warns(UserWarning):
        get_data_for_gene_pool(adata,
                               gene_pool=(adata.var_names.tolist() + ['ThisGeneDoesNotExist1',
                                                                      'ThisGeneDoesNotExist2']),
                               gene_list=gene_list,
                               check_gene_list=False)

    with pytest.raises(Exception):
        get_data_for_gene_pool(adata,
                               gene_pool=adata.var_names.tolist()[0:9],
                               gene_list=gene_list,
                               check_gene_list=False,
                               ctrl_size=ctrl_size)
    # pass empty gene pool
    with pytest.raises(Exception):
        get_data_for_gene_pool(adata,
                               gene_pool=[],
                               gene_list=gene_list,
                               check_gene_list=False)
    # Check if returns the rigth types
    res_adata, res_gene_pool = get_data_for_gene_pool(adata,
                                                      gene_pool=(adata.var_names.tolist() +
                                                                 ['ThisGeneDoesNotExist1', 'ThisGeneDoesNotExist2']),
                                                      gene_list=gene_list,
                                                      check_gene_list=False)
    assert isinstance(res_adata, ad.AnnData) and isinstance(res_gene_pool, list)
