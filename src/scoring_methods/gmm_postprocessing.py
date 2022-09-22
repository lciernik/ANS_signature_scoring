import os
import sys
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.append('../..')
from src.utils.utils import commonPrefix


def check_score_names(adata: AnnData, score_names: List[str]):
    """
    Asserts that names associated with score columns exist in the data.
    Args:
        adata: AnnData object containing the gene expression data.
        score_names: Names of score columns

    Returns:
        None

    Raises:
        Assertion error if any value in `score_names` is not contained in `adata`
    """
    assert isinstance(score_names, list) and len(score_names) > 0, \
        f'score_names needs to be of type list. We need at least one score_name'

    assert all([x in adata.obs.columns for x in score_names]), f'score_names conatains names not ' \
                                                               f'corresponding to columns in adata.obs'


class GMMPostprocessor:
    """
    The GMMPostprocessor class is used to correct for incomparable score ranges in gene signature scoring.
    If fits a Gaussian Mixture Model on gene signature scores and assigns clusters to signatures.

    Attributes:
        n_components: Defines the number of clusters we expect in the Gaussian Mixture Model. For postprocessing gene
            expression signatures we use n_components=#signatures or n_components=(#signatures+1).
        gmm: Corresponds to the GMM used for postprocessing.
    """

    def __init__(self, n_components: int = 3, covariance_type: str = 'full', init_params: str = 'k-means++',
                 n_init: int = 30):
        """
        Args:
            n_components: Number of clusters we expect in the Gaussian Mixture Model.
            covariance_type: The type of covariance used in GMM. Available methods 'full', 'tied', 'diag', 'spherical'.
            init_params: Method to initialize parameters in GMM. Available methods 'kmeans', 'k-means++', 'random',
                'random_from_data'
            n_init: Number of initializations done.
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.init_params = init_params
        self.n_init = n_init

        self.gmm = GaussianMixture(n_components=n_components,
                                   covariance_type=covariance_type,
                                   init_params=init_params,
                                   n_init=n_init,
                                   random_state=0)

    def fit_and_predict(self, adata: AnnData, score_names: List[str], store_name: Optional[str] = None,
                        inplace: bool = True) -> Union[str, List[str], Optional[DataFrame]]:
        """
        The method fits previously initialized GMM on signature scores.
        Args:
            adata: AnnData object containing the gene expression data.
            score_names: Name of signature scores columns on which the GMM is fit.
            store_name: Prefix of new columns with probabilities
            inplace: If probabilities are stored in `adata` or in a new pandas DataFrame

        Returns:
            If 'inplace=True', the names of the new columns are returned.
            If 'inplace=False', the names of the new columns and the DataFrame containing the cluster probabilities are
                returned.
        """
        if store_name is None:
            store_name = commonPrefix(score_names, 0, len(score_names) - 1)
        print(f'GMM model for {store_name} scores.')
        check_score_names(adata, score_names)
        curr_data = adata.obs[score_names].copy()
        print(f'> standardize data')
        curr_data = StandardScaler().fit_transform(curr_data)
        print(f'> fit and predict probabilities')
        gm_pred = self.gmm.fit_predict(curr_data)
        gm_proba = self.gmm.predict_proba(curr_data)

        store_name_pred = store_name + '_GMM_pred'
        store_names_proba = [(store_name + f'_{x}_GMM_proba') for x in range(self.n_components)]

        if inplace:
            adata.obs[store_name_pred] = gm_pred
            adata.obs[store_names_proba] = gm_proba
            return store_name_pred, store_names_proba, None
        else:
            pred_and_proba_df = pd.DataFrame(np.hstack([gm_pred.reshape(-1, 1), gm_proba]),
                                             index=adata.obs.index,
                                             columns=([store_name_pred] + store_names_proba))
            return store_name_pred, store_names_proba, pred_and_proba_df

    def assign_clusters_to_signatures(self, adata: AnnData, score_names: List[str], gmm_proba_names: List[str],
                                      plot: bool = False, store_plot_path: Optional[str] = None) -> dict:
        """
        The methods computed the assignments of GMM clusters to gene expression signatures by computing the correlation
        of each cluster probabilities to the signatures' scores.

        Args:
            adata: AnnData object containing the gene expression data.
            score_names: Name of signature scores columns.
            gmm_proba_names: Name of GMM cluster probability columns.
            plot: Plot scatterplots of scores and probabilities for each signature and GMM cluster.
            store_plot_path: Path to location where scatterplots should be stored. If None, plots are not stored.

        Returns:
            The assignments of each signature to a cluster from GMM postprocessing.

        """
        check_score_names(adata, score_names + gmm_proba_names)

        signature_group_assignments = {}
        if plot:
            fig, ax = plt.subplots(nrows=len(score_names), ncols=len(gmm_proba_names),
                                   figsize=(len(gmm_proba_names) * 5, len(score_names) * 5))
        for k, sco in enumerate(score_names):
            max_corr = 0
            max_group = None
            for l, group in enumerate(gmm_proba_names):
                corr = scipy.stats.pearsonr(adata.obs[sco], adata.obs[group])
                print(corr, sco, group)
                if plot:
                    ax[k, l].scatter(adata.obs[sco], adata.obs[group])
                    x_label = sco.split('_')[-1]
                    y_label = '_'.join(group.split('_')[-3:])
                    ax[k, l].set_xlabel(x_label)
                    ax[k, l].set_ylabel(y_label)
                    ax[k, l].set_title(f'corr({x_label}, {y_label})=\n{corr}')
                #                 if corr[1] < 0.01 and corr[0] > max_corr:
                if corr[0] > max_corr:
                    max_corr = corr[0]
                    max_group = group
            signature_group_assignments[sco] = max_group
        if len(set(gmm_proba_names).difference(set(signature_group_assignments.values()))) > 0:
            signature_group_assignments['rest'] = set(gmm_proba_names).difference(
                set(signature_group_assignments.values()))
        if plot:
            fig.subplots_adjust(hspace=0.3)
            store_name = commonPrefix(score_names, 0, len(score_names) - 1)
            fig.suptitle(store_name)
            if store_plot_path is not None:
                fig.savefig(os.path.join(store_plot_path, f'{store_name}scores_vs_proba.png'), format='png')

        return signature_group_assignments
