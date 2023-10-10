import logging
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from pandas import DataFrame, Series
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn import svm
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score, balanced_accuracy_score


def _check_label_col_and_get_GT(adata_obs: DataFrame, label_col: str = "healthy", label_of_interest: str = "unhealthy") -> Tuple[str, str, Series]:
    """
    The method checks for the existence of the given label column and positive class label. It then gets the ground
    truth vector, required for evaluation of the performance metrics.

    Args:
        adata_obs: '.obs' part of an AnnData object.
        label_col: Name of column containing cell type labels, i.e., ground truth for performance measures.
        label_of_interest: Name of label corresponding to the positive class, i.e., label excepted to be associated
            with high scores.

    Returns:
        The method returns the label of the positive class, the negative class, and the ground truth vector.
    """
    if label_col not in adata_obs:
        raise ValueError(f"Indicated label column {label_col} does not exist in adata_obs")
    if len(adata_obs[label_col].cat.categories) < 2:
        raise ValueError(f'Need at least two categories to measure performance.'
                         f'Available category {adata_obs[label_col].cat.categories.tolist()}')
    if label_of_interest not in adata_obs[label_col].cat.categories:
        raise ValueError(
            f"Indicated label for cells with higher scores {label_of_interest} is not a categorie of adata_obs["
            f"label_col]. Available categories {adata_obs[label_col].cat.categories.tolist()}"
        )

    # if two categories are available choose the other one as group 2
    # if more than two categories are available choose all others are group 2
    if len(adata_obs[label_col].cat.categories) == 2:
        label_G1 = label_of_interest
        pos_label_G1 = (
            np.argwhere(adata_obs[label_col].cat.categories == label_of_interest)
        ).item()
        label_G2 = adata_obs[label_col].cat.categories.tolist()[1 - pos_label_G1]
        gt = adata_obs[label_col].copy()
    else:
        label_G1 = label_of_interest
        label_G2 = 'not ' + label_of_interest
        curr_gt = adata_obs[label_col].copy().astype(str)
        curr_gt.loc[curr_gt != label_of_interest] = label_G2
        gt = curr_gt.astype('category')

    return label_G1, label_G2, gt


def _check_scoring_names(adata_obs: DataFrame, scoring_names: List[str]) -> List[str]:
    """
    The method checks for the existence of the column names associated with signature scores.

    Args:
        adata_obs: '.obs' part of an AnnData object.
        scoring_names: Names of signature scores in '.obs' to be evaluated.

    Returns:
        Filtered list of signature score column names.

    Raises:
        AssertionError
            If scoring_names is not of type list and contains more than one element.
    """
    if isinstance(scoring_names, str):
        scoring_names = [scoring_names]

    assert isinstance(scoring_names, list) and len(scoring_names) > 0, (
        f"Need to pass a list of signature scoring "
        f"method names in scoring_names. Need at least"
        f"one name. "
    )

    # check data availability for scoring_names
    scoring_names_to_ignore = list(
        set(scoring_names).difference(set(adata_obs.columns.tolist()))
    )
    scoring_names = list(
        set(scoring_names).intersection(set(adata_obs.columns.tolist()))
    )
    if scoring_names_to_ignore:
        warnings.warn(
            f"Passed scoring_names contains values ({scoring_names_to_ignore}) that are not available in the data. "
            f"These methods will not be tested for. "
        )
    return scoring_names


def get_AUC_and_F1_performance(
        adata_obs: DataFrame,
        scoring_names: List[str],
        label_col: str = "healthy",
        label_of_interest: str = "unhealthy",
        old_df: Optional[DataFrame] = None,
        sample_id: Optional[str] = None,
        store_data_path: Optional[str] = None,
        save: bool = False,
        log: Optional[str] = None
) -> DataFrame:
    """
    Compute the AUCROC and classification metrics for signature scores for given labels.

    Args:
        adata_obs: '.obs' part of an AnnData object.
        scoring_names: Names of signature scores in '.obs' to be evaluated.
        label_col: Name of column containing cell type labels, i.e., ground truth for performance measures.
        label_of_interest: Name of label corresponding to the positive class, i.e., label excepted to be associated
            with high scores.
        old_df: Existing DataFrame that should be extended.
        sample_id: Optional sample id name that can be added to a column
        store_data_path: Optional path to location in which performance metrics can be stored
        save: Indicates whether performance metrics should be stored as '.csv'
        log: Name of logger

    Returns:
        DataFrame with the evaluated performance metrics
    """
    scoring_names = _check_scoring_names(adata_obs, scoring_names)

    label_G1, label_G2, curr_GT = _check_label_col_and_get_GT(adata_obs, label_col, label_of_interest)
    pos_label_G1 = (np.argwhere(curr_GT.cat.categories == label_G1)).item()

    if old_df is not None:
        df = old_df
    else:
        df = pd.DataFrame()

    new_rows = []
    for score in scoring_names:
        t_score = (
            score.replace(" ", "_")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .lower()
        )
        curr_score = adata_obs[score]

        # labels of group with smaller scores need to be lexicographically lower than the oes of the second group
        cp_group = curr_GT.copy()

        if pos_label_G1 == 0:
            # swap labels
            cp_group = cp_group.map({label_G1: label_G2, label_G2: label_G1})

        auc = roc_auc_score(cp_group, curr_score)
        new_rows.append({
            "Scoring method": t_score,
            "Test method": 'auc',
            "Statistic": auc,
            "pvalue": np.nan,
        })

        # compute F1 for 0 threshold
        scores_hard_labels = curr_GT.copy()
        scores_hard_labels[curr_score > 0] = label_G1
        scores_hard_labels[curr_score <= 0] = label_G2

        f1_val = f1_score(curr_GT, scores_hard_labels, pos_label=label_G1)

        new_rows.append({
            "Scoring method": t_score,
            "Test method": 'f1',
            "Statistic": f1_val,
            "pvalue": np.nan,
        })

        jacc_val = jaccard_score(curr_GT, scores_hard_labels, pos_label=label_G1)

        new_rows.append({
            "Scoring method": t_score,
            "Test method": 'jaccard',
            "Statistic": jacc_val,
            "pvalue": np.nan,
        })

        bal_acc_val = balanced_accuracy_score(curr_GT, scores_hard_labels)

        new_rows.append({
            "Scoring method": t_score,
            "Test method": 'balanced_accuracy',
            "Statistic": bal_acc_val,
            "pvalue": np.nan,
        })

    new_df = pd.DataFrame.from_records(new_rows)
    new_df['scoring_for'] = label_of_interest
    if sample_id:
        new_df['sample_id'] = sample_id
    df = pd.concat([df, new_df], ignore_index=True)

    if save:
        tests_scores_fn = os.path.join(
            store_data_path, f"evaluation_of_scoring_methods.csv"
        )
        df.to_csv(tests_scores_fn, index=False)
        log.info(f"Stored test statistic to {tests_scores_fn}")

    return df


def get_test_statistics(
        adata: AnnData,
        scoring_names: List[str],
        test_method: str = "kstest",
        alternative: str = "greater",
        label_col: str = "Group",
        label_whsc: str = "Group1",
        old_df: Optional[DataFrame] = None,
        store_data_path: Optional[str] = None,
        save: bool = False,
        log: Optional[str] = None,
    ) -> DataFrame:
    """
    This function computes for each indicated scoring method the performance of the scores. It applies a two sample test based on the passed method. It can store the data in a desired folder.

    Args:
        adata: AnnData object containing the preprocessed (log-normalized) gene expression data.
        scoring_names: Column names of signature scores in '.obs' attribute.
        test_method: Selected test to conduct on scores. Available tests: kstest, mannwhitneyu, auc, or auc-dist
        alternative: Alternative of two-sample test.
        label_col: Label column in '.obs' of 'adata'.
        label_whsc: Label of positive class, i.e., label associated with high scores.
        old_df: Existing DataFrame that should be extended.
        store_data_path: Path to location to store performance files.
        save: Indicates whether performance measurements should be stored.
        log: Name of logger.

    Returns:
        A dataframe containing the test results for all names in scoring_names. It contains the following columns
        [Scoring method', 'Test method', 'Statistic', 'pvalue'].

    Raises:
        ValueError
            1. If label_col is not available in adata.obs.
            2. If label_whsc os not a category in label_col.
            3. If label_col contains not two categories.
    """
    if log is None:
        log = logging
    else:
        log = logging.getLogger(log)

    assert isinstance(scoring_names, list) and len(scoring_names) > 0, (
        f"Need to pass a list of signature scoring "
        f"method names in scoring_names. Need at least"
        f"one name. "
    )

    # check if label column exists and is in the correct order
    if label_col not in adata.obs:
        raise ValueError(
            f"Indicated label column {label_col} does not exist in adata.obs"
        )
    if label_whsc not in adata.obs[label_col].cat.categories:
        raise ValueError(
            f"Indicated label for cells with higher scores {label_whsc} is not a category of adata.obs["
            f"label_col]. Available categories {adata.obs[label_col].cat.categories.tolist()}"
        )
    if len(adata.obs[label_col].cat.categories) != 2:
        raise ValueError(
            f"At the moment we accept only two categories: {adata.obs[label_col].cat.categories.tolist()}"
        )

    # get group labels
    label_G1 = label_whsc
    pos_label_G1 = (
        np.argwhere(adata.obs[label_col].cat.categories == label_whsc)
    ).item()
    label_G2 = adata.obs[label_col].cat.categories.tolist()[1 - pos_label_G1]

    # check data availability for scoring_names
    scoring_names_to_ignore = list(
        set(scoring_names).difference(set(adata.obs.columns.tolist()))
    )
    scoring_names = list(
        set(scoring_names).intersection(set(adata.obs.columns.tolist()))
    )
    if scoring_names_to_ignore:
        warnings.warn(
            f"Passed scoring_names contains values ({scoring_names_to_ignore}) that are not available in the data. "
            f"These methods will not be tested for. "
        )

    if test_method == "kstest":
        test_function = ks_2samp
    elif test_method == "mannwhitneyu":
        test_function = mannwhitneyu
    elif test_method == "auc" or test_method == "auc-dist":
        test_function = roc_auc_score
    else:
        raise ValueError(
            f"Unknown test_method {test_method}. Available methods:[kstest, mannwhitneyu, auc, auc-dist]"
        )
    if old_df is not None:
        df = old_df
    else:
        df = pd.DataFrame()

    idx_group1 = adata.obs[label_col] == label_G1
    idx_group2 = adata.obs[label_col] == label_G2
    new_rows = []
    for score in scoring_names:
        t_score = (
            score.replace(" ", "_")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .lower()
        )
        curr_score = adata.obs[score]
        if test_method == "auc" or test_method == 'auc-dist':
            # labels of group with smaller scores need to be lexicographically lower than the oes of the second group
            cp_group = adata.obs[label_col].copy()
            if pos_label_G1 == 0:
                # swap labels
                cp_group = cp_group.map({label_G1: label_G2, label_G2: label_G1})

            auc = test_function(cp_group, curr_score)

            if test_method == 'auc-dist':
                # compute separating hyperplane
                X = np.transpose(np.vstack((curr_score, np.ones_like(curr_score))))
                clf = svm.SVC(kernel='linear')
                clf.fit(X, adata.obs[label_col])
                w = clf.coef_[0]
                x_0 = -clf.intercept_[0] / w[0]
                auc = (auc + (1 - 0.5 * (np.tanh(abs(x_0))))) / 2

            new_row = {
                "Scoring method": t_score,
                "Test method": test_method,
                "Statistic": auc,
                "pvalue": np.nan,
            }
        else:
            group1_scores = curr_score[idx_group1]
            group2_scores = curr_score[idx_group2]

            test_result = test_function(
                group1_scores, group2_scores, alternative=alternative
            )
            new_row = {
                "Scoring method": t_score,
                "Test method": test_method,
                "Statistic": test_result.statistic,
                "pvalue": test_result.pvalue,
            }
        # df = df.append(new_row, ignore_index=True)
        new_rows.append(new_row)

    df = pd.concat([df, pd.DataFrame.from_records(new_rows)], ignore_index=True)

    if save:
        tests_scores_fn = os.path.join(
            store_data_path, f"evaluation_of_scoring_methods.csv"
        )
        df.to_csv(tests_scores_fn, index=False)
        log.info(f"Stored test statistic to {tests_scores_fn}")

    return df

