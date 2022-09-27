import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score, balanced_accuracy_score


def check_label_col_and_get_GT(adata_obs: DataFrame, label_col: str = "healthy", label_of_interest: str = "unhealthy"):
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


def check_scoring_names(adata_obs: DataFrame, scoring_names: List[str]) -> List[str]:
    """
    The method checks for the existence of the column names associated with signature scores.
    Args:
        adata_obs: '.obs' part of an AnnData object.
        scoring_names: Names of signature scores in '.obs' to be evaluated.

    Returns:
        Filtered list of signature score column names.
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
    scoring_names = check_scoring_names(adata_obs, scoring_names)

    label_G1, label_G2, curr_GT = check_label_col_and_get_GT(adata_obs, label_col, label_of_interest)
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

        # compute separating hyperplane
        # X = np.transpose(np.vstack((curr_score, np.ones_like(curr_score))))
        # clf = svm.SVC(kernel='linear')
        # clf.fit(X, adata_obs[label_col])
        # w = clf.coef_[0]
        # x_0 = -clf.intercept_[0] / w[0]
        # auc_dist = (auc + (1 - 0.5 * (np.tanh(abs(x_0))))) / 2

        # new_rows.append({
        #    "Scoring method": t_score,
        #    "Test method": 'auc-dist',
        #    "Statistic": auc_dist,
        #    "pvalue": np.nan,
        # })

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
