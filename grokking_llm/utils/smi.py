# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import typing as t

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm


def smi_estimator(
    X: torch.Tensor,
    y: torch.Tensor,
    smi_quantile: float = 0.99,
    n_estimator: int = 100,
    n_neighbors: int = 3,
    random_state: int = 0,
) -> t.Tuple[float, float, float]:
    """

    Empirically, for our architectures, we observed that the mutual information
    between a random projection of the features and the labels has:
    - A mean of 4.5e-3
    - A standard deviation of 5.5e-3

    We observe that it cannot be modelled as a normal distribution. However, we observed that:
    - With 2000 estimators, the mean SMI varied of about 3% across simulations
    - With 2000 estimators, tne median SMI varied of about 14% across simulations
    - With 2000 estimators, tne maximal SMI varied of about 7% across simulations
    - With 2000 estimators, tne 99% quantile SMI varied of about 3% across simulations

    Thus, using 2000 estimators provides sufficient stability to estimate the mean SMI and the 99% quantile.


    Implements eq. 6 in https://arxiv.org/abs/2110.05279v2 for categorical data.
    """

    # Dimensions
    n_samples, dim_x = X.size()
    if len(y.size()) != 1 or y.size(0) != n_samples:
        raise ValueError(
            f"Incompatible shapes: X is {X.size()} and y is {y.size()}, whereas we expected (n_samples, dim_x) for X and (n_samples,) for y."
        )

    # Sampling theta uniformly on the hypersphere
    # Theta will have size (n_estimator, dim_x), each row uniformly sampled on the dim_x-hypersphere
    # Reference: Marsaglia, G. (1972). “Choosing a Point from the Surface of a Sphere”...
    # ... Annals of Mathematical Statistics. 43 (2): 645-646.
    theta = torch.normal(
        mean=torch.zeros((n_estimator, dim_x)),
        std=torch.ones((n_estimator, dim_x)),
    )
    theta /= torch.norm(theta, p=2, dim=1)[:, None]

    # Projected features
    # dim_x is projected in a 1-dimensional space, so the feature tensor has size (n_samples, n_estimator)
    projected_features = (X @ theta.T).view(n_samples, n_estimator)

    # Mean smi
    smi_per_direction = mutual_info_classif(
        projected_features, y, n_neighbors=n_neighbors, random_state=random_state
    )
    smi_mean = sum(smi_per_direction) / len(smi_per_direction)
    smi_max = np.quantile(smi_per_direction, smi_quantile)
    smi_min = np.quantile(smi_per_direction, 1 - smi_quantile)

    # Output
    return (smi_mean, smi_max, smi_min)


def p_smi_estimator(
    X: torch.Tensor,
    y: torch.Tensor,
    smi_quantile: float = 0.99,
    n_estimator: int = 100,
    return_std: bool = False,
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimates the (sample-wise) Pointsize Sliced Mutual Information [1]

    Empirically, for our architectures, we observed that the mutual information
    between a random projection of the features and the labels has:
    - A mean of 4.5e-3
    - A standard deviation of 5.5e-3

    We observe that it cannot be modelled as a normal distribution. However, we observed that:
    - With 2000 estimators, the mean SMI varied of about 3% across simulations
    - With 2000 estimators, tne median SMI varied of about 14% across simulations
    - With 2000 estimators, tne maximal SMI varied of about 7% across simulations
    - With 2000 estimators, tne 99% quantile SMI varied of about 3% across simulations

    Thus, using 2000 estimators provides sufficient stability to estimate the mean SMI and the 99% quantile.

    [1] DOI: 10.1109/ISIT54713.2023.10207010
    [2] ArXiv: 2110.05279v2
    """

    # Dimensions
    n_samples, dim_x = X.size()
    if len(y.size()) != 1 or y.size(0) != n_samples:
        raise ValueError(
            f"Incompatible shapes: X is {X.size()} and y is {y.size()}, whereas we expected (n_samples, dim_x) for X and (n_samples,) for y."
        )

    # Sampling theta uniformly on the hypersphere
    # Theta will have size (n_estimator, dim_x), each row uniformly sampled on the dim_x-hypersphere
    # Reference: Marsaglia, G. (1972). “Choosing a Point from the Surface of a Sphere”...
    # ... Annals of Mathematical Statistics. 43 (2): 645-646.
    theta = torch.normal(
        mean=torch.zeros((n_estimator, dim_x)),
        std=torch.ones((n_estimator, dim_x)),
    )
    theta /= torch.norm(theta, p=2, dim=1)[:, None]

    # Projected features
    # dim_x is projected in a 1-dimensional space, so the feature tensor has size (n_samples, n_estimator)
    projected_features = (X @ theta.T).view(n_samples, n_estimator)

    # Index per possible categorical label
    label_to_sample_list = collections.defaultdict(list)
    for sample_idx, label in enumerate(y.tolist()):
        label_to_sample_list[label].append(sample_idx)

    # Label frequencies
    label_to_frequency = {
        possible_label: len(sample_list) / n_samples
        for possible_label, sample_list in label_to_sample_list.items()
    }

    # Checking that each label has at least two samples
    for possible_label, sample_list in label_to_sample_list.items():
        if len(sample_list) < 2:
            raise RuntimeError(
                f"We cannot compute pointwise SMI: the following label has less than 2 samples: {possible_label}"
            )

    # Mean and std per label, per direction
    # estimated_mean[label] is a list of length `n_estimator`
    # The k-th element corresponds to the estimated mean of the k-th projected feature
    # for samples that have label `label`
    estimated_mean = dict()
    estimated_std = dict()
    for possible_label, sample_list in label_to_sample_list.items():
        features = projected_features[sample_list, :]
        estimated_mean[possible_label] = features.mean(axis=0)
        estimated_std[possible_label] = features.std(axis=0)

    # Pointise Mutual Information
    pmi_numerator = torch.zeros_like(projected_features)
    pmi_denominator = torch.zeros_like(projected_features)
    for possible_label, sample_list in label_to_sample_list.items():

        # Numerator : we estimage p(projected_features | y)
        pmi_numerator[sample_list, :] = norm_pdf(
            estimated_mean[possible_label],
            estimated_std[possible_label],
            projected_features[sample_list, :],
        )

        # Denominator: we use the formula of total probabilities
        # p(projected_features) = sum(p(y)*p(projected_features|y))
        label_freq = label_to_frequency[possible_label]
        pmi_denominator += label_freq * norm_pdf(
            estimated_mean[possible_label],
            estimated_std[possible_label],
            projected_features,
        )

    # PMI array: shape n_samples, n_estimator
    # At index (i, j), if the j-th projected features of the i-th sample is x_ij
    # we have pmi[i,j] = log(p(x_ij|y_i) | p(x_ij))
    pmi = np.log(pmi_numerator / pmi_denominator)

    # Mean smi
    smi_mean = np.mean(pmi.numpy(), axis=1)
    smi_max = np.quantile(pmi, smi_quantile, axis=1)
    smi_min = np.quantile(pmi, 1 - smi_quantile, axis=1)
    smi_std = np.std(pmi.numpy(), axis=1)

    # Output
    if not return_std:
        return (smi_mean, smi_max, smi_min)
    else:
        return (smi_mean, smi_max, smi_min, smi_std)


def norm_pdf(mean, std, x):
    return (1 / std / np.sqrt(2 * np.pi)) * np.exp(-1 * (x - mean) ** 2 / 2 / std / std)


def get_p_smi_containers(
    metrics_df: pd.DataFrame,
) -> t.Tuple[
    t.Dict[int, t.Dict[int, t.Dict[int, float]]],
    t.Dict[int, t.Dict[int, t.Dict[int, float]]],
    t.Dict[int, t.Dict[int, t.Dict[int, float]]],
]:
    """
    `metrics_df` is supposed to be the result of metrics.load_metrics_df()
    where `metrics` is an instance either of PSmiMetrics or PSmiStatic or PSmiSlopeMetrics, PSmiStdMetrics
    If `metrics` is an instance of PSmiStdMetrics, it returns (p_smi_std_per_checkpoint_per_layer_per_idx, None, None)

    Returns (
        p_smi_mean_per_checkpoint_per_layer_per_idx,
        p_smi_max_per_checkpoint_per_layer_per_idx,
        p_smi_min_per_checkpoint_per_layer_per_idx,
    ).
    """

    # Checkpoints
    checkpoints = metrics_df.iloc[:, 0].tolist()

    # Init containers
    count_mean = 0
    p_smi_mean_checkpoint_layer_idx = {
        chk: collections.defaultdict(dict) for chk in checkpoints
    }
    count_max = 0
    p_smi_max_checkpoint_layer_idx = {
        chk: collections.defaultdict(dict) for chk in checkpoints
    }
    count_min = 0
    p_smi_min_checkpoint_layer_idx = {
        chk: collections.defaultdict(dict) for chk in checkpoints
    }
    count_std = 0
    p_smi_std_checkpoint_layer_idx = {
        chk: collections.defaultdict(dict) for chk in checkpoints
    }

    # Getting checkpoint idx
    for row_idx, chk in enumerate(tqdm(checkpoints)):
        for col_idx, col_name in enumerate(metrics_df.columns):

            # "checkpoint" column
            if col_idx == 0 or col_name == "epoch":
                continue

            content = metrics_df.iloc[row_idx, col_idx]

            # Container
            if "mean_" in col_name:
                container = p_smi_mean_checkpoint_layer_idx
                if "slope" not in col_name:
                    layer_idx = col_name[len("mean_psmi_") :]
                else:
                    layer_idx = col_name[len("mean_psmi_slope_") :]
                count_mean += 1
            elif "max_" in col_name:
                container = p_smi_max_checkpoint_layer_idx
                if "slope" not in col_name:
                    layer_idx = col_name[len("max_psmi_") :]
                else:
                    layer_idx = col_name[len("max_psmi_slope_") :]
                count_max += 1
            elif "min_" in col_name:
                container = p_smi_min_checkpoint_layer_idx
                if "slope" not in col_name:
                    layer_idx = col_name[len("min_psmi_") :]
                else:
                    layer_idx = col_name[len("min_psmi_slope_") :]
                count_min += 1
            elif "std_" in col_name:
                container = p_smi_std_checkpoint_layer_idx
                if "slope" not in col_name:
                    layer_idx = col_name[len("std_psmi_") :]
                else:
                    layer_idx = col_name[len("std_psmi_slope_") :]
                count_std += 1
            else:
                raise ValueError(f"Name: {col_name}")

            # Layer, idx
            layer = int(layer_idx.split("_")[0])
            idx = int(layer_idx.split("_")[1])

            container[chk][layer][idx] = content

    # Sanity check
    if count_std > 0 and (count_mean + count_max + count_min) > 0:
        raise RuntimeError(
            "You should not have both STD and MEAN/MAX/MIN psmi within the same dataframe"
        )

    if not (count_mean == count_max == count_min):
        raise RuntimeError(
            f"Different count of MEAN/MAX/MIN psmi columns: {count_mean}, {count_max}, {count_min}"
        )

    # Output
    if count_std == 0:
        return (
            p_smi_mean_checkpoint_layer_idx,
            p_smi_max_checkpoint_layer_idx,
            p_smi_min_checkpoint_layer_idx,
        )
    else:
        return (
            p_smi_std_checkpoint_layer_idx,
            None,
            None,
        )
