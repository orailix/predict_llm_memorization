# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif


def smi_estimator(
    X: torch.Tensor,
    y: torch.Tensor,
    smi_quantile: float = 0.99,
    n_estimator: int = 100,
    n_neighbors: int = 3,
    random_state: int = 0,
) -> float:
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

    # Output
    return (smi_mean, smi_max)
