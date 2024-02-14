# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import torch
from sklearn.feature_selection import mutual_info_classif


def smi_estimator(
    X: torch.Tensor,
    y: torch.Tensor,
    n_estimator: int = 100,
    n_neighbors: int = 3,
    random_state: int = 0,
    use_max: bool = False,
) -> float:
    """

    Empirically, for our architectures, we observed that the mutual information
    between a random projection of the features and the labels has:
    - A mean of 4.5e-3
    - A standard deviation of 5.5e-3

    Thus, with n=2000 and model it with a normal law we obtain a margin of about:
    (1.96 x 5.5e-3)/sqrt(2000) = 2.4e-4   (≈5% of the mean)


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
    if not use_max:
        smi = sum(smi_per_direction) / len(smi_per_direction)
    else:
        smi = max(smi_per_direction)

    # Output
    return smi
