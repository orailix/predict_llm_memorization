# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import torch
from sklearn.decomposition import PCA


def mahalanobis_distance(
    train_features: torch.Tensor,
    target_features: torch.Tensor,
    hidden_dim: t.Optional[int] = 500,
) -> torch.Tensor:
    """Computes the Mahalanobis distance between each point of
    the target features to the distribution of the training features.

    Before computing the Mahalanobis distance, we first project the points into a
    space of lower dimension using a PCA which is fitted using training points.
    Its dimension is controled with the "hidden_dim" parameter.

    Args:
        - train_features: A tensor of size (n_train, n_features)
        - target_features: A tensor of size (n_target, n_features)
        - hidden_dim: An int representing the dimension of the intermediate space.
        Use `None` to avoid the PCA projection.

    Returns:
        - A tensor of size (n_target, )
    """

    # Convert to float 64
    train_features = train_features.to(torch.float64)
    target_features = target_features.to(torch.float64)

    # Hidden dim ?
    if hidden_dim is not None:
        reductor = PCA(n_components=100).fit(train_features)
        train_features = torch.Tensor(reductor.transform(train_features)).to(
            torch.float64
        )
        target_features = torch.Tensor(reductor.transform(target_features)).to(
            torch.float64
        )

    # Estimating mean and covariance
    train_mean = train_features.mean(dim=0)
    train_features_norm = train_features - train_mean  # Shape (n_train, n_features)
    train_sigma = (
        train_features_norm.T @ train_features_norm
    ) / train_features_norm.size(
        0
    )  # Shape (n_features, n_features)
    cov_inverse = torch.linalg.inv(train_sigma)  # Shape (n_features, n_features)

    # Computing distance
    target_features_norm = target_features - train_mean  # Shape (n_target, n_features)
    distances: torch.Tensor = (
        target_features_norm @ cov_inverse @ target_features_norm.T
    ).diag()  # Shape (n_target, )
    distances = torch.sqrt(distances)

    return distances
