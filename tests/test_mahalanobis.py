# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import torch

from grokking_llm.utils import mahalanobis_distance


def test_mahalanobis_distance():

    train_features = torch.randn(1000, 4096)
    target_features_1 = torch.randn(50, 4096)
    target_features_2 = torch.randn(50, 4096) + torch.randn(1, 4096)

    dst_1 = mahalanobis_distance(train_features, target_features_1)
    dst_2 = mahalanobis_distance(train_features, target_features_2)

    assert dst_1.mean() < dst_2.mean()
    assert dst_1.mean() + dst_1.std() < dst_2.mean() - dst_2.std()
