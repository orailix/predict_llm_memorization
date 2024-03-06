import typing as t
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MetaDistribution:
    d: int
    n_cluster: int
    a_pareto_pi: float
    a_pareto_sigma: float
    scale_pareto_sigma: float


@dataclass
class RealDistribution:
    d: int
    n_cluster: int
    pi_0: t.List[int]
    pi_1: t.List[int]
    c_0: t.List[torch.Tensor]
    c_1: t.List[torch.Tensor]
    sigma_0: t.List[float]
    sigma_1: t.List[float]

    @classmethod
    def from_meta(cls, meta: MetaDistribution):

        # Sample pi
        pi_0 = np.random.pareto(meta.a_pareto_pi, size=meta.n_cluster)
        pi_0 /= np.sum(pi_0)
        pi_0 = np.sort(pi_0)[::-1]
        pi_1 = np.random.pareto(meta.a_pareto_pi, size=meta.n_cluster)
        pi_1 /= np.sum(pi_1)
        pi_1 = np.sort(pi_1)[::-1]

        # Sample c
        c_0 = torch.randn(meta.n_cluster, meta.d)
        c_1 = torch.randn(meta.n_cluster, meta.d)

        # Sample sigma
        sigma_0 = meta.scale_pareto_sigma * np.random.pareto(
            meta.a_pareto_sigma, size=meta.n_cluster
        )
        sigma_1 = meta.scale_pareto_sigma * np.random.pareto(
            meta.a_pareto_sigma, size=meta.n_cluster
        )

        # Output
        return cls(
            d=meta.d,
            n_cluster=meta.n_cluster,
            pi_0=pi_0,
            pi_1=pi_1,
            c_0=c_0,
            c_1=c_1,
            sigma_0=sigma_0,
            sigma_1=sigma_1,
        )

    def sample_points(
        self, num_point=1
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        labels = np.random.randint(0, 2, size=(num_point,))
        cluster_idx = []
        features = []
        for l in labels:

            if l == 0:
                pi, c, sigma, cluster_offset = self.pi_0, self.c_0, self.sigma_0, 0
            else:
                pi, c, sigma, cluster_offset = (
                    self.pi_1,
                    self.c_1,
                    self.sigma_1,
                    self.n_cluster,
                )

            new_cluster_idx = np.random.choice(range(self.n_cluster), size=1, p=pi)
            new_features = c[new_cluster_idx] + torch.randn(self.d) * np.sqrt(
                sigma[new_cluster_idx]
            )

            # Appending
            cluster_idx.append(new_cluster_idx + cluster_offset)
            features.append(new_features)

        # Output
        return (
            torch.tensor(labels).long(),
            torch.tensor(cluster_idx),
            torch.cat(features, dim=0).float(),
        )
