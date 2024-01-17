# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import typing as t

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from ..training import TrainingCfg
from .dynamic_metrics_group import DynamicMetricsGroup
from .utils.smi import smi_estimator

LAYERS = [0, 7, 15, 23, 31]


class SmiMetrics(DynamicMetricsGroup):
    """Class used to compute Sliced MUtual Information metrics on the models.

    Performance metrics: (20 in total):
        For layers in [0, 7, 15, 23, 31]:
            For dl in [train_all, train_trl, train_rdl, test]:
                - [{dl}_smi_{layer}]

    Where [dl_smi_k] is the sliced mutual information between:
        - The Values of the attention bloc at layer 0 for token N-3
        - The Label of token N-3, which is the answer of the MCQ
    (for dataloader `dl`)

    For SMI computation, we observed that the mutual information between
    random 1-dimensional projection and the label has a mean of about 4.5e-3
    and a standard deviation of about 5.5e-3.

    Thus, by default we take 2000 estimators, leading to a margin at 95% of
    about (1.96 x 5.5e-3)/sqrt(2000) = 2.4e-4   (≈5% of the mean).
    """

    def __init__(
        self, training_cfg: TrainingCfg, n_estimator: int = 2000, n_neighbors: int = 3
    ) -> None:
        super().__init__(training_cfg)
        self.n_estimator = n_estimator
        self.n_neighbors = n_neighbors
        self._measure_in_progress = False

    @property
    def metrics_group_name(self) -> str:
        return "smi_metrics"

    @property
    def metrics_names(self) -> t.List[str]:
        return [
            f"{dl}_smi_{layer}"
            for dl in ["train_all", "train_trl", "train_rdl", "test"]
            for layer in LAYERS
        ]

    def prepare_forward_measure(
        self, checkpoint: int, len_trl: int, len_rdl: int, len_test: int
    ) -> None:
        if self._measure_in_progress:
            raise RuntimeError("You tries to prepare twice for forward measure.")
        self._measure_in_progress = True
        self._checkpoint_in_progress = checkpoint

        # Features: 0=train_trl, 1=train_rdl, 2=test
        # features_per_dl[0][4] = list of features of the 4th layer for train_all
        self._features_per_dl = [collections.defaultdict(list) for _ in range(3)]
        self._labels_per_dl = [[] for _ in range(3)]
        self._len_trl = len_trl
        self._len_rdl = len_rdl
        self._len_test = len_test

    def update_metrics(
        self,
        *,
        dl_idx,
        bs,
        input_ids,
        outputs,
    ):
        # Saving features
        for layer in LAYERS:
            self._features_per_dl[dl_idx - 1][layer].append(
                outputs["past_key_values"][layer][0][:, :, -3, :].view(bs, -1).cpu()
            )

        # Saving labels
        self._labels_per_dl[dl_idx - 1].append(input_ids[:, -2].cpu())

    def finalize_metrics(self) -> None:
        # SMI containers
        smi_values = np.zeros((4, len(LAYERS)))

        # SMI -- TRL
        if self._len_trl != 0:
            # Logging
            logger.info("Computing SMI for: Train -- true labels")

            # Tensors
            X_per_layer = {
                layer: torch.cat(features_list, dim=0)
                for layer, features_list in self._features_per_dl[0].items()
            }
            y = torch.cat(self._labels_per_dl[0], dim=0)

            # Logging
            logger.debug(f"X size: {X_per_layer[0].size()}")
            logger.debug(f"y size: {y.size()}")

            # Multiprocessing for each layer
            logger.info("Computing the SMI for each layer")
            smi_per_layer = []
            for layer in tqdm(LAYERS):
                smi_per_layer.append(
                    smi_estimator(
                        X_per_layer[layer],
                        y,
                        n_estimator=self.n_estimator,
                        n_neighbors=self.n_neighbors,
                    )
                )

            smi_values[1, :] = smi_per_layer
        else:
            logger.info(f"No Train -- true labels samples: SMI=0")

        # SMI -- RDL
        if self._len_rdl != 0:
            # Logging
            logger.info("Computing SMI for: Train -- random labels")

            # Tensors
            X_per_layer = {
                layer: torch.cat(features_list, dim=0)
                for layer, features_list in self._features_per_dl[1].items()
            }
            y = torch.cat(self._labels_per_dl[1], dim=0)

            # Logging
            logger.debug(f"X size: {X_per_layer[0].size()}")
            logger.debug(f"y size: {y.size()}")

            # Multiprocessing for each layer
            logger.info("Computing the SMI for each layer")
            smi_per_layer = []
            for layer in tqdm(LAYERS):
                smi_per_layer.append(
                    smi_estimator(
                        X_per_layer[layer],
                        y,
                        n_estimator=self.n_estimator,
                        n_neighbors=self.n_neighbors,
                    )
                )

            smi_values[2, :] = smi_per_layer
        else:
            logger.info(f"No Train -- random labels samples: SMI=0")

        # SMI -- Test
        if self._len_test != 0:
            # Logging
            logger.info("Computing SMI for: Test")

            # Tensors
            X_per_layer = {
                layer: torch.cat(features_list, dim=0)
                for layer, features_list in self._features_per_dl[2].items()
            }
            y = torch.cat(self._labels_per_dl[2], dim=0)

            # Logging
            logger.debug(f"X size: {X_per_layer[0].size()}")
            logger.debug(f"y size: {y.size()}")

            # Multiprocessing for each layer
            logger.info("Computing the SMI for each layer")
            smi_per_layer = []
            for layer in tqdm(LAYERS):
                smi_per_layer.append(
                    smi_estimator(
                        X_per_layer[layer],
                        y,
                        n_estimator=self.n_estimator,
                        n_neighbors=self.n_neighbors,
                    )
                )

            smi_values[3, :] = smi_per_layer
        else:
            logger.info(f"No Test samples: SMI=0")

        # SMI -- Trail all
        if self._len_trl + self._len_rdl != 0:
            # Logging
            logger.info("Computing SMI for: Train -- all")

            # Tensors
            X_trl_per_layer = {
                layer: torch.cat(features_list, dim=0)
                for layer, features_list in self._features_per_dl[0].items()
            }
            X_rdl_per_layer = {
                layer: torch.cat(features_list, dim=0)
                for layer, features_list in self._features_per_dl[1].items()
            }
            X_per_layer = {
                layer: torch.cat(
                    [X_trl_per_layer[layer], X_rdl_per_layer[layer]], dim=0
                )
                for layer in LAYERS
            }
            y_trl = torch.cat(self._labels_per_dl[0], dim=0)
            y_rdl = torch.cat(self._labels_per_dl[1], dim=0)
            y = torch.cat([y_trl, y_rdl], dim=0)

            # Logging
            logger.debug(f"X size: {X_per_layer[0].size()}")
            logger.debug(f"y size: {y.size()}")

            # Multiprocessing for each layer
            logger.info("Computing the SMI for each layer")
            smi_per_layer = []
            for layer in tqdm(LAYERS):
                smi_per_layer.append(
                    smi_estimator(
                        X_per_layer[layer],
                        y,
                        n_estimator=self.n_estimator,
                        n_neighbors=self.n_neighbors,
                    )
                )

            smi_values[0, :] = smi_per_layer
        else:
            logger.info(f"No Train -- all samples: SMI=0")

        # Output
        self._metrics_values = [
            smi_values[dl_idx, layer_idx]
            for dl_idx in range(4)
            for layer_idx in range(len(LAYERS))
        ]

        # Calling proper method
        self.compute_values(checkpoint=self._checkpoint_in_progress)

        # Remove lock
        self._measure_in_progress = False
        del self._checkpoint_in_progress
        del self._features_per_dl
        del self._labels_per_dl
        del self._metrics_values

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:
        if not hasattr(self, "_metrics_values"):
            raise ValueError(
                "This type of metric should not be called directly but throuth the ForwardMetrics class."
            )

        return self._metrics_values
