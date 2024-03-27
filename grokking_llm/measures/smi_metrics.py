# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
import typing as t
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from loguru import logger

from ..utils import TrainingCfg
from ..utils.constants import SMI_LAYERS
from .dynamic_metrics_group import DynamicMetricsGroup
from .utils.forward_values import ForwardValues, get_forward_value
from .utils.smi import smi_estimator


class SmiMetrics(DynamicMetricsGroup):
    """Class used to compute Sliced MUtual Information metrics on the models.

    Performance metrics: (20 in total):
        For type in ["mean", "max"]:
            For layers in [0, 7, 15, 23, 31]:
                For dl in [train_all, train_trl, train_rdl, test]:
                    - [{dl}_smi_{type}_{layer}]

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
        return (
            [
                f"{dl}_smi_mean_{layer}"
                for dl in ["train_all", "train_trl", "train_rdl", "test"]
                for layer in SMI_LAYERS
            ]
            + [
                f"{dl}_smi_max_{layer}"
                for dl in ["train_all", "train_trl", "train_rdl", "test"]
                for layer in SMI_LAYERS
            ]
            + [
                f"{dl}_smi_min_{layer}"
                for dl in ["train_all", "train_trl", "train_rdl", "test"]
                for layer in SMI_LAYERS
            ]
        )

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # ==================== Looking for ForwardValues ====================

        # Getting forward values
        forward_values_trl = get_forward_value(
            self.training_cfg,
            checkpoint,
            f"train_trl_on_{self.training_cfg.get_config_id()}",
            enable_compressed=False,
        )
        forward_values_rdl = get_forward_value(
            self.training_cfg,
            checkpoint,
            f"train_rdl_on_{self.training_cfg.get_config_id()}",
            enable_compressed=False,
        )
        forward_values_test = get_forward_value(
            self.training_cfg,
            checkpoint,
            f"test",
            enable_compressed=False,
        )
        forward_values_all = ForwardValues.concat(
            forward_values_trl, forward_values_rdl, new_name="train_all"
        )

        # ==================== SMI values ====================

        # Storing the values
        # Dim 1 => train_all, train_trl, train_rdl, test
        # Dim 2 => layer #1, layer #2, etc
        smi_mean = np.zeros((4, len(SMI_LAYERS)), dtype=float)
        smi_max = np.zeros((4, len(SMI_LAYERS)), dtype=float)
        smi_min = np.zeros((4, len(SMI_LAYERS)), dtype=float)

        # Iterating over forward values
        for idx, forward_values in enumerate(
            [
                forward_values_all,
                forward_values_trl,
                forward_values_rdl,
                forward_values_test,
            ]
        ):

            # Logging
            logger.info(f"Computing SMI for {forward_values.name}")

            # Is there samples ?
            if forward_values.num_samples == 0:
                logger.info("No sample! SMI=0")
                continue

            # Tensors
            X_per_layer = forward_values.mcq_states_per_layer
            y = forward_values.mcq_labels

            # Logging
            logger.debug(f"X size: {X_per_layer[1].size()}")
            logger.debug(f"y size: {y.size()}")

            # Multiprocessing each layer
            n_jobs = min(len(SMI_LAYERS), os.cpu_count())
            logger.info(
                f"Computing the SMI for each layer using a pool of {n_jobs} processes."
            )

            def process_layer(x):
                return smi_estimator(
                    x, y, n_estimator=self.n_estimator, n_neighbors=self.n_neighbors
                )

            smi_per_layer = Parallel(n_jobs=n_jobs)(
                delayed(process_layer)(forward_values.mcq_states_per_layer[layer])
                for layer in SMI_LAYERS
            )

            for layer_idx in range(len(SMI_LAYERS)):
                smi_mean[idx, layer_idx] = smi_per_layer[layer_idx][0]
                smi_max[idx, layer_idx] = smi_per_layer[layer_idx][1]
                smi_min[idx, layer_idx] = smi_per_layer[layer_idx][2]

        # ==================== Output ====================

        return (
            [
                smi_mean[dl_idx, layer_idx]
                for dl_idx in range(4)
                for layer_idx in range(len(SMI_LAYERS))
            ]
            + [
                smi_max[dl_idx, layer_idx]
                for dl_idx in range(4)
                for layer_idx in range(len(SMI_LAYERS))
            ]
            + [
                smi_min[dl_idx, layer_idx]
                for dl_idx in range(4)
                for layer_idx in range(len(SMI_LAYERS))
            ]
        )
