# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import typing as t
from functools import cached_property

import torch
from loguru import logger

from ..training import get_dataset, get_random_split
from ..utils import ForwardValues, TrainingCfg, get_forward_values, mahalanobis_distance
from .dynamic_metrics_group import DynamicMetricsGroup


class MahalanobisMetrics(DynamicMetricsGroup):
    """Class used to compute the Mahalanobis distance of each instance based on the
    distribution of the hidden representations of the test set."""

    def __init__(
        self,
        training_cfg: TrainingCfg,
        all_layers: bool = False,
    ) -> None:

        # Logging
        logger.info(f"Initializing a MahalanobisMetrics estimator.")

        # List of global idx
        logger.debug(
            "Loading dataset to retrieve global IDX of the elements of the random split."
        )
        ds = get_dataset(training_cfg)
        ds = get_random_split(ds, training_cfg)
        self.global_idx = sorted(ds["global_index"])

        # All layers ?
        self.all_layers = all_layers
        if all_layers:
            self.layers_to_process = training_cfg.all_layers
        else:
            self.layers_to_process = training_cfg.smi_layers

        # Main initialization
        super().__init__(training_cfg)

    @property
    def metrics_group_name(self) -> str:
        return "mahalanobis_metrics"

    @cached_property
    def metrics_names(self) -> t.List[str]:

        result = []
        for layer in self.layers_to_process:
            for idx in sorted(self.global_idx):
                result.append(f"mahalanobis_{layer}_{idx}")

        return result

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # ==================== Looking for ForwardValues ====================

        # Getting forward values
        forward_values_trl = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"train_trl_on_{self.training_cfg.get_config_id()}",
            enable_compressed=False,
        )
        forward_value_rdl = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"train_rdl_on_{self.training_cfg.get_config_id()}",
            enable_compressed=False,
        )
        forward_values_all = ForwardValues.concat(
            forward_values_trl, forward_value_rdl, new_name="train_all"
        )

        # Sanity check
        if forward_values_all.global_index.size(0) != len(self.global_idx):
            raise RuntimeError(
                f"Incorrect loading of the forward values: {len(forward_values_all)} != {len(self.global_idx)}"
            )

        # ==================== Estimating distance ====================

        mahalanobis_per_layer_per_idx = collections.defaultdict(dict)
        for layer in list(forward_values_all.mcq_states_per_layer):

            # Estimating
            features = forward_values_all.mcq_states_per_layer[
                layer
            ]  # Shape (num_train, 4096)
            distances = mahalanobis_distance(features, features)

            # Saving
            for pos, idx in enumerate(forward_values_all.global_index.tolist()):
                mahalanobis_per_layer_per_idx[layer][idx] = distances[pos].item()

        # ==================== Output ====================

        result = []
        for layer in self.layers_to_process:
            for idx in sorted(self.global_idx):
                result.append(mahalanobis_per_layer_per_idx[layer][idx])

        return result
