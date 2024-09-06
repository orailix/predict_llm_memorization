# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import typing as t
from functools import cached_property

from loguru import logger

from ..training import get_dataset, get_random_split
from ..utils import ForwardValues, TrainingCfg, get_forward_values, p_smi_estimator
from ..utils.constants import SMI_N_EST
from .dynamic_metrics_group import DynamicMetricsGroup


class PSmiStdMetrics(DynamicMetricsGroup):
    """Class used to compute Pointwise Sliced Mutual Information STD metrics on the models."""

    def __init__(
        self,
        training_cfg: TrainingCfg,
        n_estimator: int = SMI_N_EST,
        full_dataset: bool = False,
        all_layers: bool = False,
    ) -> None:
        # Logging
        logger.info(
            f"Initializing a PSmiMetrics with {n_estimator} estimators, and full_dataset={full_dataset}."
        )
        self.full_dataset = full_dataset

        # List of global idx
        logger.debug(
            "Loading dataset to retrieve global IDX of the elements of the random split."
        )
        if not full_dataset:
            ds = get_dataset(training_cfg)
            ds = get_random_split(ds, training_cfg)
            self.global_idx = sorted(ds["global_index"])
        else:
            ds = get_dataset(training_cfg, split="train")
            self.global_idx = sorted(ds["global_index"])

        # All layers ?
        self.all_layers = all_layers
        if all_layers:
            self.layers_to_process = training_cfg.all_layers
        else:
            self.layers_to_process = training_cfg.smi_layers

        # Main initialization
        super().__init__(training_cfg)

        # Estimators
        self.n_estimator = n_estimator

    @property
    def metrics_group_name(self) -> str:
        if not self.full_dataset:
            return "p_smi_std_metrics"
        else:
            return "p_smi_std_on_full_dataset_metrics"

    @cached_property
    def metrics_names(self) -> t.List[str]:

        result = []
        for layer in self.layers_to_process:
            for idx in self.global_idx:
                result.append(f"std_psmi_{layer}_{idx}")

        return result

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # ==================== Looking for ForwardValues ====================

        # On which dataset
        if not self.full_dataset:
            on_dataset = f"on_{self.training_cfg.get_config_id()}"
        else:
            on_dataset = "on_full_dataset"

        # Getting forward values
        forward_values_trl = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"train_trl_{on_dataset}",
            enable_compressed=False,
        )
        forward_value_rdl = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"train_rdl_{on_dataset}",
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

        # ==================== PSMI values ====================

        # Logging
        logger.info(f"Computing P-SMI for config {self.training_cfg}")

        # Tensors
        X_per_layer = forward_values_all.mcq_states_per_layer
        y = forward_values_all.mcq_labels

        # Logging
        logger.debug(f"X size: {X_per_layer[1].size()}")
        logger.debug(f"y size: {y.size()}")

        # Uniprocess computation
        logger.debug("Starting P-SMI core computation")
        p_smi_per_layer = {
            layer: p_smi_estimator(
                X_per_layer[layer], y, n_estimator=self.n_estimator, return_std=True
            )
            for layer in self.layers_to_process
        }
        logger.debug("End of P-SMI core computation")

        # ==================== Output ====================

        psmi_std_per_layer_per_idx = collections.defaultdict(dict)

        for layer, (_, _, _, psmi_std) in p_smi_per_layer.items():
            # The values in psmi_mean, etc are in the same order as in X used above
            # As a result, it is the same order as forward_values_all.global_index
            for count, idx in enumerate(forward_values_all.global_index.tolist()):
                psmi_std_per_layer_per_idx[layer][idx] = psmi_std[count]

        # Result
        result = []
        for layer in self.layers_to_process:
            for idx in self.global_idx:
                result.append(psmi_std_per_layer_per_idx[layer][idx])

        return result