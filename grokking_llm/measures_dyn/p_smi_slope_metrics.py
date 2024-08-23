# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import typing as t
from functools import cached_property

from loguru import logger

from ..training import get_dataset, get_random_split
from ..utils import TrainingCfg, get_p_smi_containers
from .dynamic_metrics_group import DynamicMetricsGroup
from .p_smi_metrics import PSmiMetrics


class PSmiSlopeMetrics(DynamicMetricsGroup):
    """Class used to compute the Slope of Pointwise Sliced Mutual Information metrics.

    Requires the PSmiMetrics to be computed.
    """

    def __init__(
        self,
        training_cfg: TrainingCfg,
        full_dataset: bool = False,
    ) -> None:
        # Logging
        logger.info(f"Initializing a PSmiSlopeMetrics with full_dataset={full_dataset}")
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

        # Main initialization
        super().__init__(training_cfg)

    @property
    def metrics_group_name(self) -> str:
        if not self.full_dataset:
            return "p_smi_slope_metrics"
        else:
            return "p_smi_slope_on_full_dataset_metrics"

    @cached_property
    def metrics_names(self) -> t.List[str]:

        result = []
        for psmi_type in ["mean", "max", "min"]:
            for layer in self.smi_layers:
                for idx in self.global_idx:
                    result.append(f"{psmi_type}_psmi_slope_{layer}_{idx}")

        return result

    @cached_property
    def p_smi_containers(
        self,
    ) -> t.Tuple[
        t.Dict[int, t.Dict[int, t.Dict[int, float]]],
        t.Dict[int, t.Dict[int, t.Dict[int, float]]],
        t.Dict[int, t.Dict[int, t.Dict[int, float]]],
    ]:
        """Returns (
            p_smi_mean_per_checkpoint_per_layer_per_idx,
            p_smi_max_per_checkpoint_per_layer_per_idx,
            p_smi_min_per_checkpoint_per_layer_per_idx,
        )
        """
        logger.info(f"Loading dynamic P-SMI values")
        metrics = PSmiMetrics(
            training_cfg=self.training_cfg, full_dataset=self.full_dataset
        )
        metrics_df = metrics.load_metrics_df()
        return get_p_smi_containers(metrics_df)

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # ==================== Loading P-SMI values ====================

        (
            p_smi_mean_per_checkpoint_per_layer_per_idx,
            p_smi_max_per_checkpoint_per_layer_per_idx,
            p_smi_min_per_checkpoint_per_layer_per_idx,
        ) = self.p_smi_containers

        # ==================== Computing slope ====================

        # Init containers
        p_smi_slope_mean_layer_idx = collections.defaultdict(dict)
        p_smi_slope_max_layer_idx = collections.defaultdict(dict)
        p_smi_slope_min_layer_idx = collections.defaultdict(dict)

        # Checking checkpoint is computed
        if checkpoint not in p_smi_mean_per_checkpoint_per_layer_per_idx:
            raise RuntimeError(
                f"P-SMI for checkpoint {checkpoint} was not computed! It is "
            )

        # Filling values
        for layer in self.smi_layers:
            for idx in self.global_idx:
                # Mean PSMI
                p_smi_slope_mean_layer_idx[layer][idx] = (
                    p_smi_mean_per_checkpoint_per_layer_per_idx[checkpoint][layer][idx]
                    - p_smi_mean_per_checkpoint_per_layer_per_idx[0][layer][idx]
                )
                # Max PSMI
                p_smi_slope_max_layer_idx[layer][idx] = (
                    p_smi_max_per_checkpoint_per_layer_per_idx[checkpoint][layer][idx]
                    - p_smi_max_per_checkpoint_per_layer_per_idx[0][layer][idx]
                )
                # Min PSMI
                p_smi_slope_min_layer_idx[layer][idx] = (
                    p_smi_min_per_checkpoint_per_layer_per_idx[checkpoint][layer][idx]
                    - p_smi_min_per_checkpoint_per_layer_per_idx[0][layer][idx]
                )

        # ==================== Output ====================

        # Result
        result = []
        for container in [
            p_smi_slope_mean_layer_idx,
            p_smi_slope_max_layer_idx,
            p_smi_slope_min_layer_idx,
        ]:
            for layer in self.smi_layers:
                for idx in self.global_idx:
                    result.append(container[layer][idx])

        return result
