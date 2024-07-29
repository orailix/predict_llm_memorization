# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import numpy as np
from loguru import logger

from ..training import get_dataset, get_random_split
from ..utils import ForwardValues, TrainingCfg, get_forward_values
from .dynamic_metrics_group import DynamicMetricsGroup


class SampleLossMetrics(DynamicMetricsGroup):
    """Class used to compute the loss for each sample of the training set."""

    column_offset = 1

    def __init__(self, training_cfg: TrainingCfg) -> None:
        # List of global idx
        logger.debug(
            "Loading dataset to retrieve global IDX of the elements of the random split."
        )
        ds = get_dataset(training_cfg)
        ds = get_random_split(ds, training_cfg)
        self.global_idx = sorted(ds["global_index"])
        super().__init__(training_cfg)

    @property
    def metrics_group_name(self) -> str:
        return "sample_loss"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["mean_loss"] + [f"loss_{idx}" for idx in self.global_idx]

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # ==================== Looking for ForwardValues ====================

        # Forward values
        forward_values_trl = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"train_trl_on_{self.training_cfg.get_config_id()}",
            enable_compressed=True,
        )
        forward_values_rdl = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"train_rdl_on_{self.training_cfg.get_config_id()}",
            enable_compressed=True,
        )
        forward_values_all = ForwardValues.concat(
            forward_values_trl, forward_values_rdl, new_name="train_all"
        )

        # ==================== Metrics values ====================
        losses = dict()
        for count, idx in enumerate(forward_values_all.global_index.tolist()):
            losses[idx] = forward_values_all.loss_asw[count].item()

        # ==================== Output ====================

        mean_metric = np.mean(losses)
        logger.debug(f"Mean Value: {mean_metric}")
        return [mean_metric] + [losses[idx] for idx in self.global_idx]
