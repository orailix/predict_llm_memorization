# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import torch
from loguru import logger
from tqdm import tqdm

from grokking_llm.utils.deployment.deployment_cfg import DeploymentCfg

from ..utils import (
    DeploymentCfg,
    get_losses_for_pointwise,
    get_shadow_forward_values_for_pointwise,
)
from .static_metrics_group import StaticMetricsGroup


class LossStatic(StaticMetricsGroup):
    """Class used to compute loss static metrics.

    The loss static metric of a sample is the mean loss accross every shadow model.

    This is a static metric, that is computed and averaged over all models
    of a deployment config."""

    column_offset = 1

    def __init__(self, deployment_cfg: DeploymentCfg) -> None:
        super().__init__(deployment_cfg)

    @property
    def metrics_group_name(self) -> str:
        return "loss_static"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["mean_loss"] + [f"loss_{idx}" for idx in self.global_idx]

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # ==================== Forward values ====================

        shadow_forward_values = get_shadow_forward_values_for_pointwise(
            training_cfg_list=self.training_cfg_list,
            checkpoint=checkpoint,
            on_dataset="full_dataset",
        )

        # Unpacking some useful variables
        num_samples = len(self.global_idx)
        num_shadow = len(shadow_forward_values)

        # Idx to idx count
        idx_to_idx_count = {
            idx: count for count, idx in enumerate(sorted(self.global_idx))
        }

        # Logging
        logger.debug(
            f"Using {num_shadow} shadow model to attack {num_samples} target samples"
        )

        # ==================== Losses + target_in_shadow ====================

        # Fetching the losses for each shadow model
        # Shape: `num_samples` entries, each enty has size `num_shadow`
        # At position losses[i][j] we find the loss for sample with index i and shadow model j
        losses = get_losses_for_pointwise(
            shadow_forward_values, global_idx=self.global_idx
        )

        # ==================== Mean loss  ====================

        mean_loss = torch.zeros(num_samples)

        logger.debug(f"Computing mean loss per sample")
        for target_global_idx in tqdm(self.global_idx):

            idx_count = idx_to_idx_count[target_global_idx]
            mean_loss[idx_count] = torch.mean(losses[target_global_idx], axis=0)

        # ==================== Output ====================

        mean_metric = torch.mean(mean_loss, axis=0)
        logger.debug(f"Mean Value: {mean_metric}")
        return [mean_metric] + mean_loss.tolist()
