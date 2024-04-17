# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from grokking_llm.utils.deployment.deployment_cfg import DeploymentCfg

from ..training import get_random_split
from ..utils import (
    DeploymentCfg,
    get_brier_scores_for_mia,
    get_shadow_forward_values_for_mia,
)
from .static_metrics_group import StaticMetricsGroup


class CounterfactualMemoStatic(StaticMetricsGroup):
    """Class used to compute counterfactual memorization metrics.

    Counterfactual memorization: difference between the average performance of the models
    on x when x was on the training set vs. when it was not.
    See: "Counterfactual Memorization in Neural Language Models", Zhang et al, 2021.

    This is a static metric, that is computed and averaged over all models
    of a deployment config."""

    def __init__(self, deployment_cfg: DeploymentCfg) -> None:
        super().__init__(deployment_cfg)
        self.combine_fct = lambda t: t[0] - t[1]

    @property
    def metrics_group_name(self) -> str:
        return "counterfactual_memorization_static"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["mean_memo"] + [f"memo_{idx}" for idx in self.global_idx]

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # ==================== Forward values ====================

        shadow_forward_values = get_shadow_forward_values_for_mia(
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

        # Fetching the brier score for each shadow model
        # Shape: `num_samples` entries, each enty has size `num_shadow`
        # At position logits_gaps[i][j] we find the brier score for sample with index i and shadow model j
        brier_scores = get_brier_scores_for_mia(
            shadow_forward_values, global_idx=self.global_idx
        )

        # Checking if each target sample was in the training set of the shadow models
        # Shape: `num_sample` entries, each entry has size `num_shadow - 1`
        target_sample_is_in_shadow = {
            target_global_idx: [] for target_global_idx in self.global_idx
        }
        logger.debug(
            "Checking if each target sample was in the training set of the shadow models"
        )
        for shadow_training_cfg in self.training_cfg_list:
            shadow_split = get_random_split(self.base_train_set, shadow_training_cfg)
            shadow_global_idx = set(shadow_split["global_index"])

            for target_global_idx in self.global_idx:
                target_sample_is_in_shadow[target_global_idx].append(
                    target_global_idx in shadow_global_idx
                )

        # ==================== Mean brier score  ====================

        mean_pos_brier_per_idx = torch.zeros(num_samples)
        mean_neg_brier_per_idx = torch.zeros(num_samples)

        logger.debug(f"Computing mean pos/neg brier score per sample")
        for target_global_idx in tqdm(self.global_idx):

            # Getting logits_gaps for pos and neg shadow models
            pos_scores = [
                brier_scores[target_global_idx][shadow_idx]
                for shadow_idx in range(num_shadow)
                if target_sample_is_in_shadow[target_global_idx][shadow_idx]
            ]
            neg_scores = [
                brier_scores[target_global_idx][shadow_idx]
                for shadow_idx in range(1, num_shadow)
                if not target_sample_is_in_shadow[target_global_idx][shadow_idx]
            ]

            # Saving mean
            idx_count = idx_to_idx_count[target_global_idx]
            if len(pos_scores) == 0:
                logger.warning(
                    f"No positive brier score for the following global index: {target_global_idx}"
                )
                mean_pos_brier_per_idx[idx_count] = 0
            else:
                mean_pos_brier_per_idx[idx_count] = float(np.mean(pos_scores))

            if len(neg_scores) == 0:
                logger.warning(
                    f"No negative brier score for the following global index: {target_global_idx}"
                )
                mean_neg_brier_per_idx[idx_count] = 0
            else:
                mean_neg_brier_per_idx[idx_count] = float(np.mean(neg_scores))

        # ==================== Combining ====================

        metric_per_idx = self.combine_fct(
            (mean_pos_brier_per_idx, mean_neg_brier_per_idx)
        )

        # ==================== Output ====================

        mean_metric = torch.mean(metric_per_idx, axis=0)
        logger.debug(f"Mean Value: {mean_metric}")
        return [mean_metric] + metric_per_idx.tolist()
