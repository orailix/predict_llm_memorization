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
    get_logit_gaps_for_pointwise,
    get_shadow_forward_values_for_pointwise,
    norm_pdf,
)
from .static_metrics_group import StaticMetricsGroup


class MemoMembershipStatic(StaticMetricsGroup):
    """Class used to compute memorization metrics.

    Memorization is defined as DP-distinguishability. As such, it is
    quantified by evaluating the accuracy of membership inference attacks.

    This is a static metric, that is computed and averaged over all models
    of a deployment config."""

    column_offset = 1

    def __init__(self, deployment_cfg: DeploymentCfg) -> None:
        super().__init__(deployment_cfg)

    @property
    def metrics_group_name(self) -> str:
        return "memo_mia_static"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["mean_memo"] + [f"memo_{idx}" for idx in self.global_idx]

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

        # ==================== Logit gaps + target_in_shadow ====================

        # Fetching the logits gap for each shadow model
        # Shape: `num_samples` entries, each enty has size `num_shadow`
        # At position logits_gaps[i][j] we find the logits gap for sample with index i and shadow model j
        logits_gaps = get_logit_gaps_for_pointwise(
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

        # ==================== Normal approximation for pos/neg likelihood ====================

        pos_likelihood_per_idx_per_shadow = torch.zeros(num_samples, num_shadow)
        neg_likelihood_per_idx_per_shadow = torch.zeros(num_samples, num_shadow)

        logger.debug(f"Normal approximation of the logits gaps")
        for target_global_idx in tqdm(self.global_idx):

            # Getting logits_gaps for pos and neg shadow models
            pos_logits_gaps = [
                logits_gaps[target_global_idx][shadow_idx]
                for shadow_idx in range(num_shadow)
                if target_sample_is_in_shadow[target_global_idx][shadow_idx]
            ]
            neg_logits_gaps = [
                logits_gaps[target_global_idx][shadow_idx]
                for shadow_idx in range(1, num_shadow)
                if not target_sample_is_in_shadow[target_global_idx][shadow_idx]
            ]

            # Special case if there is no positive or negative
            if len(pos_logits_gaps) == 0:
                logger.warning(
                    f"No positive logits gap for the following global index: {target_global_idx}"
                )
                pos_mean = 0
                pos_std = 0.1
            else:
                pos_mean = np.mean(pos_logits_gaps)
                pos_std = np.std(pos_logits_gaps)

            if len(neg_logits_gaps) == 0:
                logger.warning(
                    f"No negative logits gap for the following global index: {target_global_idx}"
                )
                neg_mean = 0
                neg_std = 0.1
            else:
                neg_mean = np.mean(neg_logits_gaps)
                neg_std = np.std(neg_logits_gaps)

            idx_count = idx_to_idx_count[target_global_idx]
            for shadow_idx in range(num_shadow):
                pos_likelihood_per_idx_per_shadow[idx_count, shadow_idx] = norm_pdf(
                    pos_mean, pos_std, logits_gaps[target_global_idx][shadow_idx]
                )
                neg_likelihood_per_idx_per_shadow[idx_count, shadow_idx] = norm_pdf(
                    neg_mean, neg_std, logits_gaps[target_global_idx][shadow_idx]
                )

        # ==================== Attack success rate ====================

        asr_per_idx = torch.zeros(num_samples)
        logger.info(f"Computing the Attack Success Rate (ASR) for each sample.")

        for target_global_idx in tqdm(self.global_idx):

            count_correct = 0
            count_incorrect = 0
            idx_count = idx_to_idx_count[target_global_idx]
            for shadow_idx in range(num_shadow):

                predict_member = (
                    pos_likelihood_per_idx_per_shadow[idx_count, shadow_idx]
                    > neg_likelihood_per_idx_per_shadow[idx_count, shadow_idx]
                )
                is_member = target_sample_is_in_shadow[target_global_idx][shadow_idx]

                if predict_member == is_member:
                    count_correct += 1
                else:
                    count_incorrect += 1

            asr_per_idx[idx_count] = count_correct / (count_correct + count_incorrect)

        # ==================== Output ====================

        mean_asr = torch.mean(asr_per_idx, axis=0)
        logger.debug(f"Mean Attack Success Rate: {mean_asr}")
        return [mean_asr] + asr_per_idx.tolist()
