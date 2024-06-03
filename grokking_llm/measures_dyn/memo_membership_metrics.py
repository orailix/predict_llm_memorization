# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from typing import List

import numpy as np
from loguru import logger
from tqdm import tqdm

from ..training import get_dataset, get_random_split
from ..utils import (
    DeploymentCfg,
    TrainingCfg,
    get_logit_gaps_for_pointwise,
    get_mia_memo_score,
    get_possible_training_cfg,
    get_shadow_forward_values_for_pointwise,
    norm_pdf,
)
from ..utils.constants import MEMO_SCORE_EPSILON
from .dynamic_metrics_group import DynamicMetricsGroup


class MemoMembershipMetrics(DynamicMetricsGroup):
    """Class used to compute memorization metrics.

    Memorization is defined as DP-distinguishability. As such, it is
    quantified by evaluating the accuracy of membership inference attacks.
    """

    column_offset = 2

    def __init__(
        self,
        training_cfg: TrainingCfg,
        shadow_deployment_cfg: DeploymentCfg,
        memo_epsilon: float = MEMO_SCORE_EPSILON,
    ) -> None:

        # Parsing args
        self.shadow_deployment_cfg = shadow_deployment_cfg
        self.memo_epsilon = memo_epsilon

        # List of global idx
        logger.debug(
            "Loading dataset to retrieve global IDX of the elements of the random split."
        )
        self.base_dataset = get_dataset(training_cfg)
        ds_split = get_random_split(self.base_dataset, training_cfg)
        self.global_idx = sorted(ds_split["global_index"])

        # Main initialization
        super().__init__(training_cfg)

        # Shadow training configurations and checkpoints
        logger.debug(
            "Loading shadow training configurations and corresponding checkpoints"
        )
        possible_training_cfg = get_possible_training_cfg(self.shadow_deployment_cfg)
        self.shadow_training_cfg: t.List[TrainingCfg] = []
        for k in range(len(possible_training_cfg)):

            # Excluding the training cfg itself
            if possible_training_cfg[k] == self.training_cfg:
                continue

            try:
                self.shadow_training_cfg.append(possible_training_cfg[k])
            except IndexError:
                raise Exception(
                    f"You initialized a MemoMembershipMetrics object, but the following shadow "
                    f"model was not trained: {possible_training_cfg[k].get_config_id()}"
                )

    @property
    def metrics_group_name(self) -> str:
        return f"memo_on_shadow_{self.shadow_deployment_cfg.get_deployment_id()}"

    @property
    def metrics_names(self) -> t.List[str]:
        return (
            ["prop_memo"] + ["mean_memo"] + [f"memo_{idx}" for idx in self.global_idx]
        )

    def metrics_computation_core(self, checkpoint: int) -> List[float]:

        # ==================== Forward values ====================

        self_and_shadow_forward_values = get_shadow_forward_values_for_pointwise(
            [self.training_cfg] + self.shadow_training_cfg,
            checkpoint=None,
            on_dataset=self.training_cfg.get_config_id(),
        )

        # Filtering global idx
        for item in self_and_shadow_forward_values:
            item.filter_global_index(self.global_idx)

        # Unpacking some useful variables
        num_samples = len(self.global_idx)
        num_shadow = len(self_and_shadow_forward_values)

        # Logging
        # Recall that the first shadow model is a fake one, because it is the target model
        logger.debug(
            f"Using {num_shadow - 1} shadow model to attack {num_samples} target samples"
        )

        # ==================== Logit gaps + target_in_shadow ====================

        # Fetching the logits gap for each shadow model
        # Shape: `num_samples` entries, each enty has size `num_shadow`
        # At position logits_gaps[i][j] we find the logits gap for sample with index i and shadow model j
        logits_gaps = get_logit_gaps_for_pointwise(
            self_and_shadow_forward_values, global_idx=self.global_idx
        )

        # Checking if each target sample was in the training set of the shadow models
        # Shape: `num_sample` entries, each entry has size `num_shadow - 1`
        target_sample_is_in_shadow = {
            target_global_idx: [] for target_global_idx in self.global_idx
        }
        logger.debug(
            "Checking if each target sample was in the training set of the shadow models"
        )
        for shadow_training_cfg in [self.training_cfg] + self.shadow_training_cfg:
            shadow_split = get_random_split(self.base_dataset, shadow_training_cfg)
            shadow_global_idx = set(shadow_split["global_index"])

            for target_global_idx in self.global_idx:
                target_sample_is_in_shadow[target_global_idx].append(
                    target_global_idx in shadow_global_idx
                )

        # Sanity check: the first shadow model is the target model, so we expect to have only "True"
        for target_global_idx in self.global_idx:
            if target_sample_is_in_shadow[target_global_idx][0] is False:
                raise RuntimeError(
                    "Inconsistency: the first shadow model should be the target model."
                )

        # ==================== Normal approximation for pos/neg likelihood ====================

        # Normal approximation
        num_pos = []
        num_neg = []
        target_global_idx_memo_score = []
        logger.debug(f"Normal approximation of the logits gaps")
        for target_global_idx in tqdm(self.global_idx):

            # Getting logits_gaps for pos and neg shadow models
            pos_logits_gaps = [
                logits_gaps[target_global_idx][shadow_idx]
                for shadow_idx in range(1, num_shadow)
                if target_sample_is_in_shadow[target_global_idx][shadow_idx]
            ]
            neg_logits_gaps = [
                logits_gaps[target_global_idx][shadow_idx]
                for shadow_idx in range(1, num_shadow)
                if not target_sample_is_in_shadow[target_global_idx][shadow_idx]
            ]

            # Updating counts
            num_pos.append(len(pos_logits_gaps))
            num_neg.append(len(neg_logits_gaps))

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

            # Evaluation of the target model
            pos_likelihood = norm_pdf(
                pos_mean, pos_std, logits_gaps[target_global_idx][0]
            )
            neg_likelihood = norm_pdf(
                neg_mean, neg_std, logits_gaps[target_global_idx][0]
            )

            # Getting memo score
            memo_score = get_mia_memo_score(
                pos_likelihood=pos_likelihood,
                neg_likelihood=neg_likelihood,
                epsilon=self.memo_epsilon,
            )
            target_global_idx_memo_score.append(memo_score)

        # ==================== Memorization score + output ====================

        # Logging
        mean_num_pos, min_num_pos, max_num_pos = (
            np.mean(num_pos),
            np.min(num_pos),
            np.max(num_pos),
        )
        mean_num_neg, min_num_neg, max_num_neg = (
            np.mean(num_neg),
            np.min(num_neg),
            np.max(num_neg),
        )
        num_memorized = sum([(item > 0) for item in target_global_idx_memo_score])
        mean_memorized = np.mean(target_global_idx_memo_score)
        logger.debug(
            f"Num positive per target sample: mean={mean_num_pos}, min={min_num_pos}, max={max_num_pos}"
        )
        logger.debug(
            f"Num negative per target sample: mean={mean_num_neg}, min={min_num_neg}, max={max_num_neg}"
        )
        logger.debug(
            f"Num target sample memorized: {num_memorized}/{num_samples} [{num_memorized/num_samples:.2%}]"
        )
        logger.debug(f"Mean memorization score: {mean_memorized}")

        # Output
        return [
            num_memorized / num_samples,
            mean_memorized,
        ] + target_global_idx_memo_score
