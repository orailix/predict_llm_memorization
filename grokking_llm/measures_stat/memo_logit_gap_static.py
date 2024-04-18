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
    get_logit_gaps_for_mia,
    get_shadow_forward_values_for_mia,
)
from ..utils.constants import SIGMA_LOGIT_GAP
from .static_metrics_group import StaticMetricsGroup


class MemoLogitGapStatic(StaticMetricsGroup):
    """Class used to compute memorizatino metrics.

    Memorization is defined as the average logit gap between the true label
    and the following label with the greatest predicted probability.

    This is a static metric, that is computed and averaged over all models
    of a deployment config."""

    def __init__(
        self, deployment_cfg: DeploymentCfg, sigma: float = SIGMA_LOGIT_GAP
    ) -> None:

        # Parsing args
        logger.info(f"Using sigma={sigma}.")
        self.sigma = sigma

        # Main init
        super().__init__(deployment_cfg)

    @property
    def metrics_group_name(self) -> str:
        return "memo_logits_gap_static"

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

        # ==================== Logit gaps ====================

        # Fetching the logits gap for each shadow model
        # Shape: `num_samples` entries, each enty has size `num_shadow`
        # At position logits_gaps[i][j] we find the logits gap for sample with index i and shadow model j
        logits_gaps = get_logit_gaps_for_mia(
            shadow_forward_values, global_idx=self.global_idx
        )

        # ==================== Memorization score ====================

        memo_score = torch.zeros(num_samples)

        logger.debug(f"Computing logit gap memorization score per sample.")
        for target_global_idx in tqdm(self.global_idx):

            idx_count = idx_to_idx_count[target_global_idx]
            current_logit_gaps = logits_gaps[target_global_idx]
            current_memo_scores = torch.tanh(current_logit_gaps / self.sigma)
            memo_score[idx_count] = torch.mean(current_memo_scores, axis=0)

        # ==================== Output ====================

        mean_metric = torch.mean(memo_score, axis=0)
        logger.debug(f"Mean Value: {mean_metric}")
        return [mean_metric] + memo_score.tolist()
