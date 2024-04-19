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
from .memo_logit_gap_static import MemoLogitGapStatic
from .static_metrics_group import StaticMetricsGroup


class MemoLogitGapStdStatic(MemoLogitGapStatic):
    """Class used to compute memorizatino metrics.

    Memorization is defined as the standard deviation of logit gaps between the true label
    and the following label with the greatest predicted probability.

    This is a static metric, that is computed and averaged over all models
    of a deployment config."""

    column_offset = 1

    def __init__(
        self,
        deployment_cfg: DeploymentCfg,
    ) -> None:

        # Main init
        super().__init__(deployment_cfg)
        self.combine_fct = torch.std

    @property
    def metrics_group_name(self) -> str:
        return "memo_logits_gap_std_static"
