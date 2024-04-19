# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

from loguru import logger

from ..utils import DeploymentCfg
from .counterfactual_memo_static import CounterfactualMemoStatic
from .counterfactual_simplicity import CounterfactualSimplicityStatic
from .loss_static import LossStatic
from .memo_logit_gap_static import MemoLogitGapStatic
from .memo_mia_static import MemoMembershipStatic
from .p_smi_static import PSmiStatic
from .static_metrics_group import StaticMetricsGroup

NAMES_TO_METRICS: t.Dict[str, t.Type[StaticMetricsGroup]] = {
    "p_smi": PSmiStatic,
    "memo_mia": MemoMembershipStatic,
    "memo_counterfactual": CounterfactualMemoStatic,
    "simplicity_counterfactual": CounterfactualSimplicityStatic,
    "loss": LossStatic,
    "memo_logit_gap": MemoLogitGapStatic,
}


def run_main_measure_stat(
    name: str,
    config: t.Optional[str] = None,
    checkpoint: t.Optional[str] = None,
    force_recompute: bool = False,
    njobs: t.Optional[str] = None,
):
    """Main function for measures."""

    # Parsing inputs -- name
    if name in NAMES_TO_METRICS:
        metrics_class = NAMES_TO_METRICS[name]
        metrics_class_kwargs = {}
    else:
        raise ValueError(
            f"Got `name`='{name}', but it should be in {list(NAMES_TO_METRICS)}"
        )

    logger.info(f"Starting a measure pipeline with metric group '{name}'")

    # Parsing inputs -- training_cfg
    deployment_cfg = DeploymentCfg.autoconfig(config)
    logger.debug(f"Deployment ID: {deployment_cfg.get_deployment_id()}")

    # Parsing inputs -- checkpoint
    if "," in checkpoint:
        checkpoint = [int(item) for item in checkpoint.split(",")]
    else:
        try:
            checkpoint = int(checkpoint)
        except ValueError:
            raise ValueError(
                f"Got `checkpoint`={checkpoint}, but should be either None or a string that can be cast to an int."
            )

    logger.info(f"Running measures for checkpoint: {checkpoint}")

    # Parsing inputs -- njobs
    if njobs is not None:
        metrics_class_kwargs["njobs"] = int(njobs)

    # Running measures
    metrics = metrics_class(deployment_cfg=deployment_cfg, **metrics_class_kwargs)
    if isinstance(checkpoint, list):
        for ch in checkpoint:
            metrics.compute_values(checkpoint=ch, recompute_if_exists=force_recompute)
    else:
        metrics.compute_values(
            checkpoint=checkpoint, recompute_if_exists=force_recompute
        )
