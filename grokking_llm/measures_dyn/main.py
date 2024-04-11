# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import re
import typing as t

from loguru import logger

from ..utils import DeploymentCfg, TrainingCfg
from .compress_forward_metrics import CompressForwardMetrics
from .dynamic_metrics_group import DynamicMetricsGroup
from .forward_metrics import ForwardMetrics
from .general_metrics import GeneralMetrics
from .memo_logit_gap_metrics import MemoLogitGapMetrics
from .memo_membership_metrics import MemoMembershipMetrics
from .memo_proba_gap_metrics import MemoProbaGapMetrics
from .p_smi_metrics import PSmiMetrics
from .perf_metrics import PerfMetrics
from .smi_metrics import SmiMetrics
from .weights_metrics import WeightsMetrics

NAMES_TO_METRICS: t.Dict[str, t.Type[DynamicMetricsGroup]] = {
    "forward": ForwardMetrics,
    "general": GeneralMetrics,
    "perf": PerfMetrics,
    "smi": SmiMetrics,
    "p_smi": PSmiMetrics,
    "weights": WeightsMetrics,
    "memo_proba_gap": MemoProbaGapMetrics,
    "memo_logit_gap": MemoLogitGapMetrics,
    "compress_forward": CompressForwardMetrics,
}

p_smi_on_full_dataset = "p_smi_on_full_dataset"
forward_on_full_dataset = "forward_on_full_dataset"
forward_on_cfg_pattern = re.compile("^forward_on_.+$")
memorized_on_shadow_pattern = re.compile("^memo_on_shadow_.+$")


def run_main_measure_dyn(
    name: str,
    config: t.Optional[str] = None,
    checkpoint: t.Optional[str] = None,
    force_recompute: bool = False,
):
    """Main function for measures."""

    # Parsing inputs -- name
    if name in NAMES_TO_METRICS:
        metrics_class = NAMES_TO_METRICS[name]
        metrics_class_kwargs = {}
    elif name == p_smi_on_full_dataset:
        metrics_class = PSmiMetrics
        logger.info(f"Detected a `p_smi_on_full_dataset` metrics.")
        metrics_class_kwargs = {"full_dataset": True}
    elif name == forward_on_full_dataset:
        metrics_class = ForwardMetrics
        logger.info(f"Detected a `forward_on_full_dataset` metrics.")
        metrics_class_kwargs = {"full_dataset": True}
    elif forward_on_cfg_pattern.match(name):
        metrics_class = ForwardMetrics
        logger.info(f"Detected a `foward_on` metrics: initializing the target_cfg.")
        metrics_class_kwargs = {
            "target_cfg": TrainingCfg.autoconfig(name[len("forward_on_") :])
        }
    elif memorized_on_shadow_pattern.match(name):
        metrics_class = MemoMembershipMetrics
        logger.info(
            f"Detected a `memo_on_shadow`: initializing the shadow deployment_cfg`"
        )
        metrics_class_kwargs = {
            "shadow_deployment_cfg": DeploymentCfg.autoconfig(
                name[len("memo_on_shadow_") :]
            )
        }
    else:
        raise ValueError(
            f"Got `name`='{name}', but it should be in {list(NAMES_TO_METRICS)} or of "
            f"type `forward_on_[...]` or of type `memo_on_shadow_[...]`"
        )

    logger.info(f"Starting a measure pipeline with metric group '{name}'")

    # Parsing inputs -- training_cfg
    training_cfg = TrainingCfg.autoconfig(config)
    logger.debug(f"Config ID: {training_cfg.get_config_id()}")

    # Parsing inputs -- checkpoint
    if checkpoint in ["all", "ALL", None]:
        checkpoint = "all"

        if force_recompute:
            raise ValueError(
                "ckeckpoint='all' is incompatible with 'force_recompute=True'. If you want to force it, delete the metric output file and call this method again."
            )
    elif checkpoint == "latest":
        checkpoint = training_cfg.latest_checkpoint
    elif "," in checkpoint:
        checkpoint = [int(item) for item in checkpoint.split(",")]
    else:
        try:
            checkpoint = int(checkpoint)
        except ValueError:
            raise ValueError(
                f"Got `checkpoint`={checkpoint}, but should be either None or a string that can be cast to an int."
            )

    logger.info(f"Running measures for checkpoint: {checkpoint}")

    # Running measures
    metrics = metrics_class(training_cfg=training_cfg, **metrics_class_kwargs)
    if checkpoint == "all":
        metrics.compute_all_values()
    elif isinstance(checkpoint, list):
        for ch in checkpoint:
            metrics.compute_values(checkpoint=ch, recompute_if_exists=force_recompute)
    else:
        metrics.compute_values(
            checkpoint=checkpoint, recompute_if_exists=force_recompute
        )
