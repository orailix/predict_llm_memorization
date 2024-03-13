# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import re
import typing as t

from loguru import logger

from ..training import TrainingCfg
from .dynamic_metrics_group import DynamicMetricsGroup
from .forward_metrics import ForwardMetrics
from .general_metrics import GeneralMetrics
from .perf_metrics import PerfMetrics
from .smi_metrics import SmiMetrics
from .weights_metrics import WeightsMetrics

NAMES_TO_METRICS: t.Dict[str, t.Type[DynamicMetricsGroup]] = {
    "forward": ForwardMetrics,
    "general": GeneralMetrics,
    "perf": PerfMetrics,
    "smi": SmiMetrics,
    "weights": WeightsMetrics,
}

forward_on_cfg_pattern = re.compile("^forward_on_.+$")


def run_main_measure(
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
    elif forward_on_cfg_pattern.match(name):
        metrics_class = ForwardMetrics
        metrics_class_kwargs = {"target_cfg_name": name[len("forward_on_") :]}
    else:
        raise ValueError(
            f"Got `name`='{name}', but it should be in {list(NAMES_TO_METRICS)} or of type `forward_on_[...]`"
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
        checkpoint = sorted(training_cfg.get_available_checkpoints())[-1]
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
    else:
        metrics.compute_values(
            checkpoint=checkpoint, recompute_if_exists=force_recompute
        )
