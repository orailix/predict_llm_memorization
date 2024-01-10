# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from pathlib import Path

from loguru import logger

from ..training import TrainingCfg
from ..utils import paths
from .dynamic_metrics_group import DynamicMetricsGroup
from .perf_metrics import PerfMetrics
from .weights_metrics import WeightsMetrics

NAMES_TO_METRICS: t.Dict[str, t.Type[DynamicMetricsGroup]] = {
    "perf": PerfMetrics,
    "weights": WeightsMetrics,
}


def run_main_measure(
    name: str,
    config: t.Optional[str] = None,
    checkpoint: t.Optional[str] = None,
):
    """Main function for measures."""

    # Parsing inputs -- name
    if name not in NAMES_TO_METRICS:
        raise ValueError(
            f"Got `name`='{name}', but it should be in {list(NAMES_TO_METRICS)}"
        )

    metrics_class = NAMES_TO_METRICS[name]
    logger.info(f"Starting a measure pipeline with metric group '{name}'")

    # Parsing inputs -- training_cfg
    if config is None:
        config = paths.training_cfg_path
    elif (paths.configs / config).exists():
        config = paths.configs / config
    else:
        config = Path(config)

    logger.info(f"Running measures on config {config}")
    training_cfg = TrainingCfg.from_file(config)
    logger.debug(f"Config ID: {training_cfg.get_config_id()}")

    # Parsing inputs -- checkpoint
    if checkpoint in ["all", "ALL", None]:
        checkpoint = "all"
    else:
        try:
            checkpoint = int(checkpoint)
        except ValueError:
            raise ValueError(
                f"Got `checkpoint`={checkpoint}, but should be either None or a string that can be cast to an int."
            )

    logger.info(f"Running measures for checkpoint: {checkpoint}")

    # Running measures
    metrics = metrics_class(training_cfg=training_cfg)
    if checkpoint == "all":
        metrics.compute_all_values()
    else:
        metrics.compute_values(checkpoint=checkpoint)
