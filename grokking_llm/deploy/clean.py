# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import re
import shutil
import typing as t
from pathlib import Path

from loguru import logger

from ..measures import ForwardMetrics
from ..utils import DeploymentCfg, get_possible_training_cfg

forward_on_pattern = re.compile("^forward_metrics_on_.+\.csv$")


def run_deploy_clean_forward_values(config: t.Union[str, Path]) -> None:
    """Initiates a deployment configuration, and deletes all forward values from
    the underlying training configs."""

    # Logging
    logger.info(
        f"Initiating the cleaning of the forward values for a deployment config."
    )

    # Autoconfig
    deployment_cfg = DeploymentCfg.autoconfig(config)

    # Get possible training_cfg cfg
    possible_training_cfg = get_possible_training_cfg(deployment_cfg)

    # Warning the user
    logger.info(
        f"You are about to delete the forward values for the {len(possible_training_cfg)} training configurations based on deployment config {deployment_cfg.get_deployment_id()}"
    )
    logger.warning(
        f"Are you sure you want to delete these forward values? You will not be able to undo it."
    )
    response = input("[Y/n] : ")
    if response != "Y":
        logger.info("Operation cancelled")
        return

    # Deleting
    for training_cfg in possible_training_cfg:

        # Logging
        logger.debug(f"Processing training config {training_cfg.get_config_id()}")
        forward_metrics = ForwardMetrics(training_cfg)

        for checkpoint in training_cfg.get_available_checkpoints():
            forward_export_dir = (
                training_cfg.get_output_dir()
                / f"checkpoint-{checkpoint}"
                / "forward_values"
            )

            if forward_export_dir.is_dir():
                logger.debug(f"Removing {forward_export_dir}")
                shutil.rmtree(forward_export_dir)

        # Deleting metrics
        for child in forward_metrics.metrics_dir.iterdir():

            if child.name == "forward_metrics.csv" or forward_on_pattern.match(
                child.name
            ):
                logger.debug(f"Removing {child}")
                child.unlink()

    # Logging
    logger.info(f"Cleaning of forward values: done.")
