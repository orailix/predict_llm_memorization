# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import re
import shutil
import typing as t
from pathlib import Path

from loguru import logger

from ..measures_dyn import CompressForwardMetrics
from ..utils import DeploymentCfg, TrainingCfg, get_possible_training_cfg

forward_on_pattern = re.compile(r"^forward_metrics_on_.+\.csv$")


def run_deploy_compress_all_forward_values(
    config: t.Union[str, Path],
) -> None:
    """Compress all forward values from a deployment config."""

    # Logging
    logger.info(
        f"Initiating the compression of all forward values for a deployment config."
    )

    # Autoconfig
    deployment_cfg = DeploymentCfg.autoconfig(config)

    # Get possible training_cfg cfg
    possible_training_cfg = get_possible_training_cfg(deployment_cfg)

    # Informing the user
    logger.info(
        f"You are about to compress all forward values for the {len(possible_training_cfg)} training configurations based on deployment config {deployment_cfg.get_deployment_id()}"
    )

    # Compression
    for training_cfg in possible_training_cfg:

        # Logging
        logger.debug(f"Processing training config {training_cfg.get_config_id()}")
        compress_metrics = CompressForwardMetrics(training_cfg)

        # Removing metric file
        if (compress_metrics.metrics_dir / "compress_forward_metrics.csv").is_file():
            logger.info(
                f'Removing {compress_metrics.metrics_dir / "compress_forward_metrics.csv"}'
            )
            (compress_metrics.metrics_dir / "compress_forward_metrics.csv").unlink()

        # Re-load object
        logger.debug(f"Re-loading Compression object after metric file deletion")
        compress_metrics = CompressForwardMetrics(training_cfg)

        # Computing all values
        compress_metrics.compute_all_values()


def run_deploy_clean_forward_values(
    config: t.Union[str, Path],
    ignore_training_cfg_id: t.Optional[str] = None,
    delete_compressed: bool = False,
    delete_uncompressed: bool = False,
) -> None:
    """Initiates a deployment configuration, and deletes all forward values from
    the underlying training configs.

    Args:
        - config: The DeploymentCfg to clean forward values from
        - ignore_training_cfg_id: If provided, the TrainingCfg with this ID will be ignored in the cleaning process
        - delete_compressed: Whether or not to remove compressed forward values
        - delete_uncompressed: Whether or not to remove uncompressed forward values"""

    # Logging
    logger.info(
        f"Initiating the cleaning of the forward values for a deployment config."
    )

    # Autoconfig
    deployment_cfg = DeploymentCfg.autoconfig(config)

    # Ignore TrainingCfg
    if ignore_training_cfg_id is not None:
        ignore_training_cfg = TrainingCfg.autoconfig(ignore_training_cfg_id)
        logger.info(
            f"Throught the cleaning process, the following training config will be ignored: {ignore_training_cfg.get_config_id()}"
        )
    else:
        ignore_training_cfg = None
        logger.info("No training cfg will be ignored during the cleaning process.")

    # Get possible training_cfg cfg
    possible_training_cfg = get_possible_training_cfg(deployment_cfg)

    # Warning the user
    logger.info(
        f"You are about to delete the forward values for the {len(possible_training_cfg)} training configurations based on deployment config {deployment_cfg.get_deployment_id()}"
    )
    logger.info(f"Will remove all   compressed forward values: {delete_compressed}")
    logger.info(f"Will remove all uncompressed forward values: {delete_uncompressed}")
    logger.warning(
        f"Are you sure you want to delete these forward values? You will not be able to undo it."
    )
    response = input("[Y/n] : ")
    if response != "Y":
        logger.info("Operation cancelled")
        return

    # Deleting
    for training_cfg in possible_training_cfg:

        # Should we ignore it?
        if training_cfg.get_config_id() == ignore_training_cfg.get_config_id():
            logger.debug(f"Skipping training config {training_cfg.get_config_id()}")
            continue

        # Logging
        logger.debug(f"Processing training config {training_cfg.get_config_id()}")
        forward_metrics = CompressForwardMetrics(training_cfg)

        for checkpoint in training_cfg.get_available_checkpoints():
            forward_export_dir = (
                training_cfg.get_output_dir()
                / f"checkpoint-{checkpoint}"
                / "forward_values"
            )

            if forward_export_dir.is_dir():

                for child in forward_export_dir.iterdir():

                    if (
                        delete_compressed
                        and child.is_file()
                        and child.name[: len("compressed_")] == "compressed_"
                    ):
                        logger.debug(f"Removing {child}")
                        child.unlink()

                    if (
                        delete_uncompressed
                        and child.is_file()
                        and ("compressed" not in child.name)
                    ):
                        logger.debug(f"Removing {child}")
                        child.unlink()

        # Deleting metrics
        for child in forward_metrics.metrics_dir.iterdir():

            if delete_compressed and child.name == "compress_forward_metrics.csv":
                logger.debug(f"Removing {child}")
                child.unlink()

            if delete_uncompressed and (
                child.name == "forward_metrics.csv"
                or forward_on_pattern.match(child.name)
            ):
                logger.debug(f"Removing {child}")
                child.unlink()

    # Logging
    logger.info(f"Cleaning of forward values: done.")
