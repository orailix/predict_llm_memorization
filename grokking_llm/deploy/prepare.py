# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from pathlib import Path

from loguru import logger

from ..utils import DeploymentCfg, get_possible_training_cfg


def run_deploy_prepare(config: t.Union[str, Path]) -> None:
    """Initiates a deployment configuration.

    - Builds a DeploymentCfg from the `name` that is provided
    - Parses the section of the DeploymentCfg object
    - Forms the combinations of training configurations
    - Dumps all possible training cfg.
    """

    # Logging
    logger.info(f"Preparing deployment from deployment config: {config}")

    # Autoconfig
    deployment_cfg = DeploymentCfg.autoconfig(config)

    # Get possible training_cfg cfg
    possible_training_cfg = get_possible_training_cfg(deployment_cfg)

    # Cleaning
    logger.debug(f"Cleaning existing stacks in {deployment_cfg.stacks_dir}")
    deployment_cfg.stack_todo.reset()
    deployment_cfg.stack_done.reset()
    for child in deployment_cfg.export_dir.iterdir():
        if (
            child.is_file()
            and child.suffix == ".json"
            and "training_cfg_" in child.stem
        ):
            child.unlink()

    # Dump
    logger.debug(f"Exporting training cfg in {deployment_cfg.configs_dir}")
    to_push = []
    for cfg_idx, cfg in enumerate(possible_training_cfg):
        export_path = deployment_cfg.configs_dir / f"training_cfg_{cfg_idx}.json"
        cfg.to_json(export_path)
        to_push.append(export_path)

    # Push to stacks
    logger.debug(f"Pushing exported configs to stacks in {deployment_cfg.stacks_dir}")
    deployment_cfg.stack_todo.push_chunk(to_push)

    # Logging
    logger.info(f"End of deployment preparation: {deployment_cfg.get_deployment_id()}")
