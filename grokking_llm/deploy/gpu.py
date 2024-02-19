# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
import typing as t
from pathlib import Path

from loguru import logger

from ..measures import run_main_measure
from ..training import run_main_train
from .deployment_cfg import DeploymentCfg


def run_deploy_gpu(
    config: t.Union[str, Path],
    gpu: str,
):
    """Executes the deployment on a GPU."""

    # Init
    if gpu is None:
        gpu = "all"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    logger.info(f"Initiating an GPU deployment agent on GPU {gpu}")
    deployment_cfg = DeploymentCfg.autoconfig(config)
    logger.info(f"Deployment configuration:\n{deployment_cfg}")

    # Deploy
    while not deployment_cfg.stack_todo_gpu.empty():

        # Getting training cfg
        training_cfg_path = deployment_cfg.stack_todo_gpu.pop()
        logger.info(
            f"The following training cfg is assigned to GPU {gpu}: {training_cfg_path}"
        )

        # Training
        logger.info(f"Starting training on GPU {gpu}: {training_cfg_path}")
        run_main_train(training_cfg_path)
        logger.info(f"Finished training on GPU {gpu}: {training_cfg_path}")

        # General measures
        logger.info(f"Starting general measure on GPU {gpu}: {training_cfg_path}")
        run_main_measure("general", training_cfg_path)
        logger.info(f"Finished general measure on GPU {gpu}: {training_cfg_path}")

        # Forward
        logger.info(f"Starting forward measure on GPU {gpu}: {training_cfg_path}")
        run_main_measure("forward", training_cfg_path)
        logger.info(f"Finished forward measure on GPU {gpu}: {training_cfg_path}")

        # Exiting
        deployment_cfg.stack_done_gpu.push(training_cfg_path)
        logger.info(
            f"GPU {gpu} has successfully processed training cfg: {training_cfg_path}"
        )
