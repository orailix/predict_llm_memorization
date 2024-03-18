# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
import typing as t
from pathlib import Path

import torch
from loguru import logger

from ..measures import run_main_measure
from ..training import TrainingCfg, run_main_train
from .deployment_cfg import DeploymentCfg


def run_deploy_gpu(
    config: t.Union[str, Path],
    skip_training: bool = False,
    skip_self_forward: bool = False,
    forward_latest_on: t.Optional[str] = None,
):
    """Executes the deployment on a GPU.

    Args:
        skip_training: If True, skips the training part
        skip_self_forward: If True, skips the part where we compute the ForwardMetrics for all checkpoints
        forward_latest_on: If not None, we will compute ForwardMetrics(target_cfg = forward_latest_on) for the latest checkpoint
    """

    # Init
    gpu = (
        os.environ["CUDA_VISIBLE_DEVICES"]
        if "CUDA_VISIBLE_DEVICES" in os.environ
        else "ALL"
    )
    logger.info(f"Initiating an GPU deployment agent on GPU {gpu}")
    logger.info(f"skip_training = {skip_training}")
    logger.info(f"skip_self_forward = {skip_self_forward}")
    logger.info(f"forward_latest_on = {forward_latest_on}")
    deployment_cfg = DeploymentCfg.autoconfig(config)
    logger.info(f"Deployment configuration:\n{deployment_cfg}")

    # Checkpoint that forward_latest_on is a valid training configuration
    if forward_latest_on is not None:
        target_cfg = TrainingCfg.autoconfig(forward_latest_on)

    # Deploy
    while not deployment_cfg.stack_todo_gpu.empty():

        try:

            # Getting training cfg
            training_cfg_path = deployment_cfg.stack_todo_gpu.pop()
            logger.info(
                f"The following training cfg is assigned to GPU {gpu}: {training_cfg_path}"
            )

            # Training ?
            if not skip_training:
                logger.info(f"Starting training on GPU {gpu}: {training_cfg_path}")
                run_main_train(training_cfg_path)
                logger.info(f"Finished training on GPU {gpu}: {training_cfg_path}")
            else:
                logger.info(f"Skipped training on GPU {gpu}: {training_cfg_path}")

            # Free cuda
            torch.cuda.empty_cache()

            # General measures
            logger.info(f"Starting general measure on GPU {gpu}: {training_cfg_path}")
            run_main_measure("general", training_cfg_path)
            logger.info(f"Finished general measure on GPU {gpu}: {training_cfg_path}")

            # Forward metrics on all checkpoints ?
            if not skip_self_forward:
                logger.info(
                    f"Starting forward measure on GPU {gpu}: {training_cfg_path}"
                )
                run_main_measure("forward", training_cfg_path)
                logger.info(
                    f"Finished forward measure on GPU {gpu}: {training_cfg_path}"
                )
            else:
                logger.info(
                    f"Skipped forward measure on GPU {gpu}: {training_cfg_path}"
                )

            # Forward metrics on a target config for the latest checkpoint ?
            if forward_latest_on is not None:
                logger.info(
                    f"Starting forward measure of latest checkpoint on GPU {gpu}: {training_cfg_path}"
                )
                logger.info(f"Target configuration: {target_cfg.get_config_id()}")
                run_main_measure(
                    f"forward_on_{target_cfg.get_config_id()}",
                    config=training_cfg_path,
                    checkpoint="latest",
                )
                logger.info(
                    f"Finished forward measure of latest checkpoint on GPU {gpu}: {training_cfg_path}"
                )
            else:
                logger.info(
                    f"Skipped forward measure of latest checkpoint on GPU {gpu}: {training_cfg_path}"
                )

            # Exiting
            deployment_cfg.stack_done_gpu.push(training_cfg_path)
            logger.info(
                f"GPU {gpu} has successfully processed training cfg: {training_cfg_path}"
            )

        except KeyboardInterrupt as e:
            logger.info(
                "KeyboadInterrupt detected, pushing the current config to the TODO_GPU stack..."
            )
            deployment_cfg.stack_todo_gpu.push(training_cfg_path)
            raise e

        except Exception as e:
            logger.info(
                "Error detected, pushing the current config to the TODO_GPU stack..."
            )
            deployment_cfg.stack_todo_gpu.push(training_cfg_path)
            raise e
