# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
import typing as t
from pathlib import Path

import torch
from accelerate import Accelerator
from loguru import logger

from ..measures import run_main_measure
from ..training import run_main_train
from .deployment_cfg import DeploymentCfg


def run_deploy_gpu(
    config: t.Union[str, Path],
    gpu: str,
    train_only: bool = False,
):
    """Executes the deployment on a GPU.

    train_only: If True, only executes the training of the model (no forward pass)
    """

    # Init
    if gpu is None:
        gpu = "all"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

    logger.info(f"Initiating an GPU deployment agent on GPU {gpu}")
    logger.info(f"Train_only = {train_only}")
    deployment_cfg = DeploymentCfg.autoconfig(config)
    logger.info(f"Deployment configuration:\n{deployment_cfg}")

    # Deploy
    while not deployment_cfg.stack_todo_gpu.empty():

        try:

            # Getting training cfg
            training_cfg_path = deployment_cfg.stack_todo_gpu.pop()
            logger.info(
                f"The following training cfg is assigned to GPU {gpu}: {training_cfg_path}"
            )

            # Training
            logger.info(f"Starting training on GPU {gpu}: {training_cfg_path}")
            run_main_train(training_cfg_path)
            logger.info(f"Finished training on GPU {gpu}: {training_cfg_path}")

            # Free cuda
            torch.cuda.empty_cache()

            # General measures
            logger.info(f"Starting general measure on GPU {gpu}: {training_cfg_path}")
            run_main_measure("general", training_cfg_path)
            logger.info(f"Finished general measure on GPU {gpu}: {training_cfg_path}")

            if not train_only:

                # Forward
                logger.info(
                    f"Starting forward measure on GPU {gpu}: {training_cfg_path}"
                )
                run_main_measure("forward", training_cfg_path)
                logger.info(
                    f"Finished forward measure on GPU {gpu}: {training_cfg_path}"
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
