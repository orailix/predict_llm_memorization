# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
import typing as t
from pathlib import Path

import torch
from loguru import logger

from ..measures_dyn import run_main_measure_dyn
from ..training import run_main_train
from ..utils import DeploymentCfg, TrainingCfg


def run_deploy_gpu(
    config: t.Union[str, Path],
    training: bool = True,
    self_forward: t.Optional[str] = None,
    forward_latest_on: t.Optional[str] = None,
):
    """Executes the deployment on a GPU.

    Args:
        training: If True, the model will first be trained
        self_forward: If None, the self-forward pass will be computed for no checkpoints. If "all", it will be
            computed for all checkpoints. If "a,b,c" it will be computed for checkpoints a and b and c.
        forward_latest_on: If not None, we will compute ForwardMetrics(target_cfg = forward_latest_on) for the latest checkpoint
    """

    # Parsing checkpoint list
    if self_forward == "all":
        self_forward = "all"
    else:
        self_forward = self_forward.split(",")
        self_forward = [
            (int(item) if item != "latest" else item) for item in self_forward
        ]

    # Init
    gpu = (
        os.environ["CUDA_VISIBLE_DEVICES"]
        if "CUDA_VISIBLE_DEVICES" in os.environ
        else "ALL"
    )
    logger.info(f"Initiating an GPU deployment agent on GPU {gpu}")
    logger.info(f"training = {training}")
    logger.info(f"self_forward = {self_forward}")
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
            if training:
                logger.info(f"Starting training on GPU {gpu}: {training_cfg_path}")
                run_main_train(training_cfg_path)
                logger.info(f"Finished training on GPU {gpu}: {training_cfg_path}")
            else:
                logger.info(f"Skipped training on GPU {gpu}: {training_cfg_path}")

            # Free cuda
            torch.cuda.empty_cache()

            # General measures
            logger.info(f"Starting general measure on GPU {gpu}: {training_cfg_path}")
            run_main_measure_dyn("general", training_cfg_path)
            logger.info(f"Finished general measure on GPU {gpu}: {training_cfg_path}")

            # Forward metrics on all checkpoints ?
            if self_forward is not None:
                logger.info(
                    f"Starting forward measure on GPU {gpu}: {training_cfg_path}"
                )

                if self_forward == "all":
                    logger.info(f"Forward measure checkpoints: ALL")
                    run_main_measure_dyn("forward", training_cfg_path)
                else:
                    logger.info(f"Forward measure checkpoints: {self_forward}")
                    for checkpoint in self_forward:
                        run_main_measure_dyn(
                            "forward", training_cfg_path, checkpoint=checkpoint
                        )

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
                run_main_measure_dyn(
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
