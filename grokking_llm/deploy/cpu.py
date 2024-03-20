# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
import typing as t
from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger

from ..measures import run_main_measure
from ..utils import DeploymentCfg


def run_deploy_cpu(
    config: t.Union[str, Path],
    checkpoint: t.Optional[str] = None,
    force_recompute: bool = False,
    njobs: t.Optional[int] = None,
):
    """Executes the deployment on a CPU."""

    # Parsing
    if njobs is None:
        njobs = os.cpu_count()
    elif not isinstance(njobs, int):
        njobs = int(njobs)

    # Init
    logger.info(f"Initiating a CPU deployment with {njobs} jobs.")
    deployment_cfg = DeploymentCfg.autoconfig(config)
    logger.info(f"Deployment configuration:\n{deployment_cfg}")

    # Deploy -- njobs == 1
    if njobs == 1:

        while not deployment_cfg.stack_todo_cpu.empty():

            try:
                training_cfg_path = deployment_cfg.stack_todo_cpu.pop()
                for measure_name in ["perf", "smi", "weights"]:
                    run_main_measure(
                        measure_name,
                        config=training_cfg_path,
                        checkpoint=checkpoint,
                        force_recompute=force_recompute,
                    )
                deployment_cfg.stack_done_cpu.push(training_cfg_path)
            except KeyboardInterrupt as e:
                logger.info(
                    "KeyboadInterrupt detected, pushing the current config to the TODO_GPU stack..."
                )
                deployment_cfg.stack_todo_cpu.push(training_cfg_path)
                raise e

            except Exception as e:
                logger.info(
                    "Error detected, pushing the current config to the TODO_GPU stack..."
                )
                deployment_cfg.stack_todo_cpu.push(training_cfg_path)
                raise e

        logger.info(f"End of CPU deployment")
        return

    # Deploy -- njobs > 1:
    logger.info(
        f"With njobs={njobs} > 1, we mix training config and measures for more efficiency."
    )
    logger.debug(
        f"As a consequence, the stacks TODO_CPU and DONE_CPU are not accurate during the computation process."
    )

    try:
        todo_cpu = deployment_cfg.stack_todo_cpu.pop_all()

        Parallel(n_jobs=njobs)(
            delayed(run_main_measure)(
                measure_name,
                config=training_cfg_path,
                checkpoint=checkpoint,
                force_recompute=force_recompute,
            )
            for measure_name in ["perf", "smi", "weights"]
            for training_cfg_path in todo_cpu
        )

        # Stacks
        logger.debug("Updating TODO_CPU and DONE_CPU stacks.")
        deployment_cfg.stack_done_cpu.push_chunk(todo_cpu)

    except KeyboardInterrupt as e:
        logger.info(
            "KeyboadInterrupt detected, pushing the current config to the TODO_GPU stack..."
        )
        deployment_cfg.stack_todo_cpu.push_chunk(todo_cpu)
        raise e

    except Exception as e:
        logger.info(
            "Error detected, pushing the current config to the TODO_GPU stack..."
        )
        deployment_cfg.stack_todo_cpu.push_chunk(todo_cpu)
        raise e

    # Logging
    logger.info(f"Finished CPU deployment.")
