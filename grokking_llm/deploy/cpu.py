# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
import typing as t
from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger

from ..measures_dyn import run_main_measure_dyn
from ..utils import DeploymentCfg, GotEndSignal

CPU_METRICS = [
    "general",
    "perf",
    "smi",
    "p_smi",
    "weights",
    "memo_proba_gap",
    "compress_forward",
]


def run_deploy_cpu(
    config: t.Union[str, Path],
    checkpoint: t.Optional[str] = None,
    force_recompute: bool = False,
    njobs: t.Optional[int] = None,
    skip_metrics: t.Optional[str] = None,
    only_metrics: t.Optional[str] = None,
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

    # skip_metrics and only_metrics
    metrics = CPU_METRICS
    if (skip_metrics is not None) and (only_metrics is not None):
        raise ValueError(f"You cannot use both `skip_metrics` and `only_metrics`.")
    elif skip_metrics is not None:
        to_skip = skip_metrics.split(",")
        logger.debug(f"Skipping metrics: {to_skip}")
        metrics = [m for m in metrics if m not in to_skip]
    elif only_metrics is not None:
        metrics = only_metrics.split(",")

    logger.debug(f"Computing CPU metrics: {metrics}")

    # Deploy -- njobs == 1
    if njobs == 1:
        while not deployment_cfg.stack_todo.empty():

            try:
                training_cfg_path = deployment_cfg.stack_todo.pop()
                for measure_name in metrics:
                    run_main_measure_dyn(
                        measure_name,
                        config=training_cfg_path,
                        checkpoint=checkpoint,
                        force_recompute=force_recompute,
                    )
                deployment_cfg.stack_done.push(training_cfg_path)
            except KeyboardInterrupt as e:
                logger.info(
                    "KeyboadInterrupt detected, pushing the current config to the TODO_GPU stack..."
                )
                deployment_cfg.stack_todo.push(training_cfg_path)
                raise e

            except GotEndSignal as e:
                logger.info(
                    "Sigterm detecter, pushing current config to the TODO_GPU stack..."
                )
                deployment_cfg.stack_todo.push(training_cfg_path)
                raise e

            except Exception as e:
                logger.info(
                    "Error detected, pushing the current config to the TODO_GPU stack..."
                )
                deployment_cfg.stack_todo.push(training_cfg_path)
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
        todo_cpu = deployment_cfg.stack_todo.pop_all()

        Parallel(n_jobs=njobs)(
            delayed(run_main_measure_dyn)(
                measure_name,
                config=training_cfg_path,
                checkpoint=checkpoint,
                force_recompute=force_recompute,
            )
            for measure_name in metrics
            for training_cfg_path in todo_cpu
        )

        # Stacks
        logger.debug("Updating TODO_CPU and DONE_CPU stacks.")
        deployment_cfg.stack_done.push_chunk(todo_cpu)

    except KeyboardInterrupt as e:
        logger.info(
            "KeyboadInterrupt detected, pushing the current config to the TODO_GPU stack..."
        )
        deployment_cfg.stack_todo.push_chunk(todo_cpu)
        raise e

    except GotEndSignal as e:
        logger.info("Sigterm detecter, pushing current config to the TODO_GPU stack...")
        deployment_cfg.stack_todo.push(training_cfg_path)
        raise e

    except Exception as e:
        logger.info(
            "Error detected, pushing the current config to the TODO_GPU stack..."
        )
        deployment_cfg.stack_todo.push_chunk(todo_cpu)
        raise e

    # Logging
    logger.info(f"Finished CPU deployment.")
