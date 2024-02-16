# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from pathlib import Path

from joblib import Parallel, delayed
from loguru import logger

from ..measures import run_main_measure
from .deployment_cfg import DeploymentCfg


def run_deploy_cpu(
    config: t.Union[str, Path],
    checkpoint: t.Optional[str] = None,
    force_recompute: bool = False,
):
    """Executes the deployment on a CPU."""

    # Init
    logger.info(f"Initiating an CPU deployment agent.")
    deployment_cfg = DeploymentCfg.autoconfig(config)
    logger.info(f"Deployment configuration:\n{deployment_cfg}")

    # Deploy
    while not deployment_cfg.stack_todo_cpu.empty():

        # Getting training cfg
        training_cfg_path = deployment_cfg.stack_todo_gpu.pop()
        logger.info(
            f"The following training cfg is assigned this agent: {training_cfg_path}"
        )

        # Running measures
        Parallel(n_jobs=3)(
            delayed(run_main_measure)(
                measure_name,
                config=training_cfg_path,
                checkpoint=checkpoint,
                force_recompute=force_recompute,
            )
            for measure_name in ["perf", "smi", "weights"]
        )

        # Running perf measures
        logger.info(f"Starting PERF measures on training config: {training_cfg_path}")
        run_main_measure(
            "perf",
            config=training_cfg_path,
            checkpoint=checkpoint,
            force_recompute=force_recompute,
        )
        logger.info(f"Finished PERF measures on training config: {training_cfg_path}")

        # Running perf measures
        logger.info(f"Starting PERF measures on training config: {training_cfg_path}")
        run_main_measure(
            "perf",
            config=training_cfg_path,
            checkpoint=checkpoint,
            force_recompute=force_recompute,
        )
        logger.info(f"Finished PERF measures on training config: {training_cfg_path}")
