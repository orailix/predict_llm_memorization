# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import typer

from .deploy import (
    run_deploy_clean_forward_values,
    run_deploy_compress_all_forward_values,
    run_deploy_cpu,
    run_deploy_gpu,
    run_deploy_prepare,
)
from .measures_dyn import run_main_measure_dyn
from .measures_stat import run_main_measure_stat
from .training import run_main_train

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(config: t.Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    run_main_train(config)


@app.command()
def measure_dyn(
    name: str,
    config: t.Optional[str] = None,
    checkpoint: t.Optional[str] = None,
    force_recompute: bool = False,
):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    run_main_measure_dyn(
        name=name,
        config=config,
        checkpoint=checkpoint,
        force_recompute=force_recompute,
    )


@app.command()
def measure_stat(
    name: str,
    config: t.Optional[str] = None,
    checkpoint: t.Optional[str] = None,
    force_recompute: bool = False,
    njobs: t.Optional[str] = None,
):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    if checkpoint is None:
        raise ValueError(f"You should pass a checkpoint with the --checkpoint option.")
    run_main_measure_stat(
        name=name,
        config=config,
        checkpoint=checkpoint,
        force_recompute=force_recompute,
        njobs=njobs,
    )


@app.command()
def deploy_prepare(
    config: t.Optional[str] = None,
):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    run_deploy_prepare(config=config)


@app.command()
def deploy_clean_forward_values(
    config: t.Optional[str] = None,
    ignore_training_cfg_id: t.Optional[str] = None,
    delete_compressed: bool = False,
    delete_uncompressed: bool = False,
):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")

    run_deploy_clean_forward_values(
        config=config,
        ignore_training_cfg_id=ignore_training_cfg_id,
        delete_compressed=delete_compressed,
        delete_uncompressed=delete_uncompressed,
    )


@app.command()
def deploy_compress_all_forward_values(
    config: t.Optional[str] = None,
):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")

    run_deploy_compress_all_forward_values(
        config=config,
    )


@app.command()
def deploy_gpu(
    config: t.Optional[str] = None,
    training: bool = True,
    self_forward: t.Optional[str] = None,
    self_forward_full_dataset: bool = False,
    self_forward_compress: bool = False,
    forward_latest_on: t.Optional[str] = None,
):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    run_deploy_gpu(
        config=config,
        training=training,
        self_forward=self_forward,
        self_forward_full_dataset=self_forward_full_dataset,
        self_forward_compress=self_forward_compress,
        forward_latest_on=forward_latest_on,
    )


@app.command()
def deploy_cpu(
    config: t.Optional[str] = None,
    checkpoint: t.Optional[str] = None,
    njobs: t.Optional[int] = None,
    force_recompute: bool = False,
    skip_metrics: t.Optional[str] = None,
    only_metrics: t.Optional[str] = None,
):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    run_deploy_cpu(
        config=config,
        checkpoint=checkpoint,
        force_recompute=force_recompute,
        njobs=njobs,
        skip_metrics=skip_metrics,
        only_metrics=only_metrics,
    )


if __name__ == "__main__":
    app()
