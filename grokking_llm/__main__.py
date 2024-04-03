# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import typer

from .deploy import (
    run_deploy_clean_forward_values,
    run_deploy_cpu,
    run_deploy_gpu,
    run_deploy_prepare,
)
from .measures_dyn import run_main_measure_dyn
from .training import run_main_train

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(config: t.Optional[str] = None):
    run_main_train(config)


@app.command()
def measure_dyn(
    name: str,
    config: t.Optional[str] = None,
    checkpoint: t.Optional[str] = None,
    force_recompute: bool = False,
):
    run_main_measure_dyn(
        name=name,
        config=config,
        checkpoint=checkpoint,
        force_recompute=force_recompute,
    )


@app.command()
def deploy_prepare(
    config: t.Optional[str] = None,
):
    run_deploy_prepare(config=config)


@app.command()
def deploy_clean_forward_values(
    config: t.Optional[str] = None,
    compressed_only: bool = False,
):
    run_deploy_clean_forward_values(config=config, compressed_only=compressed_only)


@app.command()
def deploy_gpu(
    config: t.Optional[str] = None,
    training: bool = True,
    self_forward: t.Optional[str] = None,
    self_forward_full_dataset: bool = False,
    forward_latest_on: t.Optional[str] = None,
):
    run_deploy_gpu(
        config=config,
        training=training,
        self_forward=self_forward,
        self_forward_full_dataset=self_forward_full_dataset,
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
