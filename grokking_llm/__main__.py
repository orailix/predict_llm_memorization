# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import typer

from .deploy import run_deploy_cpu, run_deploy_gpu, run_deploy_prepare
from .measures import run_main_measure
from .training import run_main_train

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(config: t.Optional[str] = None):
    run_main_train(config)


@app.command()
def measure(
    name: str,
    config: t.Optional[str] = None,
    checkpoint: t.Optional[str] = None,
    force_recompute: bool = False,
):
    run_main_measure(
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
def deploy_gpu(
    config: t.Optional[str] = None,
    gpu: t.Optional[int] = None,
):
    run_deploy_gpu(
        config=config,
        gpu=gpu,
    )


@app.command()
def deploy_cpu(
    gpu: t.Optional[int] = None,
    config: t.Optional[str] = None,
):
    raise NotImplementedError


if __name__ == "__main__":
    app()
