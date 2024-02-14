# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import typer

from .deploy import run_prepare_deploy
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
def prepare_deploy(
    config: t.Optional[str] = None,
):
    run_prepare_deploy(config=config)


@app.command()
def exec_deploy(
    gpu: t.Optional[int] = None,
    config: t.Optional[str] = None,
):
    raise NotImplementedError


if __name__ == "__main__":
    app()
