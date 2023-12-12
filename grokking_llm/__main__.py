"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import typer
from loguru import logger

app = typer.Typer()


@app.command()
def train(config: str):
    logger.info("Calling `train` command...")
    logger.warning("TO BE COMPLETED")


@app.command()
def evaluate(config: str):
    logger.info("Calling `evaluate` command...")
    logger.warning("TO BE COMPLETED")


if __name__ == "__main__":
    app()
