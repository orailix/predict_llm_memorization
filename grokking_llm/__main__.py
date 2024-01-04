# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from pathlib import Path

import typer
from loguru import logger

from .training import (
    TrainingCfg,
    add_labels,
    format_dataset,
    get_dataset,
    get_model,
    get_num_params,
    get_random_split,
    get_trainer,
    tokenize_dataset,
)
from .utils import paths

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def train(config: t.Optional[str] = None):

    # Parsing args
    if config is None:
        config = paths.training_cfg_path
    elif (paths.configs / config).exists():
        config = paths.configs / config
    else:
        config = Path(config)

    # Logging
    logger.info(f"Starting a training pipeline with config {config}")

    # Loading config
    logger.info("Loading config")
    cfg = TrainingCfg.from_file(config)
    logger.debug(cfg)

    # Dataset - train
    logger.info("Getting train dataset")
    train_dataset = get_dataset(cfg, split="train")
    train_dataset_formatted = format_dataset(train_dataset, cfg)
    train_dataset_labelled = add_labels(train_dataset_formatted, cfg)
    train_dataset_split = get_random_split(train_dataset_labelled, cfg)
    train_dataset_tokenized = tokenize_dataset(train_dataset_split, cfg)

    # Dataset - test
    logger.info("Getting test dataset")
    test_dataset = get_dataset(cfg, split="test")
    test_dataset_formatted = format_dataset(test_dataset, cfg)
    test_dataset_labelled = add_labels(test_dataset_formatted, cfg)
    test_dataset_tokenized = tokenize_dataset(test_dataset_labelled, cfg)

    # Model
    logger.info("Getting model")
    model = get_model(cfg)

    # Trainer
    logger.info("Getting trainer")
    trainer = get_trainer(
        cfg,
        model=model,
        train_dataset=train_dataset_tokenized,
        eval_dataset=test_dataset_tokenized,
    )

    # Logging trainable parameters
    all_param, trainable_params = get_num_params(model)
    logger.info(
        f"Trainable parameters: {trainable_params} / {all_param} ({trainable_params/all_param:.2%})"
    )

    # Training
    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=cfg.get_resume_from_checkpoint_status())


@app.command()
def evaluate(config: str):
    logger.info("Calling `evaluate` command...")
    logger.warning("TO BE COMPLETED")


if __name__ == "__main__":
    app()
