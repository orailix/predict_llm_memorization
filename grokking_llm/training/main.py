# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from pathlib import Path

from loguru import logger

from ..utils import paths
from .datasets import (
    add_labels,
    format_dataset,
    get_dataset,
    get_random_split,
    tokenize_dataset,
)
from .models import get_model, get_num_params
from .trainer import get_trainer
from .training_cfg import TrainingCfg


def run_main_train(config: t.Optional[str] = None):
    """Main training function."""

    # Parsing args
    logger.info("Loading config")
    cfg = TrainingCfg.autoconfig(config)
    logger.debug(cfg)

    # Dataset - train
    logger.info("Getting train dataset")
    train_dataset = get_dataset(cfg, split="train")
    train_dataset_formatted = format_dataset(train_dataset, cfg)
    train_dataset_labelled = add_labels(train_dataset_formatted, cfg, "train")
    train_dataset_split = get_random_split(train_dataset_labelled, cfg)
    train_dataset_tokenized = tokenize_dataset(train_dataset_split, cfg)

    # Dataset - test
    logger.info("Getting test dataset")
    test_dataset = get_dataset(cfg, split="test")
    test_dataset_formatted = format_dataset(test_dataset, cfg)
    test_dataset_labelled = add_labels(test_dataset_formatted, cfg, "test")
    if cfg.split_test:
        test_dataset_labelled = get_random_split(test_dataset_labelled, cfg)
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
