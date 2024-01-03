# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import transformers
from datasets import Dataset
from peft import PeftModel
from peft.utils import constants

from .training_cfg import TrainingCfg

# To avoid saving the first embedding layer since it's not needed
constants.EMBEDDING_LAYER_NAMES.remove("lm_head")


def get_trainer(
    cfg: TrainingCfg,
    *,
    model: t.Union[
        transformers.MistralForCausalLM, transformers.LlamaForCausalLM, PeftModel
    ],
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> transformers.Trainer:
    """Gets a trainer for LoRA finetuning.

    Args:
        - cfg: A TrainingCfg object describing the training.
        - model: The model to train. It should already have LoRA weights added.
        - train_dataset: The training dataset.
        - eval_dataset: The evaluation dataset.

    Returns:
        transformers.Trainer: The trainer."""

    return transformers.Trainer(
        model=model,
        train_dataset=train_dataset.select_columns(
            ["input_ids", "attention_mask", "labels"]
        ),
        eval_dataset=eval_dataset.select_columns(
            ["input_ids", "attention_mask", "labels"]
        ),
        args=transformers.TrainingArguments(
            **cfg.training_args, output_dir=cfg.get_output_dir()
        ),
    )
