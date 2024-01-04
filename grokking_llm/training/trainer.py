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


class CustomTrainer(transformers.Trainer):

    """Custom class to custom the compute_loss function."""

    def __init__(self, *args, last_token_only: bool = False, **kwargs):
        self.last_token_only = last_token_only
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)
        loss = outputs["loss"]
        last_token_loss = loss[:, -1]

        print(f"loss           : {loss.size()} : {loss}")
        print(f"last_token_loss: {last_token_loss.size()} : {last_token_loss}")

        return (last_token_loss, outputs) if return_outputs else last_token_loss


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

    return CustomTrainer(
        last_token_only=cfg.last_token_only,
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
