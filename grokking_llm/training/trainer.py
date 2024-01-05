# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import transformers
from datasets import Dataset
from peft import PeftModel
from peft.utils import constants
from torch.nn import CrossEntropyLoss

from .training_cfg import TrainingCfg

# To avoid saving the first embedding layer since it's not needed
constants.EMBEDDING_LAYER_NAMES.remove("lm_head")


class SaveAtStart(transformers.TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        control.should_save = True


class CustomTrainer(transformers.Trainer):

    """Custom class to custom the compute_loss function."""

    def __init__(self, *args, last_token_only: bool = False, **kwargs):
        self.last_token_only = last_token_only
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)

        if not self.last_token_only:
            loss = outputs["loss"]

        else:
            # We skip the EOS token on purpose
            logits_last_token = (
                outputs["logits"][:, -3].contiguous().view(-1, model.config.vocab_size)
            )
            label_last_token = inputs["labels"][:, -2].contiguous().view(-1)

            loss = CrossEntropyLoss()(logits_last_token, label_last_token)

        return (loss, outputs) if return_outputs else loss


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

    # Datasets
    processed_train_dataset = train_dataset.select_columns(
        ["input_ids", "attention_mask", "labels"]
    )
    processed_eval_dataset = eval_dataset.select_columns(
        ["input_ids", "attention_mask", "labels"]
    )

    # Training arguments
    training_args = transformers.TrainingArguments(
        **cfg.training_args, output_dir=cfg.get_output_dir()
    )

    # Callbacks
    callbacks = [SaveAtStart()]

    # Output
    return CustomTrainer(
        last_token_only=cfg.last_token_only,
        model=model,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,
        args=training_args,
        callbacks=callbacks,
    )
