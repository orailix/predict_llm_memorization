# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from loguru import logger
from peft import PeftModel
from peft.utils import constants

from .models import save_model
from .training_cfg import TrainingCfg

# To avoid saving the first embedding layer since it's not needed
constants.EMBEDDING_LAYER_NAMES.remove("lm_head")


def compute_mcq_last_token_loss(
    inputs_labels: torch.Tensor,
    outputs_logits: torch.Tensor,
    vocab_size: int,
):
    """Computes the loss that focuses on the token corresponding to the answer of the MCQ."""

    # We skip the EOS token on purpose
    logits_last_token = outputs_logits[:, -3].contiguous().view(-1, vocab_size)
    label_last_token = inputs_labels[:, -2].contiguous().view(-1)

    loss_fct = torch.nn.CrossEntropyLoss()

    return loss_fct(logits_last_token, label_last_token)


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
            loss = compute_mcq_last_token_loss(
                inputs["labels"], outputs["logits"], vocab_size=model.config.vocab_size
            )

        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics=metrics)

        # Deleting the optimizer of the non-last checkpoint
        run_dir = self._get_output_dir(trial=trial)
        current_step = self.state.global_step
        current_folder = f"checkpoint-{current_step}"

        for child in Path(run_dir).iterdir():
            if (
                child.is_dir()
                and "checkpoint-" in child.name
                and child.name != current_folder
            ):
                if (child / "optimizer.pt").is_file():
                    (child / "optimizer.pt").unlink()

                if (child / "scheduler.pt").is_file():
                    (child / "scheduler.pt").unlink()


def get_trainer(
    cfg: TrainingCfg,
    *,
    model: t.Union[
        transformers.MistralForCausalLM, transformers.LlamaForCausalLM, PeftModel
    ],
    train_dataset: Dataset = None,
    eval_dataset: Dataset = None,
) -> CustomTrainer:
    """Gets a trainer for LoRA finetuning.

    Args:
        - cfg: A TrainingCfg object describing the training.
        - model: The model to train. It should already have LoRA weights added.
        - train_dataset: The training dataset.
        - eval_dataset: The evaluation dataset.

    Returns:
        transformers.Trainer: The trainer."""

    # Datasets
    processed_train_dataset = (
        train_dataset.select_columns(["input_ids", "attention_mask", "labels"])
        if train_dataset is not None
        else None
    )
    processed_eval_dataset = (
        eval_dataset.select_columns(["input_ids", "attention_mask", "labels"])
        if eval_dataset is not None
        else None
    )

    # Training arguments
    training_args = transformers.TrainingArguments(
        **cfg.training_args, output_dir=cfg.get_output_dir()
    )

    # Creating the Trainer
    trainer = CustomTrainer(
        last_token_only=cfg.last_token_only,
        model=model,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,
        args=training_args,
    )

    # Saving model if needed
    if cfg.get_resume_from_checkpoint_status() is False:
        logger.info(f"Saving model at checkpoint 0")
        save_model(trainer.model, cfg, at_checkpoint=0)

    # Output
    return trainer
