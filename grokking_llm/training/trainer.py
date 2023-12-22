"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import typing as t

import transformers
from datasets import Dataset
from loguru import logger
from peft import PeftModel

from .training_cfg import TrainingCfg


def get_trainer(
    cfg: TrainingCfg,
    *,
    model: t.Union[
        transformers.MistralForCausalLM, transformers.LlamaForCausalLM, PeftModel
    ],
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer
) -> transformers.Trainer:
    """TO BE COMPLETED."""

    return transformers.Trainer(
        model=model,
        train_dataset=train_dataset.select_columns(
            ["input_ids", "attention_mask", "labels"]
        ),
        eval_dataset=eval_dataset.select_columns(
            ["input_ids", "attention_mask", "labels"]
        ),
        args=transformers.TrainingArguments(
            output_dir=cfg.output_dir,
            warmup_steps=5,
            per_device_train_batch_size=2,
            gradient_checkpointing=False,
            gradient_accumulation_steps=4,
            max_steps=10000,
            learning_rate=2.5e-5,  # Want about 10x smaller than the Mistral learning rate
            logging_steps=50,
            bf16=True,
            optim="adamw_torch",
            logging_dir="./logs/",  # Directory for storing logs
            save_strategy="epoch",  # Save the model checkpoint every logging step
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=100,  # Evaluate and save checkpoints every 50 steps
            do_eval=True,  # Perform evaluation at the end of training
            remove_unused_columns=False,
            resume_from_checkpoint=True,  # Resume from latest checkpoint
        ),
    )


from peft.utils import constants

constants.EMBEDDING_LAYER_NAMES.remove("lm_head")
