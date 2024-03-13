# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from pathlib import Path

from loguru import logger
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    PreTrainedModel,
)

from .training_cfg import TrainingCfg

CHECKPOINT_FORMAT = "checkpoint-{number}"


def get_num_params(model: PreTrainedModel):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return all_param, trainable_params


def get_model(cfg: TrainingCfg, at_checkpoint: t.Optional[int] = None) -> PeftModel:
    """Gets the model, and adds the LoRA weights.

    The config object should contain the details about the model, LoRA config, and hardware.

    Args:
        cfg: A configuration object.
        at_checkpoint: If provided, the model will be loaded from the checkpoint with this number.

    Returns:
        peft.peft_model.PeftModel: The model
    """

    # Logging
    logger.info(f"Loading model {cfg.model} on device {cfg.accelerator}")
    logger.debug(f"Output dir: {cfg.get_output_dir()}")

    # Are we looking for a precise checkpoint ?
    if at_checkpoint is not None:

        if at_checkpoint == "latest":
            at_checkpoint = sorted(cfg.get_available_checkpoints())[-1]

        checkpoint_dir = cfg.get_output_dir() / CHECKPOINT_FORMAT.format(
            number=at_checkpoint
        )

        logger.debug(f"Looking for checkpoint {at_checkpoint} at {checkpoint_dir}")
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint {at_checkpoint} does not exist in {cfg.get_output_dir()}. Use available_ckeckpoints(cfg) for a list of checkpoints."
            )

        # Raw model
        hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            device_map=cfg.accelerator,
        )

        return PeftModel.from_pretrained(hf_model, checkpoint_dir)

    # Else, creating a new PEFT model
    logger.debug("Creating a new PEFT model.")

    # Raw model
    hf_model: t.Union[
        LlamaForCausalLM, MistralForCausalLM
    ] = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map=cfg.accelerator,
    )

    # LoRA
    logger.debug(
        f"Adding LoRA weights with r={cfg.lora_r}, alpha={cfg.lora_alpha}, dropout={cfg.lora_dropout}"
    )
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=cfg.lora_dropout,
    )

    # Output
    return get_peft_model(hf_model, peft_config=lora_config)


def save_model(model: PeftModel, cfg: TrainingCfg, at_checkpoint: int) -> Path:
    """Saves a PEFT model and returns the saving dir.

    Args:
        model: The model to save.
        cfg: A trainign configuration object.
        at_checkpoint: The checkpoint at which to save the model.
    Returns:
        pathlib.Path: The directory in which the model has been saved.
    """

    # Saving dir path
    saving_dir = cfg.get_output_dir() / CHECKPOINT_FORMAT.format(number=at_checkpoint)

    # Saving model
    logger.info(f"Saving PEFT model based on {cfg.model} at: {saving_dir}")
    model.save_pretrained(saving_dir, save_embedding_layers=True)

    # Output
    return saving_dir
