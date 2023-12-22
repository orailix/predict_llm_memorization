"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

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

from ..utils import paths
from .training_cfg import TrainingCfg


def get_num_params(model: PreTrainedModel):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return all_param, trainable_params


def get_model(cfg: TrainingCfg, sync_config: bool = False) -> PeftModel:
    """Gets the model, and adds the LoRA weights.

    The config object should contain the details about the model, LoRA config, and hardware.

    Args:
        cfg: A configuration object.
        sync_config: If True, the config will be sync from the disk config (to get the latest model).
            Else, we get the model at the exact epoch that is specified as cfg.epochs_done.

    Returns:
        peft.peft_model.PeftModel: The model
    """

    # Logging
    logger.info(f"Loading model {cfg.model} on device {cfg.accelerator}")

    # Model already saved ?
    if sync_config:
        cfg.sync_disk_to_object()

    # Do we expect a checkpoint from the disk or from HF hub ?
    loading_path = cfg.output_dir / f"epoch_{cfg.epochs_done}"
    if not loading_path.is_dir():
        if cfg.epochs_done == 0:
            # Loading from HF hub
            logger.info(f"Creating a new PEFT model")
            return _get_new_model(cfg)

        raise FileNotFoundError(
            f"No checkpoint found for epoch {cfg.epochs_done} ar {loading_path}"
        )

    # Logging
    logger.debug(f"Loading model from {loading_path}")

    # Raw model
    hf_model: t.Union[
        LlamaForCausalLM, MistralForCausalLM
    ] = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map=cfg.accelerator,
    )

    # Gredient checkpointing
    logger.debug(f"Enabling gradient checkpointing for model {cfg.model}")

    return PeftModel.from_pretrained(hf_model, loading_path)


def _get_new_model(cfg: TrainingCfg) -> PeftModel:
    """Gets a model, and adds the LoRA weights.

    The config object should contain the details about the model, LoRA config, and hardware.
    This function should only be called when no model has been found with the same config.

    Args:
        cfg: A configuration object.

    Returns:
        peft.peft_model.PeftModel: The model
    """

    # Raw model
    hf_model: t.Union[
        LlamaForCausalLM, MistralForCausalLM
    ] = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map=cfg.accelerator,
    )

    # Gredient checkpointing
    logger.debug(f"Enabling gradient checkpointing for model {cfg.model}")

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
    model = get_peft_model(hf_model, peft_config=lora_config)

    # Output
    return model


def save_model(model: PeftModel, cfg: TrainingCfg) -> Path:
    """Saves a PEFT model and returns the saving dir.

    Args:
        model: The model to save.
        cfg: A trainign configuration object.
    Returns:
        pathlib.Path: The directory in which the model has been saved.
    """

    # Saving model
    logger.info(f"Saving PEFT model based on {cfg.model} at: {cfg.output_dir}")
    cfg.sync_object_to_disk()
    logger.debug(f"Creating a saving dir for epoch {cfg.epochs_done}")
    epoch_saving_dir = cfg.output_dir / f"epoch_{cfg.epochs_done}"
    epoch_saving_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Saving epoch={cfg.epochs_done} at {epoch_saving_dir}")
    model.save_pretrained(epoch_saving_dir, save_embedding_layers=True)

    # Output
    return epoch_saving_dir
