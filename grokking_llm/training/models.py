"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import typing as t

from loguru import logger
from peft import LoraConfig, get_peft_model
from torch.nn import Module
from transformers import AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM

from .training_cfg import TrainingCfg


def get_model(cfg: TrainingCfg) -> Module:
    """Gets the model, and adds the LoRA weights.

    The config object should contain the details about the model, LoRA config,
    and hardware.

    Args:
        cfg: A configuration object.

    Returns:
        torch.nn.Module: The model
    """

    # Logging
    logger.info(f"Loading model {cfg.model} on device {cfg.accelerator}")

    # Raw model
    hf_model: t.Union[
        LlamaForCausalLM, MistralForCausalLM
    ] = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map=cfg.accelerator,
    )

    # Gredient checkpointing
    logger.debug(f"Enabling gradient checkpointing for model {cfg.model}")
    hf_model.gradient_checkpointing_enable()

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
