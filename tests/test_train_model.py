"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import copy
import shutil
import typing as t

import pytest
import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM

from grokking_llm.training import TrainingCfg
from grokking_llm.training.models import get_model, save_model


def get_num_params(model: t.Union[LlamaForCausalLM, MistralForCausalLM, PeftModel]):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return all_param, trainable_params


def test_get_model_quality():
    cfg = TrainingCfg(model="dummy_llama")
    cfg_large_r = cfg.copy()
    cfg_large_r.lora_r = 2 * cfg_large_r.lora_r

    # Cleaning
    for dir_ in [cfg.output_dir, cfg_large_r.output_dir]:
        if dir_.is_dir():
            shutil.rmtree(dir_)

    # Models
    hf_model = AutoModelForCausalLM.from_pretrained(cfg.model)
    lora_model = get_model(cfg)
    lora_large_r_model = get_model(cfg_large_r)

    # Num of params
    hf_model_all, hf_model_trainable = get_num_params(hf_model)
    lora_model_all, lora_model_trainable = get_num_params(lora_model)
    lora_large_r_model_all, lora_large_r_model_trainable = get_num_params(
        lora_large_r_model
    )

    # Checks
    assert hf_model_all < lora_model_all < lora_large_r_model_all
    assert hf_model_all == hf_model_trainable
    assert hf_model_all == lora_model_all - lora_model_trainable
    assert hf_model_all == lora_large_r_model_all - lora_large_r_model_trainable
    assert lora_large_r_model_trainable == lora_model_trainable * 2

    # Cleaning
    for dir_ in [cfg.output_dir, cfg_large_r.output_dir]:
        if dir_.is_dir():
            shutil.rmtree(dir_)
            logger.warning(f"This folder should not need to be removed: {dir_}")


def test_get_model_integrity():
    cfg = TrainingCfg(model="dummy_llama")

    # Pre-test cleaning
    if cfg.output_dir.is_dir():
        shutil.rmtree(cfg.output_dir)

    assert not cfg.output_dir.is_dir()

    # Getting model
    logger.debug("Getting model_epoch_0")
    model_epoch_0 = get_model(cfg)
    assert not (cfg.output_dir / "epoch_0").is_dir()

    # Saving the model a first time
    logger.debug("Saving model_epoch_0")
    save_model(model_epoch_0, cfg)
    assert (cfg.output_dir / "epoch_0").is_dir()

    # Updating the model for 1 epoch
    logger.debug("Updating model_epoch_0 => model_epoch_1")
    cfg.epochs_done += 1
    model_epoch_1 = model_epoch_0
    with torch.no_grad():
        for name, param in model_epoch_1.named_parameters():
            if "lora" in name:
                first_lora_parameter_epoch_0 = copy.deepcopy(param)
                param += 1
                first_lora_parameter_epoch_1 = param
                break

    # Checking that the model was updated
    assert (first_lora_parameter_epoch_0 != first_lora_parameter_epoch_1).all()

    # Saving updated model
    logger.debug("Saving model_epoch_1")
    save_model(model_epoch_1, cfg)
    assert (cfg.output_dir / "epoch_1").is_dir()

    # Re-loading epoch 1
    logger.debug("Re-loading model_epoch_1")
    new_cfg = TrainingCfg(model="dummy_llama")
    model_epoch_1_reloaded = get_model(new_cfg, sync_config=True)
    assert new_cfg.epochs_done == 1
    for (_, param_1), (_, param_1_reloaded) in zip(
        model_epoch_1.named_parameters(), model_epoch_1_reloaded.named_parameters()
    ):
        assert (param_1 == param_1_reloaded).all()

    # Re-loading epoch 0
    logger.debug("Re-loading model_epoch_0")
    new_cfg.epochs_done = 0
    model_epoch_0_reloaded = get_model(new_cfg, sync_config=False)
    assert new_cfg.epochs_done == 0
    for name, param in model_epoch_0_reloaded.named_parameters():
        if "lora" in name:
            assert (param == first_lora_parameter_epoch_0).all()
            break

    # Check epoch 2 is not available
    logger.debug("Looking for model_epoch_2")
    new_cfg.epochs_total = 2
    new_cfg.epochs_done = 2
    with pytest.raises(FileNotFoundError):
        get_model(new_cfg)

    # Cleaning
    for dir_ in [cfg.output_dir, new_cfg.output_dir]:
        if dir_.is_dir():
            shutil.rmtree(dir_)
