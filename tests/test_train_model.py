# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import copy
import shutil

import pytest
import torch
from loguru import logger
from transformers import AutoModelForCausalLM

from grokking_llm.training import TrainingCfg, get_model, get_num_params, save_model


def test_get_model_quality():
    cfg = TrainingCfg(model="dummy_llama")
    cfg_large_r = cfg.copy()
    cfg_large_r.lora_r = 2 * cfg_large_r.lora_r

    # Cleaning
    for dir_ in [cfg.get_output_dir(), cfg_large_r.get_output_dir()]:
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
    for dir_ in [cfg.get_output_dir(), cfg_large_r.get_output_dir()]:
        if dir_.is_dir():
            shutil.rmtree(dir_)
            logger.warning(f"This folder should not need to be removed: {dir_}")


def test_get_model_integrity():
    cfg = TrainingCfg(model="dummy_llama")

    # Pre-test cleaning
    shutil.rmtree(cfg.get_output_dir())
    output_dir = cfg.get_output_dir()

    # Check available checkpoints
    assert cfg.get_available_checkpoints() == []

    # Getting model
    logger.debug("Getting model_epoch_0")
    model_checkpoint_0 = get_model(cfg)
    assert not (output_dir / "checkpoint-0").is_dir()

    # Saving the model a first time
    logger.debug("Saving model_epoch_0")
    save_model(model_checkpoint_0, cfg, at_checkpoint=0)
    assert (output_dir / "checkpoint-0").is_dir()
    assert TrainingCfg.from_json(output_dir / "training_cfg.json") == cfg

    # Check available checkpoints
    assert cfg.get_available_checkpoints() == [0]

    # Updating the model for 1 step
    logger.debug("Updating model_checkpoint_0 => model_checkpoint_1")
    model_checkpoint_1 = model_checkpoint_0
    with torch.no_grad():
        for name, param in model_checkpoint_1.named_parameters():
            if "lora" in name:
                first_lora_parameter_checkpoint_0 = copy.deepcopy(param)
                param += 1
                first_lora_parameter_checkpoint_1 = param
                break

    # Checking that the model was updated
    assert (
        first_lora_parameter_checkpoint_0 != first_lora_parameter_checkpoint_1
    ).all()

    # Saving updated model
    logger.debug("Saving model_checkpoint_1")
    save_model(model_checkpoint_1, cfg, at_checkpoint=1)
    assert (output_dir / "checkpoint-1").is_dir()
    assert TrainingCfg.from_json(output_dir / "training_cfg.json") == cfg

    # Check available checkpoints
    assert cfg.get_available_checkpoints() == [0, 1]

    # Re-loading epoch 1
    logger.debug("Re-loading model_checkpoint_1")
    model_checkpoint_1_reloaded = get_model(cfg, at_checkpoint=1)
    for (_, param_1), (_, param_1_reloaded) in zip(
        model_checkpoint_1.named_parameters(),
        model_checkpoint_1_reloaded.named_parameters(),
    ):
        assert (param_1 == param_1_reloaded).all()

    # Re-loading epoch 0
    logger.debug("Re-loading model_checkpoint_1")
    model_epoch_0_reloaded = get_model(cfg, at_checkpoint=0)
    for name, param in model_epoch_0_reloaded.named_parameters():
        if "lora" in name:
            assert (param == first_lora_parameter_checkpoint_0).all()
            break

    # Check epoch 2 is not available
    logger.debug("Looking for model_epoch_2")
    with pytest.raises(FileNotFoundError):
        get_model(cfg, at_checkpoint=2)

    # Cleaning
    for dir_ in [output_dir, cfg.get_output_dir()]:
        if dir_.is_dir():
            shutil.rmtree(dir_)
