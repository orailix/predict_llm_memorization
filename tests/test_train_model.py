"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import typing as t

from transformers import AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM

from grokking_llm.training import TrainingCfg
from grokking_llm.training.models import get_model


def get_num_params(model: t.Union[LlamaForCausalLM, MistralForCausalLM]):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return all_param, trainable_params


def test_get_model():
    cfg = TrainingCfg(model="dummy_llama")

    # Models
    hf_model = AutoModelForCausalLM.from_pretrained(cfg.model)
    lora_model = get_model(cfg)
    cfg_large_r = cfg.copy()
    cfg_large_r.lora_r = 2 * cfg_large_r.lora_r
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
