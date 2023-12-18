"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

from pathlib import Path

import pytest

from grokking_llm.training import TrainingCfg

training_cfg_path = Path(__file__).parent / "files" / "training.cfg"


def test_train_config():
    training_cfg = TrainingCfg.from_file(training_cfg_path)
    assert type(training_cfg.model) == str
    assert type(training_cfg.dataset) == str
    assert type(training_cfg.max_len) == int
    assert type(training_cfg.label_noise) == float
    assert type(training_cfg.split_id) == int
    assert type(training_cfg.split_prop) == float
    assert type(training_cfg.lora_r) == int
    assert type(training_cfg.lora_alpha) == float
    assert type(training_cfg.lora_dropout) == float
    assert type(training_cfg.accelerator) == str


def test_train_config_model():
    with pytest.raises(ValueError):
        TrainingCfg(model="hello")


def test_train_config_dataset():
    with pytest.raises(ValueError):
        TrainingCfg(dataset="hello")


def test_train_config_max_len():
    with pytest.raises(ValueError):
        TrainingCfg(max_len=0)


def test_train_config_label_noise():
    with pytest.raises(ValueError):
        TrainingCfg(label_noise=-0.1)

    with pytest.raises(ValueError):
        TrainingCfg(label_noise=1.1)


def test_train_config_split_id():
    with pytest.raises(ValueError):
        TrainingCfg(split_id="hello")


def test_train_config_split_prop():
    with pytest.raises(ValueError):
        TrainingCfg(split_prop=-0.1)

    with pytest.raises(ValueError):
        TrainingCfg(split_prop=1.1)


def test_train_config_lora_r():
    with pytest.raises(ValueError):
        TrainingCfg(lora_r=0)


def test_train_config_lora_alpha():
    with pytest.raises(ValueError):
        TrainingCfg(lora_alpha=0)


def test_train_config_lora_dropout():
    with pytest.raises(ValueError):
        TrainingCfg(lora_dropout=-0.1)

    with pytest.raises(ValueError):
        TrainingCfg(lora_dropout=1.1)


def test_train_config_accelerator():
    with pytest.raises(RuntimeError):
        TrainingCfg(accelerator="hello")

    with pytest.raises(RuntimeError):
        TrainingCfg(accelerator="vulkan")
