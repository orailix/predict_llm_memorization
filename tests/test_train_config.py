"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

from pathlib import Path

import pytest

from grokking_llm.training import TrainingCfg

training_cfg_path = Path(__file__).parent / "files" / "training.cfg"
training_cfg_json_path = Path(__file__).parent / "files" / "training_cfg.json"
training_cfg_export_path = Path(__file__).parent / "files" / "training_cfg_export.json"


def test_train_config_from_file():
    training_cfg = TrainingCfg.from_cfg(training_cfg_path)
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


def test_train_config_from_json():
    training_cfg = TrainingCfg.from_json(training_cfg_json_path)
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

    # Consistency with .cfg file ?
    training_cfg_from_cfg = TrainingCfg.from_cfg(training_cfg_path)
    assert training_cfg.model == training_cfg_from_cfg.model
    assert training_cfg.dataset == training_cfg_from_cfg.dataset
    assert training_cfg.max_len == training_cfg_from_cfg.max_len
    assert training_cfg.label_noise == training_cfg_from_cfg.label_noise
    assert training_cfg.split_id == training_cfg_from_cfg.split_id
    assert training_cfg.split_prop == training_cfg_from_cfg.split_prop
    assert training_cfg.lora_r == training_cfg_from_cfg.lora_r
    assert training_cfg.lora_alpha == training_cfg_from_cfg.lora_alpha
    assert training_cfg.lora_dropout == training_cfg_from_cfg.lora_dropout
    assert training_cfg.accelerator == training_cfg_from_cfg.accelerator


def test_train_config_export():
    training_cfg = TrainingCfg.from_json(training_cfg_json_path)
    training_cfg.to_json(training_cfg_export_path)
    training_cfg_reload = TrainingCfg.from_json(training_cfg_export_path)
    assert training_cfg.model == training_cfg_reload.model
    assert training_cfg.dataset == training_cfg_reload.dataset
    assert training_cfg.max_len == training_cfg_reload.max_len
    assert training_cfg.label_noise == training_cfg_reload.label_noise
    assert training_cfg.split_id == training_cfg_reload.split_id
    assert training_cfg.split_prop == training_cfg_reload.split_prop
    assert training_cfg.lora_r == training_cfg_reload.lora_r
    assert training_cfg.lora_alpha == training_cfg_reload.lora_alpha
    assert training_cfg.lora_dropout == training_cfg_reload.lora_dropout
    assert training_cfg.accelerator == training_cfg_reload.accelerator

    # Clean-up
    training_cfg_export_path.unlink()


def test_train_hash():

    assert hash(TrainingCfg()) == hash(TrainingCfg(model="mistral"))
    assert hash(TrainingCfg(model="mistral")) != hash(TrainingCfg(model="llama"))
    assert hash(TrainingCfg(dataset="arc")) != hash(TrainingCfg(dataset="ethics"))
    assert hash(TrainingCfg(max_len=1024)) != hash(TrainingCfg(max_len=512))
    assert hash(TrainingCfg(label_noise=0.0)) != hash(TrainingCfg(label_noise=1.0))
    assert hash(TrainingCfg(split_id=1)) != hash(TrainingCfg(split_id=0))
    assert hash(TrainingCfg(split_prop=0.0)) != hash(TrainingCfg(split_prop=1.0))
    assert hash(TrainingCfg(split_prop=0.0)) == hash(TrainingCfg(split_prop=0.0))
    assert hash(TrainingCfg(lora_r=1)) != hash(TrainingCfg(lora_r=2))
    assert hash(TrainingCfg(lora_alpha=4)) != hash(TrainingCfg(lora_alpha=5))
    assert hash(TrainingCfg(lora_dropout=0.5)) != hash(TrainingCfg(lora_dropout=0.0))


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
