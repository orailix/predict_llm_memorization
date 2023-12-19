"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import shutil
from pathlib import Path

import pytest

from grokking_llm.training import TrainingCfg

test_files = Path(__file__).parent / "files"
training_cfg_path = test_files / "training.cfg"
training_cfg_json_path = test_files / "training_cfg.json"
training_cfg_export_path = test_files / "training_cfg_export.json"


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
    assert training_cfg.get_config_id() == training_cfg_from_cfg.get_config_id()


def test_train_config_export():
    training_cfg = TrainingCfg.from_json(training_cfg_json_path)
    training_cfg.to_json(training_cfg_export_path)
    training_cfg_reload = TrainingCfg.from_json(training_cfg_export_path)
    assert training_cfg.get_config_id() == training_cfg_reload.get_config_id()

    # Clean-up
    training_cfg_export_path.unlink()


def test_train_config_hash():

    assert TrainingCfg().get_config_id() == TrainingCfg(model="mistral").get_config_id()
    assert (
        TrainingCfg(model="mistral").get_config_id()
        != TrainingCfg(model="llama").get_config_id()
    )
    assert (
        TrainingCfg(dataset="arc").get_config_id()
        != TrainingCfg(dataset="ethics").get_config_id()
    )
    assert (
        TrainingCfg(max_len=1024).get_config_id()
        != TrainingCfg(max_len=512).get_config_id()
    )
    assert (
        TrainingCfg(label_noise=0.0).get_config_id()
        != TrainingCfg(label_noise=1.0).get_config_id()
    )
    assert (
        TrainingCfg(split_id=1).get_config_id()
        != TrainingCfg(split_id=0).get_config_id()
    )
    assert (
        TrainingCfg(split_prop=0.0).get_config_id()
        != TrainingCfg(split_prop=1.0).get_config_id()
    )
    assert (
        TrainingCfg(split_prop=0.0).get_config_id()
        == TrainingCfg(split_prop=0.0).get_config_id()
    )
    assert (
        TrainingCfg(lora_r=1).get_config_id() != TrainingCfg(lora_r=2).get_config_id()
    )
    assert (
        TrainingCfg(lora_alpha=4).get_config_id()
        != TrainingCfg(lora_alpha=5).get_config_id()
    )
    assert (
        TrainingCfg(lora_dropout=0.5).get_config_id()
        != TrainingCfg(lora_dropout=0.0).get_config_id()
    )


def test_train_config_output_dir():
    training_cfg_0 = TrainingCfg(model="mistral")
    training_cfg_1 = TrainingCfg(model="mistral")
    training_cfg_2 = TrainingCfg(model="llama")

    dir_0 = training_cfg_0.get_output_dir()
    dir_1 = training_cfg_1.get_output_dir()
    dir_2 = training_cfg_2.get_output_dir()

    # Consistency ?
    assert dir_0 == dir_1
    assert dir_1 != dir_2

    # Well created ?
    assert dir_0.is_dir()
    assert dir_2.is_dir()

    # Config export
    config_reloaded = TrainingCfg.from_json(dir_2 / "training_cfg.json")
    assert config_reloaded.get_config_id() == training_cfg_2.get_config_id()

    shutil.rmtree(dir_0)
    shutil.rmtree(dir_2)


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
