# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import shutil
from pathlib import Path

import numpy as np
import pytest

from grokking_llm.training import TrainingCfg
from grokking_llm.utils import paths
from grokking_llm.utils.constants import TRAIN_CFG_DEFAULT_TRAINING_ARGS

# Test files
training_cfg_path = paths.configs / "training.cfg"
training_cfg_json_path = paths.configs / "training_cfg.json"
training_cfg_export_path = paths.configs / "training_cfg_export.json"


def test_train_config_from_file():
    training_cfg = TrainingCfg.from_cfg(training_cfg_path)
    training_cfg_json = TrainingCfg.from_json(training_cfg_path)

    for item in [training_cfg, training_cfg_json]:
        assert type(item.model) == str and item.model == "mistral"
        assert type(item.dataset) == str and item.dataset == "arc"
        assert type(item.max_len) == int and item.max_len == 1024
        assert type(item.label_noise) == float and item.label_noise == 0.0
        assert type(item.data_seed) == int and item.data_seed == 0
        assert type(item.split_id) == int and item.split_id == 0
        assert type(item.split_prop) == float and item.split_prop == 1.0
        assert type(item.split_test) == bool and item.split_test == False
        assert type(item.lora_r) == int and item.lora_r == 8
        assert type(item.lora_alpha) == float and item.lora_alpha == 16
        assert type(item.lora_dropout) == float and item.lora_dropout == 0.05
        assert type(item.accelerator) == str and item.accelerator == "cpu"
        assert type(item.accelerator) == str and item.accelerator == "cpu"
        assert type(item.last_token_only) == bool and item.last_token_only is False
        assert type(item.training_args) == dict
        assert item.training_args["warmum_steps"] == -100
        assert item.training_args["saving_strategy"] == -100
        assert item.training_args["boolean_value"] is False


def test_train_config_from_file():
    assert TrainingCfg.from_file(training_cfg_json_path) == TrainingCfg.from_json(
        training_cfg_json_path
    )
    assert TrainingCfg.from_file(training_cfg_path) == TrainingCfg.from_cfg(
        training_cfg_path
    )
    assert TrainingCfg.from_file(training_cfg_json_path) == TrainingCfg.from_file(
        training_cfg_path
    )

    with pytest.raises(ValueError):
        TrainingCfg.from_file("hello.yaml")


def test_train_config_export():
    training_cfg = TrainingCfg.from_json(training_cfg_json_path)
    training_cfg.to_json(training_cfg_export_path)
    training_cfg_reload = TrainingCfg.from_json(training_cfg_export_path)
    print(training_cfg)
    print(training_cfg_reload)
    assert training_cfg == training_cfg_reload

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
        TrainingCfg(data_seed=0).get_config_id()
        != TrainingCfg(data_seed=1).get_config_id()
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
        TrainingCfg(split_test=True).get_config_id()
        == TrainingCfg(split_test="true").get_config_id()
    )
    assert (
        TrainingCfg(split_test=True).get_config_id()
        != TrainingCfg(split_test="false").get_config_id()
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
    assert (
        TrainingCfg(last_token_only=False).get_config_id()
        != TrainingCfg(last_token_only=True).get_config_id()
    )
    assert (
        TrainingCfg(training_args=dict(num_train_epochs=1)).get_config_id()
        == TrainingCfg(training_args=dict(num_train_epochs=2)).get_config_id()
    )
    assert (
        TrainingCfg(training_args=dict(per_device_eval_batch_size=1)).get_config_id()
        == TrainingCfg(training_args=dict(per_device_eval_batch_size=2)).get_config_id()
    )
    assert (
        TrainingCfg(training_args=dict(resume_from_checkpoint=True)).get_config_id()
        == TrainingCfg(training_args=dict(resume_from_checkpoint=False)).get_config_id()
    )
    assert (
        TrainingCfg(training_args=dict(a=4)).get_config_id()
        != TrainingCfg().get_config_id()
    )
    assert (
        TrainingCfg(training_args=dict(a=4)).get_config_id()
        != TrainingCfg(training_args=dict(a=5)).get_config_id()
    )


def test_eq_training_config():
    assert TrainingCfg() != "hello"
    assert TrainingCfg(model="mistral") != TrainingCfg(model="llama")
    assert (
        TrainingCfg(training_args=dict(dataloader_persistent_workers=5))
        != TrainingCfg()
    )
    assert TrainingCfg(
        training_args=dict(dataloader_persistent_workers=5)
    ) != TrainingCfg(training_args=dict(dataloader_persistent_workers=4))
    assert TrainingCfg() != TrainingCfg(
        training_args=dict(dataloader_persistent_workers=4)
    )


def test_hash():
    assert hash(TrainingCfg()) != hash("hello")
    assert hash(TrainingCfg(model="mistral")) != hash(TrainingCfg(model="llama"))
    assert hash(
        TrainingCfg(training_args=dict(dataloader_persistent_workers=5))
    ) != hash(TrainingCfg())
    assert hash(
        TrainingCfg(training_args=dict(dataloader_persistent_workers=5))
    ) != hash(TrainingCfg(training_args=dict(dataloader_persistent_workers=4)))
    assert hash(TrainingCfg()) != hash(
        TrainingCfg(training_args=dict(dataloader_persistent_workers=4))
    )


def test_train_config_output_dir():
    training_cfg_0 = TrainingCfg(
        model="mistral", training_args=dict(num_train_epochs=1)
    )
    training_cfg_1 = TrainingCfg(
        model="mistral", training_args=dict(num_train_epochs=2)
    )
    training_cfg_2 = TrainingCfg(model="llama", training_args=dict(num_train_epochs=1))

    dir_0 = training_cfg_0.get_output_dir()
    dir_1 = training_cfg_1.get_output_dir()
    dir_2 = training_cfg_2.get_output_dir()

    # Remove directory if they already existed
    for dir_ in [dir_0, dir_1, dir_2]:
        if dir_.is_dir():
            shutil.rmtree(dir_)

    # Sync dir 0
    assert not (paths.individual_outputs / training_cfg_0.get_config_id()).is_dir()
    dir_0 = training_cfg_0.get_output_dir()
    config_reloaded = TrainingCfg.from_json(dir_0 / "training_cfg.json")
    assert dir_0.is_dir()
    assert dir_0 == paths.individual_outputs / training_cfg_0.get_config_id()
    assert config_reloaded == training_cfg_0
    assert config_reloaded.training_args["num_train_epochs"] == 1

    # Sync dir 1
    dir_1 = training_cfg_1.get_output_dir()
    config_reloaded = TrainingCfg.from_json(dir_1 / "training_cfg.json")
    assert dir_1 == dir_0
    assert config_reloaded == training_cfg_1
    assert config_reloaded.training_args["num_train_epochs"] == 2

    # Sync dir 2
    assert not (paths.individual_outputs / training_cfg_2.get_config_id()).is_dir()
    dir_2 = training_cfg_2.get_output_dir()
    config_reloaded = TrainingCfg.from_json(dir_2 / "training_cfg.json")
    assert dir_2.is_dir()
    assert dir_2 != dir_0
    assert dir_2 == paths.individual_outputs / training_cfg_2.get_config_id()
    assert config_reloaded == training_cfg_2
    assert config_reloaded.training_args["num_train_epochs"] == 1

    # Cleaning
    for dir_ in [dir_0, dir_1, dir_2]:
        if dir_.is_dir():
            shutil.rmtree(dir_)


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


def test_train_config_label_noise():
    with pytest.raises(ValueError):
        TrainingCfg(data_seed="hello")


def test_train_config_split_id():
    with pytest.raises(ValueError):
        TrainingCfg(split_id="hello")


def test_train_config_split_prop():
    with pytest.raises(ValueError):
        TrainingCfg(split_prop=-0.1)

    with pytest.raises(ValueError):
        TrainingCfg(split_prop=1.1)


def test_train_config_split_test():
    with pytest.raises(ValueError):
        TrainingCfg(split_test=1.0)

    with pytest.raises(ValueError):
        TrainingCfg(split_test="hello")


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


def test_train_config_last_token_only():
    with pytest.raises(ValueError):
        TrainingCfg(last_token_only=1)

    with pytest.raises(ValueError):
        TrainingCfg(last_token_only="False")


def test_train_config_training_args():
    with pytest.raises(ValueError):
        TrainingCfg(training_args={1: 2})

    custom_args = TRAIN_CFG_DEFAULT_TRAINING_ARGS.copy()
    for key in custom_args:
        custom_args[key] = np.random.randint(1e6)

    for _ in range(10):
        custom_args[str(np.random.randint(1e6))] = np.random.randint(1e6)

    cfg = TrainingCfg(training_args=custom_args)

    for key, value in custom_args.items():
        assert key in cfg.training_args
        assert cfg.training_args[key] == value


def test_resume_from_checkpoint_boolean_false():
    cfg = TrainingCfg()
    cfg.training_args["resume_from_checkpoint"] = False

    # Pre-test cleaning
    shutil.rmtree(cfg.get_output_dir())
    output_dir = cfg.get_output_dir()

    # Tests
    assert cfg.get_resume_from_checkpoint_status() is False
    (output_dir / "checkpoint-0").mkdir()
    assert cfg.get_resume_from_checkpoint_status() is False

    # Cleaning
    shutil.rmtree(cfg.get_output_dir())


def test_resume_from_checkpoint_boolean_true():
    cfg = TrainingCfg()
    cfg.training_args["resume_from_checkpoint"] = True

    # Pre-test cleaning
    shutil.rmtree(cfg.get_output_dir())
    output_dir = cfg.get_output_dir()

    # Tests
    assert cfg.get_resume_from_checkpoint_status() is False
    (output_dir / "checkpoint-0").mkdir()
    assert cfg.get_resume_from_checkpoint_status() is True
    (output_dir / "checkpoint-1").mkdir()
    assert cfg.get_resume_from_checkpoint_status() is True

    # Cleaning
    shutil.rmtree(cfg.get_output_dir())


def test_resume_from_checkpoint_non_boolean():
    cfg = TrainingCfg()
    cfg.training_args["resume_from_checkpoint"] = "/a/fictional/path/to/checkpoint"

    # Pre-test cleaning
    shutil.rmtree(cfg.get_output_dir())
    output_dir = cfg.get_output_dir()

    # Tests
    assert cfg.get_resume_from_checkpoint_status() == "/a/fictional/path/to/checkpoint"
    (output_dir / "checkpoint-0").mkdir()
    assert cfg.get_resume_from_checkpoint_status() == "/a/fictional/path/to/checkpoint"

    # Cleaning
    shutil.rmtree(cfg.get_output_dir())


def test_autoconfig():

    # Cleaning
    for child in paths.individual_outputs.iterdir():
        if child.is_dir():
            try:
                shutil.rmtree(child)
            except OSError:
                pass

    # Autoconfig -- path
    assert TrainingCfg.autoconfig(training_cfg_path) == TrainingCfg.from_file(
        training_cfg_path
    )
    assert TrainingCfg.autoconfig(str(training_cfg_path)) == TrainingCfg.from_file(
        training_cfg_path
    )
    assert TrainingCfg.autoconfig("training") == TrainingCfg.from_file(
        paths.configs / "training.cfg"
    )
    assert TrainingCfg.autoconfig("training.cfg") == TrainingCfg.from_file(
        paths.configs / "training.cfg"
    )

    # Creating output dir for a config
    TrainingCfg(model="mistral", split_prop=0.3).get_output_dir()
    TrainingCfg(model="mistral", split_prop=0.31).get_output_dir()
    TrainingCfg(model="mistral", split_prop=0.314).get_output_dir()
    cfg = TrainingCfg(model="mistral", split_prop=0.3141)
    cfg.get_output_dir()
    cfg_id = cfg.get_config_id()
    assert TrainingCfg.autoconfig(cfg_id[:4]) == cfg
    assert TrainingCfg.autoconfig(cfg_id[:6]) == cfg
    assert TrainingCfg.autoconfig(cfg_id[:8]) == cfg
    assert TrainingCfg.autoconfig(cfg_id) == cfg

    with pytest.raises(ValueError):
        TrainingCfg.autoconfig(cfg_id[:3])

    with pytest.raises(ValueError):
        TrainingCfg.autoconfig(str(np.random.randint(1e6)))

    # Cleaning
    for child in paths.individual_outputs.iterdir():
        if child.is_dir():
            try:
                shutil.rmtree(child)
            except OSError:
                pass
