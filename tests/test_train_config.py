"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import pytest

from grokking_llm.training import TrainingCfg
from grokking_llm.utils.constants import TRAIN_CFG_MISTRAL, TRAIN_CFG_MMLU
from grokking_llm.utils.paths import training_cfg_path


def test_train_config():
    training_cfg = TrainingCfg.from_file(training_cfg_path)
    assert type(training_cfg.model) == str
    assert type(training_cfg.dataset) == str
    assert type(training_cfg.max_len) == int
    assert type(training_cfg.label_noise) == float
    assert type(training_cfg.split_id) == int
    assert type(training_cfg.split_prop) == float
    assert type(training_cfg.r) == int
    assert type(training_cfg.alpha) == float


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


def test_train_config_r():
    with pytest.raises(ValueError):
        TrainingCfg(r=0)


def test_train_config_alpha():
    with pytest.raises(ValueError):
        TrainingCfg(alpha=0)
