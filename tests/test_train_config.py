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
    assert type(training_cfg.r) == int
    assert type(training_cfg.alpha) == float


def test_train_config_model():
    with pytest.raises(ValueError):
        TrainingCfg(
            model="hello",
            dataset=TRAIN_CFG_MMLU,
            max_len=1024,
            label_noise=0.0,
            r=8,
            alpha=16,
        )


def test_train_config_dataset():
    with pytest.raises(ValueError):
        TrainingCfg(
            model=TRAIN_CFG_MISTRAL,
            dataset="hello",
            max_len=1024,
            label_noise=0.0,
            r=8,
            alpha=16,
        )


def test_train_config_max_len():
    with pytest.raises(ValueError):
        TrainingCfg(
            model=TRAIN_CFG_MISTRAL,
            dataset=TRAIN_CFG_MMLU,
            max_len=0,
            label_noise=0.0,
            r=8,
            alpha=16,
        )


def test_train_config_label_noise():
    with pytest.raises(ValueError):
        TrainingCfg(
            model=TRAIN_CFG_MISTRAL,
            dataset=TRAIN_CFG_MMLU,
            max_len=1024,
            label_noise=-0.1,
            r=8,
            alpha=16,
        )

    with pytest.raises(ValueError):
        TrainingCfg(
            model=TRAIN_CFG_MISTRAL,
            dataset=TRAIN_CFG_MMLU,
            max_len=1024,
            label_noise=1.1,
            r=8,
            alpha=16,
        )


def test_train_config_r():
    with pytest.raises(ValueError):
        TrainingCfg(
            model=TRAIN_CFG_MISTRAL,
            dataset=TRAIN_CFG_MMLU,
            max_len=1024,
            label_noise=0.0,
            r=0,
            alpha=16,
        )


def test_train_config_alpha():
    with pytest.raises(ValueError):
        TrainingCfg(
            model=TRAIN_CFG_MISTRAL,
            dataset=TRAIN_CFG_MMLU,
            max_len=1024,
            label_noise=0.0,
            r=8,
            alpha=0,
        )
