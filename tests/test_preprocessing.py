from grokking_llm.training import TrainingCfg
from grokking_llm.training.datasets import format_dataset, get_dataset
from grokking_llm.training.formatting import format_arc, format_ethics, format_mmlu
from grokking_llm.utils.constants import DATASET_BARE_LABEL

# Configs
ethics_cfg = TrainingCfg(dataset="ethics")
mmlu_cfg = TrainingCfg(dataset="mmlu")
arc_cfg = TrainingCfg(dataset="arc")


# Datasets
def test_datasets_creation():
    ethics_ds_train = get_dataset(ethics_cfg, split="train")
    ethics_ds_test = get_dataset(ethics_cfg, split="test")
    mmlu_ds_train = get_dataset(mmlu_cfg, split="train")
    mmlu_ds_test = get_dataset(mmlu_cfg, split="test")
    arc_ds_train = get_dataset(arc_cfg, split="train")
    arc_ds_test = get_dataset(arc_cfg, split="test")
    assert True


def test_formatting_ethics():
    ethics_ds_train = get_dataset(ethics_cfg, split="train")
    sample = ethics_ds_train[0]
    formatted = format_ethics(sample)
    assert "prompt" in formatted
    assert type(formatted["prompt"] == str)
    assert "label" in formatted
    assert type(formatted["label"] == str)
    assert "possible_labels" in formatted
    assert type(formatted["possible_labels"] == list)
    assert "label_status" in formatted
    assert formatted["label_status"] == DATASET_BARE_LABEL


def test_formatting_mmlu():
    mmlu_ds_train = get_dataset(mmlu_cfg, split="train")
    sample = mmlu_ds_train[0]
    formatted = format_mmlu(sample)
    assert "prompt" in formatted
    assert type(formatted["prompt"] == str)
    assert "label" in formatted
    assert type(formatted["label"] == str)
    assert "possible_labels" in formatted
    assert type(formatted["possible_labels"] == list)
    assert "label_status" in formatted
    assert formatted["label_status"] == DATASET_BARE_LABEL


def test_formatting_arc():
    arc_ds_train = get_dataset(arc_cfg, split="train")
    sample = arc_ds_train[0]
    formatted = format_arc(sample)
    assert "prompt" in formatted
    assert type(formatted["prompt"] == str)
    assert "label" in formatted
    assert type(formatted["label"] == str)
    assert "possible_labels" in formatted
    assert type(formatted["possible_labels"] == list)
    assert "label_status" in formatted
    assert formatted["label_status"] == DATASET_BARE_LABEL


def test_map_ethics_formatting():
    ethics_ds_train = get_dataset(ethics_cfg, split="test")
    formatted_ds = format_dataset(ethics_ds_train, ethics_cfg)

    # Quality tests
    assert len(ethics_ds_train) == len(formatted_ds)
    assert formatted_ds[0]["label_status"] == DATASET_BARE_LABEL
