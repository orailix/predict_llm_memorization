"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import collections

import numpy as np

from grokking_llm.training import TrainingCfg
from grokking_llm.training.datasets import (
    add_labels,
    format_dataset,
    get_dataset,
    get_random_split,
)
from grokking_llm.training.formatting import format_arc, format_ethics, format_mmlu
from grokking_llm.utils.constants import (
    DATASET_BARE_LABEL,
    DATASET_RANDOM_LABEL,
    DATASET_TRUE_LABEL,
)


# Datasets
def test_datasets_creation():
    # Configs
    ethics_cfg = TrainingCfg(dataset="ethics")
    mmlu_cfg = TrainingCfg(dataset="mmlu")
    arc_cfg = TrainingCfg(dataset="arc")

    # Datasets
    get_dataset(ethics_cfg, split="train")
    get_dataset(ethics_cfg, split="test")
    get_dataset(mmlu_cfg, split="train")
    get_dataset(mmlu_cfg, split="test")
    get_dataset(arc_cfg, split="train")
    get_dataset(arc_cfg, split="test")
    assert True


def test_formatting_ethics():
    ethics_cfg = TrainingCfg(dataset="ethics")
    ethics_ds_test = get_dataset(ethics_cfg, split="test")

    for _ in range(30):
        sample = ethics_ds_test[np.random.randint(len(ethics_ds_test))]
        formatted = format_ethics(sample)
        assert "prompt" in formatted
        assert type(formatted["prompt"]) == str
        assert "label" in formatted
        assert type(formatted["label"]) == str
        assert "possible_labels" in formatted
        assert type(formatted["possible_labels"]) == list
        assert "label_status" in formatted
        assert formatted["label_status"] == DATASET_BARE_LABEL
        assert "index" in formatted
        assert type(formatted["index"]) == int

        # Test determinism
        formatted = format_ethics(sample, random_state=np.random.RandomState(seed=42))
        for _ in range(10):
            new_formatted = format_ethics(
                sample, random_state=np.random.RandomState(seed=42)
            )
            assert new_formatted["prompt"] == formatted["prompt"]


def test_formatting_mmlu():
    mmlu_cfg = TrainingCfg(dataset="mmlu")
    mmlu_ds_test = get_dataset(mmlu_cfg, split="test")

    for _ in range(30):
        sample = mmlu_ds_test[np.random.randint(len(mmlu_ds_test))]
        formatted = format_mmlu(sample)
        assert "prompt" in formatted
        assert type(formatted["prompt"]) == str
        assert "label" in formatted
        assert type(formatted["label"]) == str
        assert "possible_labels" in formatted
        assert type(formatted["possible_labels"]) == list
        assert "label_status" in formatted
        assert formatted["label_status"] == DATASET_BARE_LABEL
        assert "index" in formatted
        assert type(formatted["index"]) == int

        # Test determinism
        formatted = format_mmlu(sample, random_state=np.random.RandomState(seed=42))
        for _ in range(30):
            new_formatted = format_mmlu(
                sample, random_state=np.random.RandomState(seed=42)
            )
            assert new_formatted["prompt"] == formatted["prompt"]


def test_formatting_arc():
    arc_cfg = TrainingCfg(dataset="arc")
    arc_ds_test = get_dataset(arc_cfg, split="test")

    for _ in range(30):
        sample = arc_ds_test[np.random.randint(len(arc_ds_test))]
        formatted = format_arc(sample)
        assert "prompt" in formatted
        assert type(formatted["prompt"]) == str
        assert "label" in formatted
        assert type(formatted["label"]) == str
        assert "possible_labels" in formatted
        assert type(formatted["possible_labels"]) == list
        assert "label_status" in formatted
        assert formatted["label_status"] == DATASET_BARE_LABEL
        assert "index" in formatted
        assert type(formatted["index"]) == int

        # Test determinism
        formatted = format_arc(sample, random_state=np.random.RandomState(seed=42))
        for _ in range(30):
            new_formatted = format_arc(
                sample, random_state=np.random.RandomState(seed=42)
            )
            assert new_formatted["prompt"] == formatted["prompt"]


def test_map_ethics_formatting():
    ethics_cfg = TrainingCfg(dataset="ethics")
    ethics_ds_test = get_dataset(ethics_cfg, split="test")
    formatted_ds = format_dataset(ethics_ds_test, ethics_cfg)

    # Quality tests
    assert len(ethics_ds_test) == len(formatted_ds)
    assert formatted_ds[0]["label_status"] == DATASET_BARE_LABEL


def test_map_ethics_formatting_determinism():
    ethics_cfg = TrainingCfg(dataset="ethics")
    ethics_ds_test = get_dataset(ethics_cfg, split="test")

    # With RandomState
    formatted_ds_0 = format_dataset(ethics_ds_test, ethics_cfg, seed=42)
    formatted_ds_1 = format_dataset(ethics_ds_test, ethics_cfg, seed=42)

    # With force_template
    formatted_ds_2 = format_dataset(ethics_ds_test, ethics_cfg, force_template=True)
    formatted_ds_3 = format_dataset(ethics_ds_test, ethics_cfg, force_template=True)

    # Quality tests
    for _ in range(30):
        idx = np.random.randint(len(formatted_ds_0))
        assert formatted_ds_0[idx]["prompt"] == formatted_ds_1[idx]["prompt"]
        assert formatted_ds_2[idx]["prompt"] == formatted_ds_3[idx]["prompt"]


def test_add_labels_to_dataset():
    ethics_cfg = TrainingCfg(dataset="ethics", label_noise=0.0)
    ethics_ds_test = get_dataset(ethics_cfg, split="test")
    formatted = format_dataset(ethics_ds_test, ethics_cfg, seed=42)

    # With label_noise = 0.0
    labelled_ds = add_labels(formatted, ethics_cfg)
    for _ in range(30):
        idx = np.random.randint(len(labelled_ds))
        assert labelled_ds[idx]["label_status"] == DATASET_TRUE_LABEL
        assert labelled_ds[idx]["prompt"][-1] == str(labelled_ds[idx]["label"])
        assert str(labelled_ds[idx]["label"]) in labelled_ds[idx]["possible_labels"]

    # With label_noise = 0.5
    ethics_cfg.label_noise = 0.5
    labelled_ds = add_labels(formatted, ethics_cfg, seed=42)

    # Correct label flips ?
    for _ in range(30):
        idx = np.random.randint(len(labelled_ds))
        assert labelled_ds[idx]["prompt"][-1] == str(labelled_ds[idx]["label"])
        assert str(labelled_ds[idx]["label"]) in labelled_ds[idx]["possible_labels"]

    # Counting label flips
    count = collections.defaultdict(int)
    for sample in labelled_ds:
        count[sample["label_status"]] += 1

    assert len(count) == 2  # Only DATASET_RANDOM_LABEL and DATASET_TRUE_LABEL
    assert count[DATASET_RANDOM_LABEL] >= 0.9 * 0.5 * (
        count[DATASET_RANDOM_LABEL] + count[DATASET_TRUE_LABEL]
    )
    assert count[DATASET_RANDOM_LABEL] <= 1.1 * 0.5 * (
        count[DATASET_RANDOM_LABEL] + count[DATASET_TRUE_LABEL]
    )
    assert count[DATASET_TRUE_LABEL] >= 0.9 * 0.5 * (
        count[DATASET_RANDOM_LABEL] + count[DATASET_TRUE_LABEL]
    )
    assert count[DATASET_TRUE_LABEL] <= 1.1 * 0.5 * (
        count[DATASET_RANDOM_LABEL] + count[DATASET_TRUE_LABEL]
    )


def test_add_label_determinism():
    ethics_cfg = TrainingCfg(dataset="ethics", label_noise=0.5)
    ethics_ds_test = get_dataset(ethics_cfg, split="test")
    formatted = format_dataset(ethics_ds_test, ethics_cfg, seed=42)

    # Labelling
    labelled_ds_0 = add_labels(formatted, ethics_cfg, seed=42)
    labelled_ds_1 = add_labels(formatted, ethics_cfg, seed=42)

    for _ in range(30):
        idx = np.random.randint(len(labelled_ds_0))
        assert labelled_ds_0[idx] == labelled_ds_1[idx]


def test_dataset_splits():
    ethics_cfg = TrainingCfg(dataset="ethics", split_prop=0.25, split_id=0)
    ethics_ds_test = get_dataset(ethics_cfg, split="test")
    formatted = format_dataset(ethics_ds_test, ethics_cfg, seed=42)
    with_labels = add_labels(formatted, ethics_cfg, seed=42)

    # No noise
    split_0 = get_random_split(with_labels, cfg=ethics_cfg)
    split_0_again = get_random_split(with_labels, cfg=ethics_cfg)
    ethics_cfg.split_id = 42
    split_42 = get_random_split(with_labels, cfg=ethics_cfg)
    ethics_cfg.split_prop = 0.5
    split_42_large = get_random_split(with_labels, cfg=ethics_cfg)

    # Split sizes
    assert abs(len(split_0) - (0.25 * len(with_labels))) <= 1.0
    assert len(split_0_again) == len(split_0)
    assert len(split_42) == len(split_0)
    assert abs(len(split_42_large) - (0.5 * len(with_labels))) <= 1.0

    # Quality
    unique_samples = set()
    for item in split_0:
        assert (
            type(item["index"]) == int
            and item["index"] >= 0
            and item["index"] < len(with_labels)
        )
        unique_samples.add(item["index"])

    assert len(unique_samples) == len(split_0)

    # Reproducibility
    for _ in range(30):
        idx = np.random.randint(len(split_0))
        assert split_0[idx] == split_0_again[idx]

    # Difference between seed 0 and 42
    are_the_same = True
    for idx in range(len(split_0)):
        if split_0[idx] != split_42[idx]:
            are_the_same = False
            break
    assert not are_the_same
