# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import json
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from grokking_llm.training import TrainingCfg
from grokking_llm.training.datasets import (
    add_labels,
    add_tokenized_possible_labels,
    format_dataset,
    get_dataset,
    get_random_split,
    get_tokenizer,
    tokenize_dataset,
)
from grokking_llm.training.formatting import format_arc, format_ethics, format_mmlu
from grokking_llm.utils import paths
from grokking_llm.utils.constants import (
    DATASET_BARE_LABEL,
    DATASET_RANDOM_LABEL,
    DATASET_TRUE_LABEL,
    MAX_NUM_MCQ_ANSWER,
)

# Test files
ethics_global_index = paths.configs / "ethics_global_index.json"


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

    # Wrong split
    with pytest.raises(ValueError):
        get_dataset(ethics_cfg, split="hello")

    assert True


def test_datasets_train_test():

    # Configs
    ethics_cfg = TrainingCfg(dataset="ethics")

    # Creation
    ds_train = get_dataset(ethics_cfg, split="train")
    ds_test = get_dataset(ethics_cfg, split="test")
    ds_train_hf = get_dataset(ethics_cfg, split="train_hf")
    ds_test_hf = get_dataset(ethics_cfg, split="test_hf")

    # Tests -- length
    assert abs(len(ds_test) - 0.125 * len(ds_train)) <= 1

    # Groundtruth global indices
    with ethics_global_index.open("r") as f:
        groundtruth_indices = json.load(f)

    # GLobal indices
    train_indices = sorted([item["global_index"] for item in ds_train])
    test_indices = sorted([item["global_index"] for item in ds_test])
    train_hf_indices = sorted([item["global_index"] for item in ds_train_hf])
    test_hf_indices = sorted([item["global_index"] for item in ds_test_hf])

    # Tests -- global indices
    assert train_indices == sorted(groundtruth_indices["train"])
    assert test_indices == sorted(groundtruth_indices["test"])
    assert train_hf_indices == list(range(len(ds_train_hf)))
    assert test_hf_indices == list(
        range(len(ds_train_hf), len(ds_train_hf) + len(ds_test_hf))
    )


def test_formatting_ethics():
    ethics_cfg = TrainingCfg(dataset="ethics")
    ethics_ds_test = get_dataset(ethics_cfg, split="test")

    for _ in range(30):
        sample = ethics_ds_test[np.random.randint(len(ethics_ds_test))]
        formatted = format_ethics(sample)
        assert "prompt" in formatted
        assert type(formatted["prompt"]) == str
        assert "cls_label" in formatted
        assert type(formatted["cls_label"]) == str
        assert "possible_cls_labels" in formatted
        assert type(formatted["possible_cls_labels"]) == list
        assert "cls_label_status" in formatted
        assert formatted["cls_label_status"] == DATASET_BARE_LABEL
        assert "global_index" in formatted
        assert type(formatted["global_index"]) == int

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
        assert "cls_label" in formatted
        assert type(formatted["cls_label"]) == str
        assert "possible_cls_labels" in formatted
        assert type(formatted["possible_cls_labels"]) == list
        assert "cls_label_status" in formatted
        assert formatted["cls_label_status"] == DATASET_BARE_LABEL
        assert "global_index" in formatted
        assert type(formatted["global_index"]) == int

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
        assert "cls_label" in formatted
        assert type(formatted["cls_label"]) == str
        assert "possible_cls_labels" in formatted
        assert type(formatted["possible_cls_labels"]) == list
        assert "cls_label_status" in formatted
        assert formatted["cls_label_status"] == DATASET_BARE_LABEL
        assert "global_index" in formatted
        assert type(formatted["global_index"]) == int

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
    assert formatted_ds[0]["cls_label_status"] == DATASET_BARE_LABEL


def test_map_ethics_formatting_determinism():
    ethics_cfg = TrainingCfg(dataset="ethics")
    ethics_ds_test = get_dataset(ethics_cfg, split="test")

    # With RandomState
    formatted_ds_0 = format_dataset(ethics_ds_test, ethics_cfg)
    formatted_ds_1 = format_dataset(ethics_ds_test, ethics_cfg)

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
    labelled_ds = add_labels(formatted, ethics_cfg, "train")
    for _ in range(30):
        idx = np.random.randint(len(labelled_ds))
        assert labelled_ds[idx]["cls_label_status"] == DATASET_TRUE_LABEL
        assert labelled_ds[idx]["prompt"][-1] == str(labelled_ds[idx]["cls_label"])
        assert (
            str(labelled_ds[idx]["cls_label"])
            in labelled_ds[idx]["possible_cls_labels"]
        )

    # With label_noise = 0.5 -- test mode (no label flip expected)
    ethics_cfg.label_noise = 0.5
    labelled_ds = add_labels(formatted, ethics_cfg, "test", seed=42)
    for _ in range(30):
        idx = np.random.randint(len(labelled_ds))
        assert labelled_ds[idx]["cls_label_status"] == DATASET_TRUE_LABEL
        assert labelled_ds[idx]["prompt"][-1] == str(labelled_ds[idx]["cls_label"])
        assert (
            str(labelled_ds[idx]["cls_label"])
            in labelled_ds[idx]["possible_cls_labels"]
        )

    # With label_noise = 0.5 -- train mode
    ethics_cfg.label_noise = 0.5
    labelled_ds = add_labels(formatted, ethics_cfg, "train", seed=42)

    # Correct label flips ?
    for _ in range(30):
        idx = np.random.randint(len(labelled_ds))
        assert labelled_ds[idx]["prompt"][-1] == str(labelled_ds[idx]["cls_label"])
        assert (
            str(labelled_ds[idx]["cls_label"])
            in labelled_ds[idx]["possible_cls_labels"]
        )

    # Counting label flips
    count = collections.defaultdict(int)
    for sample in labelled_ds:
        count[sample["cls_label_status"]] += 1

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
    formatted = format_dataset(ethics_ds_test, ethics_cfg)

    # Labelling
    labelled_ds_0 = add_labels(formatted, ethics_cfg, "train")
    labelled_ds_1 = add_labels(formatted, ethics_cfg, "train")

    for _ in range(30):
        idx = np.random.randint(len(labelled_ds_0))
        assert labelled_ds_0[idx] == labelled_ds_1[idx]


def test_dataset_splits():
    ethics_cfg = TrainingCfg(dataset="ethics", split_prop=0.25, split_id=0)
    ethics_ds_test = get_dataset(ethics_cfg, split="test")
    formatted = format_dataset(ethics_ds_test, ethics_cfg, seed=42)
    with_labels = add_labels(formatted, ethics_cfg, "train", seed=42)

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
    with_labels_indices = set(with_labels["global_index"])
    for item in split_0:
        assert (
            type(item["global_index"]) == int
            and item["global_index"] in with_labels_indices
        )
        unique_samples.add(item["global_index"])

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


def test_get_tokenizer():
    get_tokenizer(TrainingCfg(dataset="ethics"))
    get_tokenizer(TrainingCfg(model="dummy_llama"))


def test_dataset_tokenization():
    ethics_cfg = TrainingCfg(dataset="ethics", split_prop=0.1, split_id=0)
    ethics_ds_test = get_dataset(ethics_cfg, split="test")
    formatted = format_dataset(ethics_ds_test, ethics_cfg, seed=42)
    with_labels = add_labels(formatted, ethics_cfg, "train", seed=42)
    split = get_random_split(with_labels, cfg=ethics_cfg)

    # Tokenization
    tokenizer_obj = AutoTokenizer.from_pretrained(ethics_cfg.model)
    tokenized = tokenize_dataset(split, cfg=ethics_cfg)

    # Quality checks
    assert len(tokenized) == len(split)
    for _ in range(30):
        idx = np.random.randint(len(split))

        # Sanity checks
        assert "input_ids" in tokenized[idx]
        assert "input_ids" not in split[idx]
        assert "labels" in tokenized[idx]
        assert "labels" not in split[idx]
        assert "attention_mask" in tokenized[idx]
        assert "attention_mask" not in split[idx]
        assert tokenized[idx]["input_ids"] == tokenized[idx]["labels"]

        # BOS token and padding
        assert 1 in tokenized[idx]["input_ids"]
        bos_idx = tokenized[idx]["input_ids"].index(1)
        for _ in range(30):
            random_pad_idx = np.random.randint(bos_idx)
            assert (
                tokenized[idx]["input_ids"][random_pad_idx]
                == tokenizer_obj.eos_token_id
            )

        # CLS label and EOS token
        assert tokenized[idx]["input_ids"][-1] == tokenizer_obj.eos_token_id
        assert (
            tokenized[idx]["input_ids"][-2]
            == tokenizer_obj.encode(tokenized[idx]["cls_label"])[-1]
        )

        # Length check
        assert len(tokenized[idx]["input_ids"]) == ethics_cfg.max_len

    # Truncation checks
    LEN = 5
    ethics_cfg.max_len = LEN
    shortly_tokenized = tokenize_dataset(split, cfg=ethics_cfg)

    for _ in range(30):
        idx = np.random.randint(len(split))

        # Length checks
        assert len(shortly_tokenized[idx]["input_ids"]) == ethics_cfg.max_len

        # Trucation side check
        # The first token differs, because it is the BOS token.
        assert (
            shortly_tokenized[idx]["input_ids"][1:]
            == tokenized[idx]["input_ids"][-LEN + 1 :]
        )


def test_add_tokenized_possible_labels():

    # Preparing
    ethics_cfg = TrainingCfg(dataset="ethics", split_prop=0.1, split_id=0)
    ethics_ds_test = get_dataset(ethics_cfg, split="test")
    formatted = format_dataset(ethics_ds_test, ethics_cfg, seed=42)
    with_labels = add_labels(formatted, ethics_cfg, "train", seed=42)
    split = get_random_split(with_labels, cfg=ethics_cfg)
    tokenized = tokenize_dataset(split, cfg=ethics_cfg)

    # Tokenizer
    tokenizer_obj = AutoTokenizer.from_pretrained(ethics_cfg.model)

    # APplying fct to test
    with_possible_labels = add_tokenized_possible_labels(tokenized, cfg=ethics_cfg)

    # Quality checks
    assert len(with_possible_labels) == len(tokenized)
    for _ in range(30):
        idx = np.random.randint(len(split))

        # Sanity checks -- column names
        assert "tokenized_possible_labels" in with_possible_labels[idx]
        assert "inserted_label_index" in with_possible_labels[idx]
        assert "tokenized_possible_labels" not in tokenized[idx]
        assert "inserted_label_index" not in tokenized[idx]

        # Unpacking
        tokenized_possible_labels = with_possible_labels[idx][
            "tokenized_possible_labels"
        ]
        inserted_label_index = with_possible_labels[idx]["inserted_label_index"]
        possible_cls_labels = with_possible_labels[idx]["possible_cls_labels"]

        # Type and length checks
        assert (
            isinstance(tokenized_possible_labels, list)
            and len(tokenized_possible_labels) == MAX_NUM_MCQ_ANSWER
        )
        assert isinstance(inserted_label_index, int)

        # Check inserted value
        assert (
            tokenized_possible_labels[inserted_label_index]
            == with_possible_labels[idx]["input_ids"][-2]
        )

        # Check tokenized values of possible answers
        for label in possible_cls_labels:
            assert tokenizer_obj.encode("\n\n" + label)[-1] in tokenized_possible_labels

        assert tokenized_possible_labels.count(0) == max(
            0, MAX_NUM_MCQ_ANSWER - len(possible_cls_labels)
        )
