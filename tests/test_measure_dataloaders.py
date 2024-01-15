# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import shutil
import typing as t

import pytest

from grokking_llm.measures import get_dataloaders_for_measures
from grokking_llm.training import (
    TrainingCfg,
    format_dataset,
    get_dataset,
    get_random_split,
)


def test_get_measure_dataloaders():

    # Training cfg
    cfg = TrainingCfg(label_noise=0.5)

    # Raw dataset
    raw_test_dataset = get_dataset(cfg, split="test")
    raw_train_dataset = get_dataset(cfg, split="train")
    raw_train_dataset = format_dataset(raw_train_dataset, cfg=cfg)
    raw_train_dataset = get_random_split(raw_train_dataset, cfg=cfg)

    # Dataloaders
    train_trl_dl, train_rdl_dl, test_all_dl = get_dataloaders_for_measures(cfg)

    # Check lengths
    assert len(train_trl_dl) + len(train_rdl_dl) == len(raw_train_dataset)
    assert len(test_all_dl) == len(raw_test_dataset)
    assert (
        abs(len(train_rdl_dl) - len(train_trl_dl))
        / max(len(train_rdl_dl), len(train_trl_dl))
        < 0.1
    )

    # Check columns
    assert sorted(list(next(iter(train_trl_dl)))) == sorted(
        [
            "input_ids",
            "attention_mask",
            "labels",
            "tokenized_possible_labels",
            "inserted_label_index",
        ]
    )
    assert sorted(list(next(iter(train_rdl_dl)))) == sorted(
        [
            "input_ids",
            "attention_mask",
            "labels",
            "tokenized_possible_labels",
            "inserted_label_index",
        ]
    )
    assert sorted(list(next(iter(test_all_dl)))) == sorted(
        [
            "input_ids",
            "attention_mask",
            "labels",
            "tokenized_possible_labels",
            "inserted_label_index",
        ]
    )
