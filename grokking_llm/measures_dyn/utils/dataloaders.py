# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator

from ...training import (
    add_labels,
    add_tokenized_possible_labels,
    format_dataset,
    get_dataset,
    get_random_split,
    tokenize_dataset,
)
from ...utils import TrainingCfg


def get_dataloaders_for_measures(
    training_cfg: TrainingCfg, full_dataset: bool = False
) -> t.Tuple[DataLoader, DataLoader, DataLoader]:
    """Gets the train_trl, train_rdl and test dataloader.

    Definitions:
        - [train_all] The train set
        - [train_trl] The train set with TRue Labels
        - [train_rdl] The train set with RanDom Labels
        - [test] The test set

    The train_all dataloader is not relevant, because we can simply
    add the results with train_trl and train_rdl.

    If `full_dataset` is True, we skip the call to `get_random_split`,
    so the dataloaders correspond to the full dataset."""

    # DATASET -- TRAIN
    train_dataset = get_dataset(training_cfg, split="train")
    train_dataset_formatted = format_dataset(train_dataset, training_cfg)
    train_dataset_labelled = add_labels(train_dataset_formatted, training_cfg, "train")
    if full_dataset:
        train_dataset_split = train_dataset_labelled
    else:
        train_dataset_split = get_random_split(train_dataset_labelled, training_cfg)
    train_dataset_tokenized = tokenize_dataset(train_dataset_split, training_cfg)
    train_dataset_complete = add_tokenized_possible_labels(
        train_dataset_tokenized, training_cfg
    )

    # all / trl / rdl
    trl_selector = [
        i for i, x in enumerate(train_dataset_complete["cls_label_status"]) if x == 1
    ]
    rdl_selector = [
        i for i, x in enumerate(train_dataset_complete["cls_label_status"]) if x == 0
    ]
    train_all = train_dataset_complete.select_columns(
        [
            "input_ids",
            "attention_mask",
            "labels",
            "tokenized_possible_labels",
            "inserted_label_index",
            "global_index",
        ]
    )
    train_trl = train_all.select(trl_selector)
    train_rdl = train_all.select(rdl_selector)

    # DATASET -- TEST
    test_dataset = get_dataset(training_cfg, split="test")
    test_dataset_formatted = format_dataset(test_dataset, training_cfg)
    test_dataset_labelled = add_labels(test_dataset_formatted, training_cfg, "test")
    if training_cfg.split_test:
        test_dataset_labelled = get_random_split(test_dataset_labelled, training_cfg)
    test_dataset_tokenized = tokenize_dataset(test_dataset_labelled, training_cfg)
    test_dataset_complete = add_tokenized_possible_labels(
        test_dataset_tokenized, training_cfg
    )

    # all / trl / rdl
    test_all = test_dataset_complete.select_columns(
        [
            "input_ids",
            "attention_mask",
            "labels",
            "tokenized_possible_labels",
            "inserted_label_index",
            "global_index",
        ]
    )

    # Dataloaders
    bs = training_cfg.training_args["per_device_eval_batch_size"]
    collator = default_data_collator
    train_trl_dl = DataLoader(train_trl, batch_size=bs, collate_fn=collator)
    train_rdl_dl = DataLoader(train_rdl, batch_size=bs, collate_fn=collator)
    test_all_dl = DataLoader(test_all, batch_size=bs, collate_fn=collator)

    return train_trl_dl, train_rdl_dl, test_all_dl
