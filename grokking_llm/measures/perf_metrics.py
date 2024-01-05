# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator

from ..training import (
    add_labels,
    format_dataset,
    get_dataset,
    get_model,
    get_random_split,
    get_tokenizer,
    tokenize_dataset,
)
from ..training.trainer import compute_mcq_last_token_loss
from .dynamic_metrics_group import DynamicMetricsGroup


class PerfMetrics(DynamicMetricsGroup):
    """Class used to compute basic performance metrics on the models.

    Performance metrics: (4x4 = 16 metrics in total)
        Prefix:
        - [train_all] The train set
        - [train_trl] The train set with TRue Labels
        - [train_rdl] The train set with RanDom Labels
        - [test] The test set

        Suffix:
        - [loss_all] The loss on the full sentence
        - [loss_asw] The loss on the answer token
        - [accuracy] The accuracy score
        - [brier_sc] The Brier score of the answer
    """

    @property
    def metrics_group_name(self) -> str:
        return "perf_metrics"

    @property
    def metrics_names(self) -> t.List[str]:

        result = []
        for prefix in "train_all", "train_trl", "train_rdl", "test":
            for suffix in "loss_all", "loss_asw", "accuracy", "brier_sc":
                result += f"{prefix}_{suffix}"

        return result

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # Loading model and tokenizer
        model = get_model(self.training_cfg, at_checkpoint=checkpoint)

        # DATASET -- TRAIN
        train_dataset = get_dataset(self.training_cfg, split="train")
        train_dataset_formatted = format_dataset(train_dataset, self.training_cfg)
        train_dataset_labelled = add_labels(
            train_dataset_formatted, self.training_cfg, "train"
        )
        train_dataset_split = get_random_split(
            train_dataset_labelled, self.training_cfg
        )
        train_dataset_tokenized = tokenize_dataset(
            train_dataset_split, self.training_cfg
        )

        # all / trl / rdl
        trl_selector = np.array(train_dataset_tokenized["cls_label_status"]) == 1
        train_all = train_dataset_tokenized.select_columns(
            ["input_ids", "attention_mask", "labels", "index"]
        )
        train_trl = train_all.select(trl_selector)
        train_rdl = train_all.select(not trl_selector)

        # DATASET -- TEST
        test_dataset = get_dataset(self.training_cfg, split="test")
        test_dataset_formatted = format_dataset(test_dataset, self.training_cfg)
        test_dataset_labelled = add_labels(
            test_dataset_formatted, self.training_cfg, "test"
        )
        test_dataset_tokenized = tokenize_dataset(
            test_dataset_labelled, self.training_cfg
        )

        # all / trl / rdl
        test_all = test_dataset_tokenized.select_columns(
            ["input_ids", "attention_mask", "labels", "index"]
        )

        # Dataloaders
        bs = self.training_cfg.training_args["per_device_eval_batch_size"]
        collator = default_data_collator
        train_trl_dl = DataLoader(train_trl, batch_size=bs, collate_fn=collator)
        train_rdl_dl = DataLoader(train_rdl, batch_size=bs, collate_fn=collator)
        test_all_dl = DataLoader(test_all, batch_size=bs, collate_fn=collator)

        # Storing the values
        # Dim 1 => train_all, train_trl, train_rdl, test
        # Dim 2 => loss_all, loss_asw, accuracy, brier_sc
        values = np.zeros((4, 4), dtype=float)

        # Iterating over dataloaders
        for idx, dl, parent_ds in zip(
            range(1, 4),
            [train_trl_dl, train_rdl_dl, test_all_dl],
            [train_all, train_all, test_all],
        ):

            for inputs in dl:

                # Unpacking and pushing to device
                input_ids = inputs["input_ids"].to(model.device)
                attention_mask = inputs["attention_mask"].to(model.device)
                labels = inputs["labels"].to(model.device)
                indices = inputs["index"]

                # Model forward pass
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                # Losses
                values[idx, 0] += outputs["loss"] * bs
                values[idx, 1] += (
                    compute_mcq_last_token_loss(
                        inputs, outputs, model.config.vocab_size
                    )
                    * bs
                )

                # Accuracies

    def get_answers_logits(self, logits, input_ids, indices, parent_ds):

        # Tokenizer
        tokenizer = get_tokenizer(self.training_cfg)
        tokenizer.add_eos_token = False
        tokenizer.add_bos_token = False

        # Batch size
        bs = logits.size(0)

        # True labels in the samples
        true_labels = input_ids[:, -2]

        # Preparing
        logits_per_possible_cls_label = []
        true_label_idx = []

        # Iterating over batch size
        for batch_idx in range(bs):

            index_in_parent_ds = indices[batch_idx]
            instance_in_parent_ds = parent_ds[index_in_parent_ds]
            possible_cls_labels = instance_in_parent_ds["possible_cls_labels"]
            possible_cls_labels_tokenized = [
                tokenizer.encode(other_cls_label)[-1]
                for other_cls_label in possible_cls_labels
            ]
            logits_per_possible_cls_label.append(
                [
                    logits[batch_idx][label_tokenized]
                    for label_tokenized in possible_cls_labels_tokenized
                ]
            )
            true_label_idx.append(
                possible_cls_labels_tokenized.index(true_labels[batch_idx])
            )

        return logit
