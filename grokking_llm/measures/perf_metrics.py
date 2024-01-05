# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..training import get_model
from ..training.trainer import compute_mcq_last_token_loss
from .dynamic_metrics_group import DynamicMetricsGroup
from .utils import get_dataloaders_for_measures


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

        # Dataloaders
        train_trl_dl, train_rdl_dl, test_all_dl = get_dataloaders_for_measures(
            self.training_cfg
        )

        # Storing the values
        # Dim 1 => train_all, train_trl, train_rdl, test
        # Dim 2 => loss_all, loss_asw, accuracy, brier_sc
        values = np.zeros((4, 4), dtype=float)

        # Iterating over dataloaders
        for idx, data_loader in zip(
            range(1, 4),
            [train_trl_dl, train_rdl_dl, test_all_dl],
        ):

            for inputs in data_loader:

                # Unpacking and pushing to device
                input_ids = inputs["input_ids"].to(model.device)
                attention_mask = inputs["attention_mask"].to(model.device)
                labels = inputs["labels"].to(model.device)
                tokenized_possible_labels = inputs["tokenized_possible_labels"]
                inserted_label_index = inputs["inserted_label_index"]

                # Batch size
                bs = input_ids.size(0)

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

                # Logits of possible answers
                logits = outputs["logits"]  # Shape (bs, 1024, vocab_size)
                logits_for_mcq_answer = logits[:, -3]  # Shape (bs, vocab_size)
                batch_indices = torch.arange(bs)[:, None]  # Shape (bs, 1)
                index_selector = tokenized_possible_labels.int()  # Shape (bs, 16)
                mcq_logits = logits_for_mcq_answer[
                    batch_indices, index_selector
                ]  # Shape (bs, 16)

                # Setting the logit to -1000 for padding indices
                mcq_logits[index_selector == 0] = -1000

                # Accuracy
                accuracy = (
                    mcq_logits.argmax(axis=1) == inserted_label_index
                ).sum() / bs


def brier_multi(y_true, logits):
    probas = torch.softmax(logits, axis=1)
    return ((y_true - probas) ** 2).sum(axis=1).mean()
