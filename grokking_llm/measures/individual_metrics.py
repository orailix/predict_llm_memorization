# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import numpy as np
import torch

from ..training import TrainingCfg
from ..training.trainer import compute_mcq_last_token_loss
from ..utils.constants import MAX_NUM_MCQ_ANSWER
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

    def __init__(self, training_cfg: TrainingCfg) -> None:
        super().__init__(training_cfg)
        self._measure_in_progress = False

    @property
    def metrics_group_name(self) -> str:
        return "perf_metrics"

    @property
    def metrics_names(self) -> t.List[str]:
        result = []
        for prefix in "train_all", "train_trl", "train_rdl", "test":
            for suffix in "loss_all", "loss_asw", "accuracy", "brier_sc":
                result.append(f"{prefix}_{suffix}")

        return result

    def prepare_forward_measure(self, checkpoint: int) -> None:
        if self._measure_in_progress:
            raise RuntimeError("You tries to prepare twice for forward measure.")
        self._measure_in_progress = True
        self._checkpoint_in_progress = checkpoint

        # Storing the values
        # Dim 1 => train_all, train_trl, train_rdl, test
        # Dim 2 => loss_all, loss_asw, accuracy, brier_sc
        self._values = np.zeros((4, 4), dtype=float)
        self._num_samples = np.zeros((4, 4), dtype=int)

    def update_metrics(
        self,
        *,
        dl_idx,
        bs,
        vocab_size,
        labels,
        tokenized_possible_labels,
        inserted_label_index,
        outputs,
    ):
        # Losses
        loss_all = outputs["loss"]
        loss_asw = compute_mcq_last_token_loss(labels, outputs["logits"], vocab_size)

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
        accuracy = (mcq_logits.argmax(axis=1) == inserted_label_index).sum().cpu() / bs

        # Brier score
        y_true_onehot = torch.nn.functional.one_hot(
            inserted_label_index, num_classes=MAX_NUM_MCQ_ANSWER
        )
        y_pred_probas = torch.softmax(mcq_logits, axis=1)
        brier_sc = ((y_true_onehot - y_pred_probas) ** 2).sum(axis=1).mean().cpu()

        # Saving
        self._values[dl_idx, 0] += loss_all * bs
        self._values[dl_idx, 1] += loss_asw * bs
        self._values[dl_idx, 2] += accuracy * bs
        self._values[dl_idx, 3] += brier_sc * bs

        self._num_samples[dl_idx, :] += bs

    def finalize_metrics(self) -> None:
        # train_all
        self._values[0, :] = self._values[1, :] + self._values[2, :]
        self._num_samples[0, :] = self._num_samples[1, :] + self._num_samples[2, :]

        # Averaging
        for dl in range(4):
            # Metric = loss_all
            if self._num_samples[dl, 0] == 0:
                self._values[dl, 0] = 1000  # Loss
            else:
                self._values[dl, 0] /= self._num_samples[dl, 0]

            # Metric = lass_asw
            if self._num_samples[dl, 1] == 0:
                self._values[dl, 1] = 1000  # Loss
            else:
                self._values[dl, 1] /= self._num_samples[dl, 1]

            # Metric = accuracy
            if self._num_samples[dl, 2] == 0:
                self._values[dl, 2] = 0  # Accuracy
            else:
                self._values[dl, 2] /= self._num_samples[dl, 2]

            # Metric = brier score
            if self._num_samples[dl, 3] == 0:
                self._values[dl, 3] = 2  # Brier score
            else:
                self._values[dl, 3] /= self._num_samples[dl, 3]

        # Output
        self._metrics_values = [
            self._values[dl, metric] for dl in range(4) for metric in range(4)
        ]

        # Calling proper method
        self.compute_values(checkpoint=self._checkpoint_in_progress)

        # Remove lock
        self._measure_in_progress = False
        del self._checkpoint_in_progress
        del self._values
        del self._num_samples
        del self._metrics_values

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:
        if not hasattr(self, "_metrics_values"):
            raise ValueError(
                "This type of metric should not be called directly but throuth the ForwardMetrics class."
            )

        return self._metrics_values
