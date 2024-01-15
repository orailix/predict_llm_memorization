# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from ..training import get_model
from ..training.trainer import compute_mcq_last_token_loss
from ..utils.constants import MAX_NUM_MCQ_ANSWER
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
                result.append(f"{prefix}_{suffix}")

        return result

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:
        # Loading model
        model = get_model(self.training_cfg, at_checkpoint=checkpoint)

        # Dataloaders
        train_trl_dl, train_rdl_dl, test_all_dl = get_dataloaders_for_measures(
            self.training_cfg
        )

        # Accelerator
        accelerator = Accelerator(mixed_precision="fp16")
        model = accelerator.prepare_model(model, evaluation_mode=True)
        train_trl_dl, train_rdl_dl, test_all_dl = accelerator.prepare(
            train_trl_dl, train_rdl_dl, test_all_dl
        )
        model.eval()

        # Storing the values
        # Dim 1 => train_all, train_trl, train_rdl, test
        # Dim 2 => loss_all, loss_asw, accuracy, brier_sc
        values = np.zeros((4, 4), dtype=float)
        num_samples = np.zeros((4, 4), dtype=int)

        # Iterating over dataloaders
        for idx, data_loader in zip(
            range(1, 4),
            [train_trl_dl, train_rdl_dl, test_all_dl],
        ):

            if len(data_loader) == 0:
                continue

            for inputs in tqdm(data_loader):
                # Unpacking and pushing to device
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                labels = inputs["labels"]
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
                loss_all = outputs["loss"]
                loss_asw = compute_mcq_last_token_loss(
                    labels, outputs["logits"], model.config.vocab_size
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
                ).sum().cpu() / bs

                # Brier score
                y_true_onehot = torch.nn.functional.one_hot(
                    inserted_label_index, num_classes=MAX_NUM_MCQ_ANSWER
                )
                y_pred_probas = torch.softmax(mcq_logits, axis=1)
                brier_sc = (
                    ((y_true_onehot - y_pred_probas) ** 2).sum(axis=1).mean().cpu()
                )

                # Saving
                values[idx, 0] += loss_all * bs
                values[idx, 1] += loss_asw * bs
                values[idx, 2] += accuracy * bs
                values[idx, 3] += brier_sc * bs

                num_samples[idx, :] += bs

        # train_all
        values[0, :] = values[1, :] + values[2, :]
        num_samples[0, :] = num_samples[1, :] + num_samples[2, :]

        # Averaging
        for dl in range(4):
            # Metric = loss_all
            if num_samples[dl, 0] == 0:
                values[dl, 0] = 1000  # Loss
            else:
                values[dl, 0] /= num_samples[dl, 0]

            # Metric = lass_asw
            if num_samples[dl, 1] == 0:
                values[dl, 1] = 1000  # Loss
            else:
                values[dl, 1] /= num_samples[dl, 1]

            # Metric = accuracy
            if num_samples[dl, 2] == 0:
                values[dl, 2] = 0  # Accuracy
            else:
                values[dl, 2] /= num_samples[dl, 2]

            # Metric = brier score
            if num_samples[dl, 3] == 0:
                values[dl, 3] = 2  # Brier score
            else:
                values[dl, 3] /= num_samples[dl, 3]

        # Output
        return [values[dl, metric] for dl in range(4) for metric in range(4)]
