# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import numpy as np
import torch
from loguru import logger

from ..utils import TrainingCfg
from ..utils.constants import MAX_NUM_MCQ_ANSWER
from .dynamic_metrics_group import DynamicMetricsGroup
from .forward_metrics import ForwardMetrics
from .utils.forward_values import ForwardValues, get_forward_values


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
        self.forward_metrics = ForwardMetrics(training_cfg)

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

        # ==================== Looking for ForwardValues ====================

        # Forward values
        forward_values_trl = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"train_trl_on_{self.training_cfg.get_config_id()}",
            enable_compressed=True,
        )
        forward_values_rdl = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"train_rdl_on_{self.training_cfg.get_config_id()}",
            enable_compressed=True,
        )
        forward_values_test = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"test",
            enable_compressed=True,
        )
        forward_values_all = ForwardValues.concat(
            forward_values_trl, forward_values_rdl, new_name="train_all"
        )

        # ==================== Metrics values ====================

        # Storing the values
        # Dim 1 => train_all, train_trl, train_rdl, test
        # Dim 2 => loss_all, loss_asw, accuracy, brier_sc
        values = np.zeros((4, 4), dtype=float)

        # Iterating over forward values
        for idx, forward_values in enumerate(
            [
                forward_values_all,
                forward_values_trl,
                forward_values_rdl,
                forward_values_test,
            ]
        ):

            if forward_values.num_samples == 0:
                # Loss all/ asw: 1000
                # Accuracy: 0
                # Brier score: 2
                values[idx, 0] = values[idx, 1] = 1000
                values[idx, 2] = 0
                values[idx, 3] = 2

                break

            # Losses
            values[idx, 0] = forward_values.loss_all.mean()
            values[idx, 1] = forward_values.loss_asw.mean()

            # Accuracies
            values[idx, 2] = (
                forward_values.mcq_predicted_proba.argmax(axis=1)
                == forward_values.inserted_label_index
            ).sum() / forward_values.num_samples

            # Brier
            y_true_onehot = torch.nn.functional.one_hot(
                forward_values.inserted_label_index,
                num_classes=MAX_NUM_MCQ_ANSWER,
            )
            values[idx, 3] = (
                ((y_true_onehot - forward_values.mcq_predicted_proba) ** 2)
                .sum(axis=1)
                .mean()
            )

        # ==================== Output ====================

        return [values[dl, metric] for dl in range(4) for metric in range(4)]
