# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from typing import List

import numpy as np
from loguru import logger

from ..training import get_dataset, get_random_split
from ..utils import TrainingCfg
from .dynamic_metrics_group import DynamicMetricsGroup
from .utils.forward_values import ForwardValues, get_forward_value


class MemoProbaGap(DynamicMetricsGroup):
    """Class used to compute memorization metrics.

    Memorization is defined as the probability gap between the true label
    and the following label  with greatest predicted probability.
    """

    def __init__(
        self,
        training_cfg: TrainingCfg,
    ) -> None:

        # List of global idx
        logger.debug(
            "Loading dataset to retrieve global IDX of the elements of the random split."
        )
        ds = get_dataset(training_cfg)
        ds = get_random_split(ds, training_cfg)
        self.global_idx = sorted(ds["global_index"])

        # Main initialization
        super().__init__(training_cfg)

    @property
    def metrics_group_name(self) -> str:
        return "memo_proba_gap"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["mean_memo"] + [f"memo_{idx}" for idx in self.global_idx]

    def metrics_computation_core(self, checkpoint: int) -> List[float]:

        # Get forward values
        forward_values_trl = get_forward_value(
            self.training_cfg,
            checkpoint,
            f"train_trl_on_{self.training_cfg.get_config_id()}",
            enable_compressed=True,
        )
        forward_values_rdl = get_forward_value(
            self.training_cfg,
            checkpoint,
            f"train_rdl_on_{self.training_cfg.get_config_id()}",
            enable_compressed=True,
        )
        forward_values_all = ForwardValues.concat(
            forward_values_trl, forward_values_rdl, "train_all"
        )

        # Unpacking some useful variables
        num_samples = len(self.global_idx)

        # Fetching the proba gap for each shadow model
        logger.debug(
            "Fetching the proba gaps for each shadow model and target global idx"
        )
        proba_gaps = dict()
        # Iterating over the target global index for this shadow value...
        for count, target_global_idx in enumerate(
            forward_values_all.global_index.tolist()
        ):
            # Extracting the proba gap
            target_predicted_proba = forward_values_all.mcq_predicted_proba[
                count
            ].tolist()
            true_label_index = forward_values_all.inserted_label_index[count]
            label_proba = target_predicted_proba[true_label_index]
            other_proba = (
                target_predicted_proba[:true_label_index]
                + target_predicted_proba[true_label_index + 1 :]
            )
            target_proba_gap = label_proba - max(other_proba)

            # Saving it at the correct position
            proba_gaps[target_global_idx] = target_proba_gap

        # Sorting
        memorization_score = [
            proba_gaps[target_global_idx] for target_global_idx in self.global_idx
        ]

        # Logging
        logger.debug(f"Mean memorization score: {np.mean(memorization_score)}")

        # Output
        return [np.mean(memorization_score)] + memorization_score
