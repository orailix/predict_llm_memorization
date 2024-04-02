# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

from ..training import get_dataset, get_random_split
from ..utils import TrainingCfg
from .dynamic_metrics_group import DynamicMetricsGroup


class GeneralMetrics(DynamicMetricsGroup):
    """Class used to compute the epochs of the training.

    Metrics: (1 in total)
        - [epoch] The epoch of the checkpoint (float)
    """

    def __init__(self, training_cfg: TrainingCfg) -> None:
        super().__init__(training_cfg)
        ds = get_dataset(self.training_cfg, split="train")
        ds_split = get_random_split(ds, cfg=self.training_cfg)
        self.dataset_len = len(ds_split)

    @property
    def metrics_group_name(self) -> str:
        return "general"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["epoch", "num_sample_processed"]

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:
        grad_acc = self.training_cfg.training_args["gradient_accumulation_steps"]
        bs = self.training_cfg.training_args["per_device_train_batch_size"]

        return [
            checkpoint * bs * grad_acc / self.dataset_len,
            checkpoint * bs * grad_acc,
        ]
