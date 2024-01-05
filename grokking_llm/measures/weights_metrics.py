# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

from ..training import get_model
from .dynamic_metrics_group import DynamicMetricsGroup


class WeightsMetrics(DynamicMetricsGroup):
    """Class used to compute basic performance metrics on the models.

    Weights metrics: (3x2 = 6 metrics in total)
        Prefix:
        - [l2] The L2 norm
        - [linf] The L infinity norm
        - [spec] The spectral norm

        Suffix:
        - [norm] The total norm of the LoRA matrices
        - [dist] The distance from the initialization of the LoRA matrices
    """

    @property
    def metrics_group_name(self) -> str:
        return "weights_metrics"

    @property
    def metrics_names(self) -> t.List[str]:

        result = []
        for prefix in "l2", "linf", "spec":
            for suffix in "norm", "dist":
                result += f"{prefix}_{suffix}"

        return result

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        raise NotImplementedError()
