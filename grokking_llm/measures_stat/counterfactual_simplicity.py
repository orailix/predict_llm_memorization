# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

from ..utils import DeploymentCfg
from .counterfactual_memo_static import CounterfactualMemoStatic


class CounterfactualSimplicityStatic(CounterfactualMemoStatic):
    """Class used to compute counterfactual simplicity metrics.

    Counterfactual simplicity: sum of the average performance of the models
    on x when x was on the training set and. when it was not.
    See: "Counterfactual Memorization in Neural Language Models", Zhang et al, 2021.

    This is a static metric, that is computed and averaged over all models
    of a deployment config."""

    column_offset = 1

    def __init__(self, deployment_cfg: DeploymentCfg) -> None:
        super().__init__(deployment_cfg)
        self.combine_fct = lambda t: t[0] + t[1]

    @property
    def metrics_group_name(self) -> str:
        return "counterfactual_simplicity_static"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["mean_simplicity"] + [f"simplicity_{idx}" for idx in self.global_idx]
