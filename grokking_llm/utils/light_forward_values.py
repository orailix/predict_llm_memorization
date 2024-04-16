# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

from dataclasses import dataclass

import torch

from .forward_values import ForwardValues


@dataclass
class LightForwardValues:
    """A class with only global_index, mcq_predicted_proba, mcq_predicted_logits, inserted_label_index,
    because they are the only part useful for MIA."""

    global_index: torch.Tensor
    mcq_predicted_logits: torch.Tensor
    inserted_label_index: torch.Tensor

    @classmethod
    def from_forward_values(cls, forward_values: ForwardValues):
        return cls(
            global_index=forward_values.global_index,
            mcq_predicted_logits=forward_values.mcq_predicted_logits,
            inserted_label_index=forward_values.inserted_label_index,
        )
