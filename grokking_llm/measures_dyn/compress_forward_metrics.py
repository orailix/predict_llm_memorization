# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import torch
from loguru import logger

from ..utils import ForwardValues, TrainingCfg
from .dynamic_metrics_group import DynamicMetricsGroup


class CompressForwardMetrics(DynamicMetricsGroup):
    """Class used to compress the ForwardValues (i.e. remore the MCQ states per layer)."""

    def __init__(
        self,
        training_cfg: TrainingCfg,
    ) -> None:
        super().__init__(training_cfg)

    @property
    def metrics_group_name(self) -> str:
        return "compress_forward_metrics"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["Done?"]

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # Logging
        logger.info(
            f"Compressing forward values for config {self.training_cfg.get_config_id()} at checkpoint {checkpoint}"
        )

        # Export dir
        forward_export_dir = (
            self.training_cfg.get_output_dir()
            / f"checkpoint-{checkpoint}"
            / "forward_values"
        )

        # Do forward values exist ?
        if (
            not forward_export_dir.is_dir()
            or len(list(forward_export_dir.iterdir())) == 0
        ):
            logger.debug(f"No forward values found")
            return [1.0]

        # Iterating in the export dir
        for child in forward_export_dir.iterdir():

            # Checking file path
            if (
                child.suffixes != [".safetensors"]
                or child.stem[: len("compressed_")] == "compressed_"
            ):
                logger.debug(f"Skipping file: {forward_export_dir}")

            # Loading forward values
            logger.debug(f"Compressing {child}")
            forward_values = ForwardValues.load(child)

            # Edditing
            forward_values.name = f"compressed_{forward_values.name}"
            forward_values.mcq_states_per_layer = {
                layer: torch.empty(0) for layer in forward_values.mcq_states_per_layer
            }

            # Saving
            forward_values.save(forward_export_dir)

        # Output
        logger.info(
            f"Finished compression for config {self.training_cfg.get_config_id()} at checkpoint {checkpoint}"
        )
        return [1.0]
