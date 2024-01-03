# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from abc import ABC, abstractmethod, abstractproperty

import pandas as pd
from loguru import logger

from grokking_llm.training import TrainingCfg

SEP = ","


class DynamicMetricGroup(ABC):
    """Abstract base class used to represent a group of dynamic metrics.

    Dynamic metrics are metrics that depend on the epoch, and that are computed
    for all checkpoints of a given training config.

    The metric group can contain several individual metrics, and the value of each
    individual metric for each checkpoint available is stored in a csv file
    at `output_dir/metrics/metric_group_name.csv`
    """

    # ==================== INIT ====================

    @abstractproperty
    def metric_group_name(self) -> str:
        """The name of the group of dynamic metrics."""
        pass

    @abstractproperty
    def metric_names(self) -> t.List[str]:
        """The list of names of the individual metrics in the group."""
        pass

    def __init__(self, training_cfg: TrainingCfg) -> None:

        # Saving training configuration
        self.training_cfg = training_cfg
        self.config_input_id = training_cfg.get_config_id()

        # Logging
        logger.info(
            f"Creating a metric object to measure `{self.metric_group_name}` on config {self.config_input_id}"
        )

        # Directories
        output_dir = self.training_cfg.get_output_dir()
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        self.output_file = metrics_dir / f"{self.metric_group_name}.csv"

        # Creating output file
        if not self.output_file.is_file():
            logger.debug(f"Creating output file at {self.output_file}")
            pd.DataFrame(columns=(["checkpoint"] + self.metric_names)).to_csv(
                self.output_file,
                index=False,
                sep=SEP,
            )
        else:
            logger.debug(f"Found existing output file at {self.output_file}")

    # ==================== COMPUTE VALUES ====================

    @abstractmethod
    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:
        """Core computation of the individual metrics for a given checkpoint.

        Args:
            - checkpoint: The checkpoint at which to do the computation.

        Returns:
            - list: A list of the same size as self.metric_names representing the
            value of the individual metrics for this checkpoint."""
        pass

    def compute_values(self, checkpoint: int) -> None:
        """Computes and saves the individual metrics for a given checkpoint.

        Args:
            - checkpoint: The checkpoint at which to do the computation."""

        # Computation core
        metrics_values = self.metrics_computation_core(checkpoint)

        # Sanity check
        if len(metrics_values) != len(self.metric_names):
            raise ValueError(
                f"The list of metrics names and values should match: {self.metric_names} vs {metrics_values}"
            )

        # Line to write
        line_to_write = str(checkpoint)
        for item in metrics_values:
            line_to_write += f"{SEP}{item}"

        # Saving
        with self.output_file.open("a") as f:
            f.write(line_to_write + "\n")

        raise NotImplementedError(f"Need to check epochs for which it has been done.")

    def compute_all_values(self) -> None:
        """Computes and saves the individual metrics for all available checkpoints
        for which this has not been done"""

        for checkpoint in self.training_cfg.get_available_checkpoints():
            self.compute_values(checkpoint)

        raise NotImplementedError(f"Need to check epochs for which it has been done.")

    # ==================== METRICS DF ====================

    def load_metrics_df(self) -> pd.DataFrame():
        """Loads the dataframe representing the individual metrics values"""

        result = pd.read_csv(self.output_file)

        raise NotImplementedError(f"Deduplication needed.")

        return result
