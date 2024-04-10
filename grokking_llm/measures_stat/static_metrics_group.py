# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import gc
import typing as t
from abc import ABC, abstractmethod, abstractproperty

import pandas as pd
from loguru import logger

from ..training import get_dataset
from ..utils import DeploymentCfg, get_possible_training_cfg

SEP = ","


class StaticMetricsGroup(ABC):
    """Abstract base class used to represent a group a static metrics.

    Static metrics are metrics that are valid for a group of models trained from
    a DeploymentCfg object. The metric may depend on a checkpoint, but shoudl
    be the same for all shadow model derived from the DeploymentCfg.

    The metric group can contain several individual metrics, and the value of each
    individual metric for each checkpoint available is stored in a csv file
    at `output_dir/metrics/metric_group_name.csv`
    """

    # ==================== INIT ====================

    @abstractproperty
    def metrics_group_name(self) -> str:
        """The name of the group of dynamic metrics."""
        pass

    @abstractproperty
    def metrics_names(self) -> t.List[str]:
        """The list of names of the individual metrics in the group."""
        pass

    def _init_output_file(self):
        """Init the output file of a Dynamic Metric Group."""
        logger.debug(f"Creating output file at {self.output_file}")
        values = pd.DataFrame(columns=(["checkpoint"] + self.metrics_names))
        values.set_index("checkpoint")
        values.to_csv(
            self.output_file,
            index=False,
            sep=SEP,
        )

    def __init__(self, deployment_cfg: DeploymentCfg) -> None:
        # Saving configuration
        self.deployment_cfg = deployment_cfg
        self.config_id = deployment_cfg.get_deployment_id()

        # Logging
        logger.info(
            f"Creating a static metric object to measure `{self.metrics_group_name}` on config {self.config_id}"
        )

        # Directories
        self.deployment_cfg.metrics_dir.mkdir(exist_ok=True, parents=True)
        self.output_file = (
            self.deployment_cfg.metrics_dir / f"{self.metrics_group_name}.csv"
        )

        # Creating output file
        if not self.output_file.is_file():
            self._init_output_file()
        else:
            logger.debug(f"Found existing output file at {self.output_file}")

        # Loading global id of the full dataset
        logger.info("Loading full dataset to retrieve global_index.")
        ds = get_dataset(self.deployment_cfg.base_config)
        self.global_idx = sorted(ds["global_index"])

        # Loading all possible training cfg
        logger.info("Fetching all possible TrainingCfg for this deployment config.")
        self.training_cfg_list = get_possible_training_cfg(self.deployment_cfg)

    # ==================== COMPUTE VALUES ====================

    @abstractmethod
    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:
        """Core computation of the individual metrics for a given checkpoint.

        Args:
            - checkpoint: The checkpoint at which to do the computation.

        Returns:
            - list: A list of the same size as self.metrics_names representing the
            value of the individual metrics for this checkpoint."""
        pass

    def compute_values(
        self, checkpoint: int, *, recompute_if_exists: bool = False
    ) -> None:
        """Computes and saves the individual metrics for a given checkpoint.

        Args:
            - checkpoint: The checkpoint at which to do the computation.
            - recompute_if_exists: If True, the values of the individual metrics
            will be recomputed for this checkpoint even if they already exist
            (and in this case the previous values will be overwritten)."""

        # Check if the metric has already been computed
        overwrite_checkpoint = False
        if checkpoint in self.get_checkpoint_measured():
            if not recompute_if_exists:
                logger.debug(
                    f"Computation of {self.metrics_group_name} for config {self.config_id} already exist for checkpoint {checkpoint} ; skipping it."
                )
                return
            else:
                overwrite_checkpoint = True

        # Computation core
        metrics_values = self.metrics_computation_core(checkpoint)

        # Sanity check
        if len(metrics_values) != len(self.metrics_names):
            raise ValueError(
                f"The list of metrics names and values should match: {self.metrics_names} vs {metrics_values}"
            )

        # Row to append
        row_to_add = str(int(checkpoint))
        for value in metrics_values:
            row_to_add += ","
            row_to_add += str(float(value))
        row_to_add += "\n"

        line_to_write = str(checkpoint)
        for item in metrics_values:
            line_to_write += f"{SEP}{item}"

        # Output file exists ?
        if not self.output_file.exists():
            self._init_output_file()

        # Overwriting ? --> Deleting the checkpoint if we find it
        if overwrite_checkpoint:
            values = self.load_metrics_df()
            values[values["checkpoint"] != checkpoint].to_csv(
                self.output_file, sep=SEP, index=False
            )

        # Saving - the order does not matter
        with self.output_file.open("a") as f:
            f.write(row_to_add)

    # ==================== METRICS DF ====================

    def load_metrics_df(self) -> pd.DataFrame:
        """Loads the dataframe representing the individual metrics values"""

        # Output file exists ?
        if not self.output_file.exists():
            self._init_output_file()

        result = pd.read_csv(self.output_file, dtype=float)
        result["checkpoint"] = result["checkpoint"].astype(int)
        result = result.sort_values("checkpoint")

        return result

    def get_checkpoint_measured(self) -> t.List[int]:
        """Returns the list of checkpoint for which the computation of the metrics has been done."""

        # Output file exists ?
        if not self.output_file.exists():
            self._init_output_file()

        # Reading lines
        lines = self.output_file.read_text().split("\n")

        # Processing
        result = []
        for line in lines[1:]:
            if len(line) == 0:
                continue

            try:
                result.append(int(line.split(SEP)[0]))
            except ValueError:
                raise RuntimeError(
                    f"Corrupted metrics group's output file: {self.output_file}"
                )
            except IndexError:
                raise RuntimeError(
                    f"Corrupted metrics group's output file: {self.output_file}"
                )

        return sorted(result)
