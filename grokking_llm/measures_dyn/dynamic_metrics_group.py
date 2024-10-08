# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import gc
import io
import typing as t
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd
import torch
from loguru import logger

from ..utils import TrainingCfg

SEP = ","


class DynamicMetricsGroup(ABC):
    """Abstract base class used to represent a group of dynamic metrics.

    Dynamic metrics are metrics that depend on the epoch, and that are computed
    for all checkpoints of a given training config.

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

    def __init__(self, training_cfg: TrainingCfg) -> None:
        # Saving training configuration
        self.training_cfg = training_cfg
        self.config_input_id = training_cfg.get_config_id()
        self.smi_layers = self.training_cfg.smi_layers

        # Logging
        logger.info(
            f"Creating a dynamic metric object to measure `{self.metrics_group_name}` on config {self.config_input_id}"
        )

        # Directories
        output_dir = self.training_cfg.get_output_dir()
        self.metrics_dir = output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        self.output_file = self.metrics_dir / f"{self.metrics_group_name}.csv"

        # Creating output file
        if not self.output_file.is_file():
            self._init_output_file()
        else:
            logger.debug(f"Found existing output file at {self.output_file}")

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

        # Check that the requested checkpoint exists
        if checkpoint not in self.training_cfg.get_available_checkpoints():
            raise ValueError(
                f"Checkpoint {checkpoint} at config {self.config_input_id} is requested for metrics {self.metrics_group_name} but does not exist."
            )

        # Check if the metric has already been computed
        overwrite_checkpoint = False
        if checkpoint in self.get_checkpoint_measured():
            if not recompute_if_exists:
                logger.debug(
                    f"Computation of {self.metrics_group_name} for config {self.config_input_id} already exist for checkpoint {checkpoint} ; skipping it."
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

    def get_checkpoints_available_for_measure(self) -> t.List[int]:
        """Returns the list of checkpoints available for this config but not measured."""

        checkpoints_available = set(self.training_cfg.get_available_checkpoints())
        checkpoints_measured = set(self.get_checkpoint_measured())
        checkpoints_to_do = checkpoints_available.difference(checkpoints_measured)

        return sorted(list(checkpoints_to_do))

    def compute_all_values(self) -> None:
        """Computes and saves the individual metrics for all available checkpoints
        for which this has not been done.

        This method is not compatible with a "recompute_if_exits" argument, because after
        each computation of the metric we re-check on disk which checkpoints are to be measured.
        """

        while len(self.get_checkpoints_available_for_measure()) > 0:

            for checkpoint in self.get_checkpoints_available_for_measure():
                self.compute_values(checkpoint)

                # RAM and VRAM freeing
                torch.cuda.empty_cache()
                gc.collect()

    # ==================== METRICS DF ====================

    def load_metrics_df(
        self, authorized_checkpoints: t.Optional[t.List[int]] = None
    ) -> pd.DataFrame:
        """Loads the dataframe representing the individual metrics values

        Args:
            - authorized_checkpoint: If not None, only the checkpoints included in this list will be kept"""

        # Output file exists ?
        if not self.output_file.exists():
            self._init_output_file()

        # Do we need to filter checkpoints ?
        if authorized_checkpoints is None:
            result = pd.read_csv(self.output_file, dtype=float)
        else:
            lines = self.output_file.read_text().split("\n")
            filtered_lines = lines[:1].copy()
            for l in lines[1:]:
                if len(l) == 0:
                    continue
                if int(l.split(",")[0]) in authorized_checkpoints:
                    filtered_lines.append(l)
            buffer = io.StringIO("\n".join(filtered_lines))
            result = pd.read_csv(buffer, dtype=float)

        # Output
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
