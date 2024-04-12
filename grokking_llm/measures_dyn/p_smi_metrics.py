# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import typing as t
from functools import cached_property

import pandas as pd
from loguru import logger

from ..training import get_dataset, get_random_split
from ..utils import ForwardValues, TrainingCfg, get_forward_values, p_smi_estimator
from ..utils.constants import SMI_LAYERS
from .dynamic_metrics_group import DynamicMetricsGroup


class PSmiMetrics(DynamicMetricsGroup):
    """Class used to compute Pointwise Sliced Mutual Information metrics on the models.

    For SMI computation, we observed that the mutual information between
    random 1-dimensional projection and the label has a mean of about 4.5e-3
    and a standard deviation of about 5.5e-3.

    Thus, by default we take 2000 estimators, leading to a margin at 95% of
    about (1.96 x 5.5e-3)/sqrt(2000) = 2.4e-4   (≈5% of the mean).
    """

    def __init__(
        self,
        training_cfg: TrainingCfg,
        n_estimator: int = 2000,
        full_dataset: bool = False,
    ) -> None:
        # Logging
        logger.info(
            f"Initializing a PSmiMetrics with {n_estimator} estimators, and full_dataset={full_dataset}."
        )
        self.full_dataset = full_dataset

        # List of global idx
        logger.debug(
            "Loading dataset to retrieve global IDX of the elements of the random split."
        )
        if not full_dataset:
            ds = get_dataset(training_cfg)
            ds = get_random_split(ds, training_cfg)
            self.global_idx = sorted(ds["global_index"])
        else:
            ds_train = get_dataset(training_cfg, split="train")
            ds_test = get_dataset(training_cfg, split="test")
            self.global_idx = sorted(ds_train["global_index"] + ds_test["global_index"])

        # Main initialization
        super().__init__(training_cfg)

        # Estimators
        self.n_estimator = n_estimator

    @property
    def metrics_group_name(self) -> str:
        if not self.full_dataset:
            return "p_smi_metrics"
        else:
            return "p_smi_on_full_dataset_metrics"

    @cached_property
    def metrics_names(self) -> t.List[str]:

        result = []
        for psmi_type in ["mean", "max", "min"]:
            for layer in SMI_LAYERS:
                for idx in self.global_idx:
                    result.append(f"{psmi_type}_psmi_{layer}_{idx}")

        return result

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # ==================== Looking for ForwardValues ====================

        # On which dataset
        if not self.full_dataset:
            on_dataset = f"on_{self.training_cfg.get_config_id()}"
        else:
            on_dataset = "on_full_dataset"

        # Getting forward values
        forward_values_trl = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"train_trl_{on_dataset}",
            enable_compressed=False,
        )
        forward_value_rdl = get_forward_values(
            self.training_cfg,
            checkpoint,
            f"train_rdl_{on_dataset}",
            enable_compressed=False,
        )
        forward_values_all = ForwardValues.concat(
            forward_values_trl, forward_value_rdl, new_name="train_all"
        )

        # Need also the test values ?
        if self.full_dataset:
            forward_value_test = get_forward_values(
                self.training_cfg,
                checkpoint,
                f"test_on_full_dataset",
                enable_compressed=False,
            )

            forward_values_all = ForwardValues.concat(
                forward_values_all, forward_value_test, new_name="train_all"
            )

        # Sanity check
        if forward_values_all.global_index.size(0) != len(self.global_idx):
            raise RuntimeError(
                f"Incorrect loading of the forward values: {len(forward_values_all)} != {len(self.global_idx)}"
            )

        # ==================== PSMI values ====================

        # Logging
        logger.info(f"Computing P-SMI for config {self.training_cfg}")

        # Tensors
        X_per_layer = forward_values_all.mcq_states_per_layer
        y = forward_values_all.mcq_labels

        # Logging
        logger.debug(f"X size: {X_per_layer[1].size()}")
        logger.debug(f"y size: {y.size()}")

        # Uniprocess computation
        logger.debug("Starting P-SMI core computation")
        p_smi_per_layer = {
            layer: p_smi_estimator(X_per_layer[layer], y, n_estimator=self.n_estimator)
            for layer in SMI_LAYERS
        }
        logger.debug("End of P-SMI core computation")

        # ==================== Output ====================

        psmi_mean_per_layer_per_idx = collections.defaultdict(dict)
        psmi_max_per_layer_per_idx = collections.defaultdict(dict)
        psmi_min_per_layer_per_idx = collections.defaultdict(dict)

        for layer, (psmi_mean, psmi_max, psmi_min) in p_smi_per_layer.items():
            # The values in psmi_mean, etc are in the same order as in X used above
            # As a result, it is the same order as forward_values_all.global_index
            for count, idx in enumerate(forward_values_all.global_index.tolist()):
                psmi_mean_per_layer_per_idx[layer][idx] = psmi_mean[count]
                psmi_max_per_layer_per_idx[layer][idx] = psmi_max[count]
                psmi_min_per_layer_per_idx[layer][idx] = psmi_min[count]

        # Result
        result = []
        for container in [
            psmi_mean_per_layer_per_idx,
            psmi_max_per_layer_per_idx,
            psmi_min_per_layer_per_idx,
        ]:
            for layer in SMI_LAYERS:
                for idx in self.global_idx:
                    result.append(container[layer][idx])

        return result

    def get_p_smi_containers(
        self, metrics_df: pd.DataFrame
    ) -> t.Dict[int, t.Dict[int, float]]:
        """Returns (
            p_smi_mean_per_checkpoint_per_layer_per_idx,
            p_smi_max_per_checkpoint_per_layer_per_idx,
            p_smi_min_per_checkpoint_per_layer_per_idx,
        ).
        """

        # Checkpoints
        checkpoints = metrics_df.iloc[:, 0].tolist()

        # Init containers
        p_smi_mean_checkpoint_layer_idx = {
            chk: collections.defaultdict(dict) for chk in checkpoints
        }
        p_smi_max_checkpoint_layer_idx = {
            chk: collections.defaultdict(dict) for chk in checkpoints
        }
        p_smi_min_checkpoint_layer_idx = {
            chk: collections.defaultdict(dict) for chk in checkpoints
        }

        # Getting checkpoint idx
        for row_idx, chk in enumerate(checkpoints):
            for col_idx, col_name in enumerate(metrics_df.columns):

                # "checkpoint" column
                if col_idx == 0 or col_name == "epoch":
                    continue

                content = metrics_df.iloc[row_idx, col_idx]

                # Container
                if "mean_" in col_name:
                    container = p_smi_mean_checkpoint_layer_idx
                    layer_idx = col_name[len("mean_psmi_") :]
                elif "max_" in col_name:
                    container = p_smi_max_checkpoint_layer_idx
                    layer_idx = col_name[len("max_psmi_") :]
                elif "min_" in col_name:
                    container = p_smi_min_checkpoint_layer_idx
                    layer_idx = col_name[len("min_psmi_") :]
                else:
                    raise ValueError(f"Name: {col_name}")

                # Layer, idx
                layer = int(layer_idx.split("_")[0])
                idx = int(layer_idx.split("_")[1])

                container[chk][layer][idx] = content

        # Output
        return (
            p_smi_mean_checkpoint_layer_idx,
            p_smi_max_checkpoint_layer_idx,
            p_smi_min_checkpoint_layer_idx,
        )
