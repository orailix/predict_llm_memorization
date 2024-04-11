# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import torch
from loguru import logger
from tqdm import tqdm

from ..utils import DeploymentCfg, ForwardValues, get_forward_values, p_smi_estimator
from ..utils.constants import SMI_LAYERS
from .static_metrics_group import StaticMetricsGroup

SEP = ","


class PSmiStatic(StaticMetricsGroup):

    """Class used to compute Pointwise Sliced Mutual Information static metric."""

    def __init__(self, deployment_cfg: DeploymentCfg, n_estimator: int = 2000) -> None:
        super().__init__(deployment_cfg)
        self.n_estimator = n_estimator

    @property
    def metrics_group_name(self) -> str:
        return "p_smi_static"

    @property
    def metrics_names(self) -> t.List[str]:

        result = []
        for psmi_type in ["mean", "max", "min"]:
            for layer in SMI_LAYERS:
                for idx in self.global_idx:
                    result.append(f"{psmi_type}_psmi_{layer}_{idx}")

        return result

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # ==================== Init containers ====================

        num_model = len(self.training_cfg_list)
        num_idx = len(self.global_idx)
        num_layer = len(SMI_LAYERS)
        p_smi_mean_per_shadow_per_layer_per_idx = torch.zeros(
            num_model, num_layer, num_idx, dtype=float
        )
        p_smi_max_per_shadow_per_layer_per_idx = torch.zeros(
            num_model, num_layer, num_idx, dtype=float
        )
        p_smi_min_per_shadow_per_layer_per_idx = torch.zeros(
            num_model, num_layer, num_idx, dtype=float
        )

        for shadow_idx, shadow_training_cfg in enumerate(tqdm(self.training_cfg_list)):

            # ==================== Looking for ForwardValues ====================

            # Getting forward values
            forward_values_trl = get_forward_values(
                shadow_training_cfg,
                checkpoint,
                f"train_trl_on_full_dataset",
                enable_compressed=False,
            )
            forward_value_rdl = get_forward_values(
                shadow_training_cfg,
                checkpoint,
                f"train_rdl_on_full_dataset",
                enable_compressed=False,
            )
            forward_value_test = get_forward_values(
                shadow_training_cfg,
                checkpoint,
                f"test_on_full_dataset",
                enable_compressed=False,
            )
            forward_values_all = ForwardValues.concat(
                forward_values_trl, forward_value_rdl, new_name="train_all"
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
            logger.info(
                f"Computing P-SMI for shadow model {shadow_training_cfg.get_config_id()}"
            )

            # Tensors
            X_per_layer = forward_values_all.mcq_states_per_layer
            y = forward_values_all.mcq_labels

            # Logging
            logger.debug(f"X size: {X_per_layer[1].size()}")
            logger.debug(f"y size: {y.size()}")

            # Uniprocess computation
            logger.debug("Starting P-SMI core computation")
            p_smi_per_layer = {
                layer: p_smi_estimator(
                    X_per_layer[layer], y, n_estimator=self.n_estimator
                )
                for layer in SMI_LAYERS
            }
            logger.debug("End of P-SMI core computation")

            for layer, (psmi_mean, psmi_max, psmi_min) in p_smi_per_layer.items():
                # The values in psmi_mean, etc are in the same order as in X used above
                # As a result, it is the same order as forward_values_all.global_index
                for count, idx in enumerate(forward_values_all.global_index.tolist()):
                    p_smi_mean_per_shadow_per_layer_per_idx[shadow_idx][layer][
                        idx
                    ] = float(psmi_mean[count])
                    p_smi_max_per_shadow_per_layer_per_idx[shadow_idx][layer][
                        idx
                    ] = float(psmi_max[count])
                    p_smi_min_per_shadow_per_layer_per_idx[shadow_idx][layer][
                        idx
                    ] = float(psmi_min[count])

        # ==================== Output ====================

        psmi_mean_per_layer_per_idx = torch.mean(
            p_smi_mean_per_shadow_per_layer_per_idx, axis=0
        )
        psmi_max_per_layer_per_idx = torch.mean(
            p_smi_max_per_shadow_per_layer_per_idx, axis=0
        )
        psmi_min_per_layer_per_idx = torch.mean(
            p_smi_min_per_shadow_per_layer_per_idx, axis=0
        )

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
