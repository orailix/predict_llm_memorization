# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import torch
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from ..measures_dyn import PSmiMetrics
from ..utils import DeploymentCfg, TrainingCfg, get_p_smi_containers
from ..utils.constants import SMI_LAYERS
from .static_metrics_group import StaticMetricsGroup


class PSmiStatic(StaticMetricsGroup):

    """Class used to compute Pointwise Sliced Mutual Information static metric."""

    def __init__(
        self, deployment_cfg: DeploymentCfg, n_estimator: int = 2000, njobs: int = 1
    ) -> None:
        super().__init__(deployment_cfg)
        logger.info(f"Using n_estimator={n_estimator} and njobs={njobs}")
        self.n_estimator = n_estimator
        self.njobs = njobs

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

        # ==================== Loading static metrics ====================

        logger.info(f"Loading dynamic P-SMI values")
        dynamic_p_smi_containers = Parallel(n_jobs=self.njobs)(
            delayed(load_dyn_df)(training_cfg)
            for training_cfg in self.training_cfg_list
        )

        layer_to_layer_count = {
            layer: count for count, layer in enumerate(sorted(SMI_LAYERS))
        }
        idx_to_idx_count = {
            idx: count for count, idx in enumerate(sorted(self.global_idx))
        }

        logger.info(f"Filling values of P-SMI")
        for shadow_idx, (psmi_mean, psmi_max, psmi_min) in enumerate(
            tqdm(dynamic_p_smi_containers)
        ):
            for layer in SMI_LAYERS:
                for idx in self.global_idx:

                    # Counters
                    layer_count = layer_to_layer_count[layer]
                    idx_count = idx_to_idx_count[idx]

                    # Countainers
                    p_smi_mean_per_shadow_per_layer_per_idx[shadow_idx][layer_count][
                        idx_count
                    ] = psmi_mean[checkpoint][layer][idx]
                    p_smi_max_per_shadow_per_layer_per_idx[shadow_idx][layer_count][
                        idx_count
                    ] = psmi_max[checkpoint][layer][idx]
                    p_smi_min_per_shadow_per_layer_per_idx[shadow_idx][layer_count][
                        idx_count
                    ] = psmi_min[checkpoint][layer][idx]

        # ==================== Filling containers ====================

        psmi_mean_per_layer_per_idx = torch.mean(
            p_smi_mean_per_shadow_per_layer_per_idx, axis=0
        )
        psmi_max_per_layer_per_idx = torch.mean(
            p_smi_max_per_shadow_per_layer_per_idx, axis=0
        )
        psmi_min_per_layer_per_idx = torch.mean(
            p_smi_min_per_shadow_per_layer_per_idx, axis=0
        )

        # ==================== Results ====================

        result = []
        for container in [
            psmi_mean_per_layer_per_idx,
            psmi_max_per_layer_per_idx,
            psmi_min_per_layer_per_idx,
        ]:
            for layer in SMI_LAYERS:
                for idx in self.global_idx:
                    # Counters
                    layer_count = layer_to_layer_count[layer]
                    idx_count = idx_to_idx_count[idx]

                    # Result
                    result.append(container[layer_count][idx_count])

        return result


def load_dyn_df(training_cfg: TrainingCfg):
    metrics = PSmiMetrics(training_cfg=training_cfg, full_dataset=True)
    metrics_df = metrics.load_metrics_df()
    return get_p_smi_containers(metrics_df)
