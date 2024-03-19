# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from typing import List

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from ..deploy.deployment_cfg import DeploymentCfg
from ..deploy.prepare import get_possible_training_cfg
from ..training import TrainingCfg, get_dataset, get_random_split
from .dynamic_metrics_group import DynamicMetricsGroup
from .utils.forward_values import ForwardValues


class MemoProbaGap(DynamicMetricsGroup):
    """Class used to compute memorization metrics.

    Memorization is defined as the probability gap between the true label
    and the following label  with greatest predicted probability.
    """

    def __init__(
        self,
        training_cfg: TrainingCfg,
    ) -> None:

        # List of global idx
        logger.debug(
            "Loading dataset to retrieve global IDX of the elements of the random split."
        )
        ds = get_dataset(training_cfg)
        ds = get_random_split(ds, training_cfg)
        self.global_idx = sorted(ds["global_index"])

        # Main initialization
        super().__init__(training_cfg)

    @property
    def metrics_group_name(self) -> str:
        return "memo_proba_gap"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["memo_prop"] + [f"memo_{idx}" for idx in self.global_idx]

    def _get_shadow_values(self, checkpoint):
        """TODO"""

        # Forward values on self for all shadow models
        # We add a fake shadow model, which is the target model, at index 0 in augmented_shadow_training_cfg_and_checkpoints
        # The purpose of this addition is that we load the proba_gap at the same time that we
        # do it for the shadow models, without additional code
        logger.info(f"Loading forward values from shadow models")
        shadow_forward_values = []
        augmented_shadow_training_cfg_and_checkpoints = [
            (self.training_cfg, checkpoint)
        ]
        augmented_shadow_training_cfg_and_checkpoints += (
            self.shadow_training_cfg_and_checkpoints.copy()
        )
        for count, (training_cfg, latest_checkpoint) in enumerate(
            tqdm(augmented_shadow_training_cfg_and_checkpoints)
        ):

            # Skipping a shadow model if it was trained on the same config than the target config
            if (count > 0) and (training_cfg == self.training_cfg):
                continue

            # Paths
            forward_export_dir = (
                training_cfg.get_output_dir()
                / f"checkpoint-{latest_checkpoint}"
                / "forward_values"
            )
            trl_path = (
                forward_export_dir
                / f"train_trl_on_{self.training_cfg.get_config_id()}.safetensors"
            )
            rdl_path = (
                forward_export_dir
                / f"train_rdl_on_{self.training_cfg.get_config_id()}.safetensors"
            )

            # Special case for the target model - we authorize either train_trl_on_[config_id].safetensors
            # or simply train_trl.safetensors
            if count == 0:
                if not trl_path.is_file():
                    trl_path = forward_export_dir / f"train_trl.safetensors"
                if not rdl_path.is_file():
                    rdl_path = forward_export_dir / f"train_rdl.safetensors"

            # Loading forward values
            trl_forward_values = ForwardValues.load(trl_path)
            rdl_forward_values = ForwardValues.load(rdl_path)
            all_forward_values = ForwardValues.concat(
                trl_forward_values, rdl_forward_values, "train_all"
            )

            # Saving
            shadow_forward_values.append(all_forward_values)

        # Output
        return shadow_forward_values, augmented_shadow_training_cfg_and_checkpoints

    def metrics_computation_core(self, checkpoint: int) -> List[float]:

        # Paths of forward values
        forward_export_dir = (
            self.training_cfg.get_output_dir()
            / f"checkpoint-{checkpoint}"
            / "forward_values"
        )
        trl_path = (
            forward_export_dir
            / f"train_trl_on_{self.training_cfg.get_config_id()}.safetensors"
        )
        rdl_path = (
            forward_export_dir
            / f"train_rdl_on_{self.training_cfg.get_config_id()}.safetensors"
        )

        # We authorize either train_trl_on_[config_id].safetensors
        # or simply train_trl.safetensors
        if not trl_path.is_file():
            trl_path = forward_export_dir / f"train_trl.safetensors"
        if not rdl_path.is_file():
            rdl_path = forward_export_dir / f"train_rdl.safetensors"

        # Loading forward values
        trl_forward_values = ForwardValues.load(trl_path)
        rdl_forward_values = ForwardValues.load(rdl_path)
        all_forward_values = ForwardValues.concat(
            trl_forward_values, rdl_forward_values, "train_all"
        )

        # Unpacking some useful variables
        num_samples = len(self.global_idx)

        # Fetching the proba gap for each shadow model
        logger.debug(
            "Fetching the proba gaps for each shadow model and target global idx"
        )
        proba_gaps = dict()
        # Iterating over the target global index for this shadow value...
        for count, target_global_idx in enumerate(
            all_forward_values.global_index.tolist()
        ):
            # Extracting the proba gap
            target_predicted_proba = all_forward_values.mcq_predicted_proba[
                count
            ].tolist()
            true_label_index = all_forward_values.inserted_label_index[count]
            label_proba = target_predicted_proba[true_label_index]
            other_proba = (
                target_predicted_proba[:true_label_index]
                + target_predicted_proba[true_label_index + 1 :]
            )
            target_proba_gap = label_proba - max(other_proba)

            # Saving it at the correct position
            proba_gaps[target_global_idx] = target_proba_gap

        # Sorting
        memorization_score = [
            proba_gaps[target_global_idx] for target_global_idx in self.global_idx
        ]

        # Logging
        logger.debug(f"Mean memorization score: {np.mean(memorization_score)}")

        # Output
        return [np.mean(memorization_score)] + memorization_score
