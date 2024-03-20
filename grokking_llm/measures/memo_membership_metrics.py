# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from typing import List

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from ..training import get_dataset, get_random_split
from ..utils import DeploymentCfg, TrainingCfg, get_possible_training_cfg
from .dynamic_metrics_group import DynamicMetricsGroup
from .utils.forward_values import ForwardValues


class MemoMembership(DynamicMetricsGroup):
    """Class used to compute memorization metrics.

    Memorization is defined as DP-distinguishability. As such, it is
    quantified by evaluating the accuracy of membership inference attacks.
    """

    def __init__(
        self,
        training_cfg: TrainingCfg,
        shadow_deployment_cfg: DeploymentCfg,
    ) -> None:

        # Parsing shadow_deployment_cfg
        self.shadow_deployment_cfg = shadow_deployment_cfg

        # List of global idx
        logger.debug(
            "Loading dataset to retrieve global IDX of the elements of the random split."
        )
        ds = get_dataset(training_cfg)
        ds = get_random_split(ds, training_cfg)
        self.global_idx = sorted(ds["global_index"])

        # Main initialization
        super().__init__(training_cfg)

        # Shadow training configurations and checkpoints
        logger.debug(
            "Loading shadow training configurations and corresponding checkpoints"
        )
        possible_training_cfg = get_possible_training_cfg(self.shadow_deployment_cfg)
        self.shadow_training_cfg_and_checkpoints = []
        for k in range(len(possible_training_cfg)):
            try:
                self.shadow_training_cfg_and_checkpoints.append(
                    (
                        possible_training_cfg[k],
                        possible_training_cfg[k].latest_checkpoint,
                    )
                )
            except IndexError:
                raise Exception(
                    f"You initialized a MemoMembership object, but the following shadow "
                    f"model was not trained: {possible_training_cfg[k].get_config_id()}"
                )

    @property
    def metrics_group_name(self) -> str:
        return f"memo_on_shadow_{self.shadow_deployment_cfg.get_deployment_id()}"

    @property
    def metrics_names(self) -> t.List[str]:
        return (
            ["prop_memo"] + ["mean_memo"] + [f"memo_{idx}" for idx in self.global_idx]
        )

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

        # Shadow forward values
        (
            shadow_forward_values,
            augmented_shadow_training_cfg_and_checkpoints,
        ) = self._get_shadow_values(checkpoint)

        # Unpacking some useful variables
        num_samples = len(self.global_idx)
        num_shadow = len(shadow_forward_values)

        # Logging
        # Recall that the first shadow model is a fake one, because it is the target model
        logger.debug(
            f"Using {num_shadow - 1} shadow model to attack {num_samples} target samples"
        )

        # Fetching the proba gap for each shadow model
        # Shape: `num_samples` entries, each enty has size `num_shadow`
        # At position proba_gaps[i][j] we find the proba gap for sample with index i and shadow model j
        logger.debug(
            "Fetching the proba gaps for each shadow model and target global idx"
        )
        proba_gaps = {
            target_global_idx: torch.zeros(num_shadow)
            for target_global_idx in self.global_idx
        }
        # Iterating over shadow values...
        for shadow_idx, shadow_values in enumerate(shadow_forward_values):
            # Iterating over the target global index for this shadow value...
            for count, target_global_idx in enumerate(
                shadow_values.global_index.tolist()
            ):
                # Extracting the proba gap
                target_predicted_proba = shadow_values.mcq_predicted_proba[
                    count
                ].tolist()
                true_label_index = shadow_values.inserted_label_index[count]
                label_proba = target_predicted_proba[true_label_index]
                other_proba = (
                    target_predicted_proba[:true_label_index]
                    + target_predicted_proba[true_label_index + 1 :]
                )
                target_proba_gap = label_proba - max(other_proba)

                # Saving it at the correct position
                proba_gaps[target_global_idx][shadow_idx] = target_proba_gap

        # Checking if each target sample was in the training set of the shadow models
        # Shape: `num_sample` entries, each entry has size `num_shadow`
        target_sample_is_in_shadow = {
            target_global_idx: [] for target_global_idx in self.global_idx
        }
        logger.debug(
            "Checking if each target sample was in the training set of the shadow models"
        )
        base_dataset = get_dataset(self.training_cfg)
        for shadow_training_cfg, _ in augmented_shadow_training_cfg_and_checkpoints:
            shadow_split = get_random_split(base_dataset, shadow_training_cfg)
            shadow_global_idx = set(shadow_split["global_index"])

            for target_global_idx in self.global_idx:
                target_sample_is_in_shadow[target_global_idx].append(
                    target_global_idx in shadow_global_idx
                )

        # Sanity check: the first shadow model is the target model, so we expect to have only "True"
        for target_global_idx in self.global_idx:
            if target_sample_is_in_shadow[target_global_idx][0] is False:
                raise RuntimeError(
                    "Inconsistency: the first shadow model should be the target model."
                )

        # Normal approximation
        num_pos = []
        num_neg = []
        target_global_idx_memo_score = []
        for target_global_idx in self.global_idx:

            # Getting proba_gaps for pos and neg shadow models
            pos_proba_gaps = [
                proba_gaps[target_global_idx][shadow_idx]
                for shadow_idx in range(1, num_shadow)
                if target_sample_is_in_shadow[target_global_idx][shadow_idx]
            ]
            neg_proba_gaps = [
                proba_gaps[target_global_idx][shadow_idx]
                for shadow_idx in range(1, num_shadow)
                if not target_sample_is_in_shadow[target_global_idx][shadow_idx]
            ]

            # Updating counts
            num_pos.append(len(pos_proba_gaps))
            num_neg.append(len(neg_proba_gaps))

            # Special case if there is no positive or negative
            if len(pos_proba_gaps) == 0:
                logger.warning(
                    f"No positive proba gap for the following global index: {target_global_idx}"
                )
                pos_mean = 0
                pos_std = 0.1
            else:
                pos_mean = np.mean(pos_proba_gaps)
                pos_std = max(np.std(pos_proba_gaps), 1e-2)

            if len(neg_proba_gaps) == 0:
                logger.warning(
                    f"No negative proba gap for the following global index: {target_global_idx}"
                )
                neg_mean = 0
                neg_std = 0.1
            else:
                neg_mean = np.mean(neg_proba_gaps)
                neg_std = max(np.std(neg_proba_gaps), 1e-2)

            # Evaluation of the target model
            def norm_pdf(mean, std, x):
                return (1 / std / np.sqrt(2 * np.pi)) * np.exp(
                    -1 * (x - mean) ** 2 / 2 / std / std
                )

            pos_likelihood = norm_pdf(
                pos_mean, pos_std, proba_gaps[target_global_idx][0]
            )
            neg_likelihood = norm_pdf(
                neg_mean, neg_std, proba_gaps[target_global_idx][0]
            )

            # Normalizing likelihoods
            pos_proba, neg_proba = torch.softmax(
                torch.Tensor([pos_likelihood, neg_likelihood]), dim=0
            ).tolist()
            target_global_idx_memo_score.append(pos_proba - neg_proba)

        # Logging
        mean_num_pos, min_num_pos, max_num_pos = (
            np.mean(num_pos),
            np.min(num_pos),
            np.max(num_pos),
        )
        mean_num_neg, min_num_neg, max_num_neg = (
            np.mean(num_neg),
            np.min(num_neg),
            np.max(num_neg),
        )
        num_memorized = sum([(item > 0) for item in target_global_idx_memo_score])
        mean_memorized = np.mean(target_global_idx_memo_score)
        logger.debug(
            f"Num positive per target sample: mean={mean_num_pos}, min={min_num_pos}, max={max_num_pos}"
        )
        logger.debug(
            f"Num negative per target sample: mean={mean_num_neg}, min={min_num_neg}, max={max_num_neg}"
        )
        logger.debug(
            f"Num target sample memorized: {num_memorized}/{num_samples} [{num_memorized/num_samples:.2%}]"
        )
        logger.debug(f"Mean memorization score: {mean_memorized}")

        # Output
        return [
            num_memorized / num_samples,
            mean_memorized,
        ] + target_global_idx_memo_score
