# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from .forward_values import ForwardValues, get_forward_values
from .training_cfg import TrainingCfg


def get_pointwise_container(
    metrics_df: pd.DataFrame,
    column_offset: int,
) -> t.Dict[int, t.Dict[int, float]]:
    """
    `metrics_df` is supposed to be the result of metrics.load_metrics_df()
    where `metrics` is an instance of one of:
    - Static measures: CounterfactualSimplicityStatic, CounterfactualMemoStatic, LossStatic,
        MemoLogitGapStatic, or MemoMembershipStatic, MemoLogitGapStdStatic
    - Dynamic measures: MemoMembershipMetrics, SampleLossMetrics, LogitGapMetrics

    Both static and dynamic P-SMI measures are excluded from this format, because they are
    more complex and include layer and smi-type details.
    See `sm.get_p_smi_containers` for a similar funciton for P-SMI measures.

    `column_offset` is an int corresponding to the number of column to skip in the df,
    in addition to the "checkpoint" column, that is always skipped. For example for LossStatic,
    the first column is the checkpoint, the second is the mean loss, and then starting
    from the third one we have the individual values. Thus, there is one column to skip in addition
    to the checkpoint one, so column_offset=1.

    Returns values_per_checkpoint_per_idx.
    """

    # Init container
    values_per_checkpoint_per_idx = collections.defaultdict(dict)

    # Checkpoints
    checkpoints = metrics_df.iloc[:, 0].tolist()

    # Getting values
    for row_idx, chk in enumerate(checkpoints):
        for col_idx, col_name in enumerate(metrics_df.columns):

            # Offset
            if col_idx <= column_offset or col_name == "epoch":
                continue

            content = metrics_df.iloc[row_idx, col_idx]
            idx = int(col_name.split("_")[1])

            values_per_checkpoint_per_idx[chk][idx] = content

    # Output
    return values_per_checkpoint_per_idx


@dataclass
class LightForwardValues:
    """A class with only global_index, mcq_predicted_proba, mcq_predicted_logits, inserted_label_index,
    because they are the only part useful for MIA."""

    global_index: torch.Tensor
    loss_asw: torch.Tensor
    mcq_predicted_logits: torch.Tensor
    inserted_label_index: torch.Tensor

    @classmethod
    def from_forward_values(cls, forward_values: ForwardValues):
        return cls(
            global_index=forward_values.global_index,
            loss_asw=forward_values.loss_asw,
            mcq_predicted_logits=forward_values.mcq_predicted_logits,
            inserted_label_index=forward_values.inserted_label_index,
        )


def get_shadow_forward_values_for_pointwise(
    training_cfg_list: t.List[TrainingCfg],
    checkpoint: t.Optional[int] = None,
    on_dataset="full_dataset",
) -> t.List[LightForwardValues]:
    """Gets forward values of all shadow models.

    Args:
        - training_cfg_list: A list of TrainingCfg referring to the shadow models
        - checkpoint: If not None, the checkpoint at which to lokk at forward values.
        If None, the latest checkpoint will be used.
        - on_dataset: If "full_dataset", the test set will be included as well.
        If not, it is supposed to be the config_id of a valid target TrainingCfg.
    """

    # Logging
    logger.info(f"Loading forward values from shadow models")
    shadow_forward_values = []
    for shadow_cfg in tqdm(training_cfg_list):

        if checkpoint is None:
            current_checkpoint = shadow_cfg.latest_checkpoint
            logger.debug(
                f"Latest checkpoint found for config {shadow_cfg.get_config_id()} : {current_checkpoint}"
            )
        else:
            current_checkpoint = checkpoint

        # Getting forward values
        forward_values_trl = get_forward_values(
            training_cfg=shadow_cfg,
            checkpoint=current_checkpoint,
            name=f"train_trl_on_{on_dataset}",
            enable_compressed=True,
        )
        forward_values_rdl = get_forward_values(
            training_cfg=shadow_cfg,
            checkpoint=current_checkpoint,
            name=f"train_rdl_on_{on_dataset}",
            enable_compressed=True,
        )
        forward_values_all = ForwardValues.concat(
            forward_values_trl, forward_values_rdl, "train_all"
        )

        # Converting to LightForwardValues and Saving
        shadow_forward_values.append(
            LightForwardValues.from_forward_values(forward_values_all)
        )

    # Output
    return shadow_forward_values


def get_logit_gaps_for_pointwise(
    forward_values_list: t.List[LightForwardValues],
    global_idx: t.List[int],
) -> t.Dict[int, torch.Tensor]:
    """Fetches the logit gap for each shadow model.
    Shape: `num_samples` entries, each enty has size `num_shadow`
    At position logits_gaps[i][j] we find the logits gap for sample with index i and shadow model j
    """

    # Init
    logger.debug("Fetching the logits gaps for each shadow model and target global idx")
    num_shadow = len(forward_values_list)
    logits_gaps = {
        target_global_idx: torch.zeros(num_shadow) for target_global_idx in global_idx
    }

    # Converting into set to accelerate lookup
    global_idx = set(global_idx)

    # Iterating over shadow values...
    for shadow_idx, shadow_values in enumerate(tqdm(forward_values_list)):
        # Iterating over the target global index for this shadow value...
        for count, target_global_idx in enumerate(shadow_values.global_index.tolist()):

            # Skipping idx that are not in global_idx
            if target_global_idx not in global_idx:
                continue

            # Extracting the logits gap
            target_predicted_logits = shadow_values.mcq_predicted_logits[count].tolist()
            true_label_index = shadow_values.inserted_label_index[count]
            label_logits = target_predicted_logits[true_label_index]
            other_logits = (
                target_predicted_logits[:true_label_index]
                + target_predicted_logits[true_label_index + 1 :]
            )
            target_logits_gap = label_logits - max(other_logits)

            # Saving it at the correct position
            logits_gaps[target_global_idx][shadow_idx] = target_logits_gap

    # Output
    return logits_gaps


def get_losses_for_pointwise(
    forward_values_list: t.List[LightForwardValues],
    global_idx: t.List[int],
) -> t.Dict[int, torch.Tensor]:
    """Fetches the losses for each shadow model.
    Shape: `num_samples` entries, each enty has size `num_shadow`
    At position brier_scores[i][j] we find the loss for sample with index i and shadow model j
    """

    # Init
    logger.debug("Fetching the losses for each shadow model and target global idx")
    num_shadow = len(forward_values_list)
    losses = {
        target_global_idx: torch.zeros(num_shadow) for target_global_idx in global_idx
    }

    # Converting into set to accelerate lookup
    global_idx = set(global_idx)

    # Iterating over shadow values...
    for shadow_idx, shadow_values in enumerate(tqdm(forward_values_list)):
        # Iterating over the target global index for this shadow value...
        for count, target_global_idx in enumerate(shadow_values.global_index.tolist()):

            # Skipping idx that are not in global_idx
            if target_global_idx not in global_idx:
                continue

            # Saving at the correct position
            losses[target_global_idx][shadow_idx] = shadow_values.loss_asw[count]

    # Output
    return losses


# Utils
def norm_pdf(mean, std, x):
    return (1 / std / np.sqrt(2 * np.pi)) * np.exp(-1 * (x - mean) ** 2 / 2 / std / std)


def get_mia_memo_score(pos_likelihood, neg_likelihood, epsilon):

    if pos_likelihood < epsilon and neg_likelihood < epsilon:
        return 0

    total = pos_likelihood + neg_likelihood
    pos_proba = pos_likelihood / total
    neg_proba = neg_likelihood / total

    return pos_proba - neg_proba
