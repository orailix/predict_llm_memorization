# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import typing as t

import pandas as pd


def get_static_container(
    metrics_df: pd.DataFrame,
    column_offset: int,
) -> t.Dict[int, t.Dict[int, float]]:
    """
    `metrics_df` is supposed to be the result of metrics.load_metrics_df()
    where `metrics` is an instance of one of CounterfactualSimplicityStatic,
    CounterfactualMemoStatic, LossStatic, MemoLogitGapStatic, or MemoMembershipStatic.

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
