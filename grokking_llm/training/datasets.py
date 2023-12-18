"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import typing as t

import numpy as np
from datasets import Dataset, load_dataset
from loguru import logger
from transformers import AutoTokenizer

from ..utils.hf_hub import DS_ARC, DS_ETHICS, DS_MMLU
from .formatting import format_arc, format_ethics, format_label, format_mmlu
from .training_cfg import TrainingCfg

# Dataset splits
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


def get_dataset(
    cfg: TrainingCfg = None,
    split: str = TRAIN_SPLIT,
) -> Dataset:
    """Loads a dataset based on a training configuration.

    Args:
        cfg: A configuration object
        split: One of ["train", "test"]

    Returns:
        datasets.Dataset: The dataset.
    """

    # Sanity check
    if split not in [TRAIN_SPLIT, TEST_SPLIT]:
        raise ValueError(f"split={split} should be in {[[TRAIN_SPLIT, TEST_SPLIT]]}")

    # Loading the dataset
    if cfg.dataset == DS_ARC:
        args = ("ARC-Challenge",)
        split = "train" if split == TRAIN_SPLIT else "test"
    elif cfg.dataset == DS_MMLU:
        args = ("all",)
        split = "auxiliary_train" if split == TRAIN_SPLIT else "test"
    elif cfg.dataset == DS_ETHICS:
        args = ()
        split = "train" if split == TRAIN_SPLIT else "test"

    logger.debug(f"Loading dataset {cfg.dataset} split {split}")
    result = load_dataset(cfg.dataset, *args, split=split)

    # Add a index column
    result = result.add_column("index", np.array(range(len(result))))
    return result


def format_dataset(
    dataset: Dataset,
    cfg: TrainingCfg,
    seed: t.Optional[int] = None,
    force_template: bool = False,
) -> Dataset:
    """Formats a dataset.

    Args:
        dataset: The dataset to be formatted
        cfg: A configuration object
        split: One of ["train", "test"]
        seed: If not None, a random state will be initiated with this seed and used for sampling the templates

    Returns:
        datasets.Dataset: The dataset.
    """

    # Sampling determinism
    if seed is not None:
        random_state = np.random.RandomState(seed=seed)
    else:
        random_state = None

    # Formatting function
    if cfg.dataset == DS_ARC:
        formatting_fct = lambda args: format_arc(
            args, force_template=force_template, random_state=random_state
        )
    elif cfg.dataset == DS_ETHICS:
        formatting_fct = lambda args: format_ethics(
            args, force_template=force_template, random_state=random_state
        )
    elif cfg.dataset == DS_MMLU:
        formatting_fct = lambda args: format_mmlu(
            args, force_template=force_template, random_state=random_state
        )

    return dataset.map(formatting_fct)


def add_labels(
    dataset: Dataset,
    cfg: TrainingCfg,
    seed: t.Optional[int] = None,
) -> Dataset:
    """Adds the label at the end of a prompt.

    If some random noise is declared in the config, the label will be randomly flipped:
        - If it is flipped, sample["label_status"] will be set to "random"
        - If it is not, sample["label_status"] will be set to "true"

    Args:
        dataset: The dataset to which the labels will be added
        cfg: A configuration object
        seed: If not None, a random state will be initiated with this seed and used for sampling the templates

    Returns:
        datasets.Dataset: The dataset.
    """

    # Sampling determinism
    if seed is not None:
        random_state = np.random.RandomState(seed=seed)
    else:
        random_state = None

    # Formatting function
    formatting_fct = lambda args: format_label(
        args, random_state=random_state, label_noise=cfg.label_noise
    )

    return dataset.map(formatting_fct)


def get_random_split(
    dataset: Dataset,
    cfg: TrainingCfg,
) -> Dataset:
    """Gets a random split of a dataset.

    The config object contains the seed (used as a split id) and
    the proportion of sample that should be contained in the split.

    Args:
        dataset: The dataset from which to sample the split
        cfg: A configuration object

    Returns:
        datasets.Dataset: The dataset.
    """

    # Sapling indices
    random_state = np.random.RandomState(seed=cfg.split_id)
    num_row_to_sample = max(1, int(cfg.split_prop * len(dataset)))
    indices = random_state.choice(len(dataset), size=num_row_to_sample, replace=False)

    # Splitting
    result = dataset.select(
        indices,
        keep_in_memory=True,
    )

    return result
