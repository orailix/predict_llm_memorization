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
        - cfg: A configuration object
        - split: One of ["train", "test"]
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
    return load_dataset(cfg.dataset, *args, split=split)


def format_dataset(
    dataset: Dataset,
    cfg: TrainingCfg,
    seed: t.Optional[int] = None,
    force_template: bool = False,
) -> Dataset:
    """Formats a dataset.

    Args:
        cfg: A configuration object
        split: One of ["train", "test"]
        seed: If not None, a random state will be initiated with this seed and used for sampling the templates
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
        cfg: A configuration object
        seed: If not None, a random state will be initiated with this seed and used for sampling the templates
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
    return
