"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import typing as t

import numpy as np
from datasets import Dataset, load_dataset
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizer

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

    logger.info(f"Loading dataset {cfg.dataset} split {split}")
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
        seed: If None, the seed specified in cfg.data_seed will be used to sample the tempates.
            Else, a random generator will be initialized with `seed` and used for this.

    Returns:
        datasets.Dataset: The dataset.
    """

    # Parsing args
    if seed is not None:
        logger.info(f"Overwriting cfg.data_seed with seed={seed}")
    else:
        seed = cfg.data_seed

    # Logging
    logger.info(f"Formatting dataset {cfg.dataset}")
    logger.debug(
        f"Using seed={seed} and force_template={force_template} for formatting."
    )

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

    return dataset.map(formatting_fct, remove_columns=dataset.column_names)


def add_labels(
    dataset: Dataset,
    cfg: TrainingCfg,
    seed: t.Optional[int] = None,
) -> Dataset:
    """Adds the label at the end of a prompt.

    If some random noise is declared in the config, the label will be randomly flipped:
        - If it is flipped, sample["cls_label_status"] will be set to "random"
        - If it is not, sample["cls_label_status"] will be set to "true"

    Args:
        dataset: The dataset to which the labels will be added
        cfg: A configuration object
        seed: If None, the seed specified in cfg.data_seed will be used to sample the label noise.
            Else, a random generator will be initialized with `seed` and used for this.

    Returns:
        datasets.Dataset: The dataset.
    """

    # Parsing args
    if seed is not None:
        logger.info(f"Overwriting cfg.data_seed with seed={seed}")
    else:
        seed = cfg.data_seed

    # Logging
    logger.info(f"Adding labels to dataset {cfg.dataset}")
    logger.debug(f"Using proportion label_noise={cfg.label_noise} with seed={seed}")

    # Sampling determinism
    random_state = np.random.RandomState(seed=seed)

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

    # Logging
    logger.info(f"Getting random split {cfg.split_id} from dataset {cfg.dataset}")
    logger.debug(
        f"Using split_prop={cfg.split_prop}, resulting in a split of length={len(result)}"
    )

    return result


def get_tokenizer(
    cfg: TrainingCfg,
) -> PreTrainedTokenizer:
    """Gets the tokenizer for a dataset.

    Args:
        cfg: A configuration object.

    Returns:
        transformers.PreTrainedTokenizer: The tokenizer.
    """

    # Creating the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model,
        model_max_length=cfg.max_len,
        padding_side="left",
        truncation_side="left",
        add_eos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenize_dataset(
    dataset: Dataset,
    cfg: TrainingCfg,
) -> Dataset:
    """Tokenizes the dataset.

    The config object contains a `max_len` attribute that will be used
    for padding and clipping the prompts (this is done to speed up training).

    Args:
        dataset: The dataset to tokenize.
        cfg: A configuration object.

    Returns:
        datasets.Dataset: The dataset.
    """

    # Logging
    logger.info(f"Tokenizing dataset {cfg.dataset}")
    logger.debug(
        f"Using model_max_length={cfg.max_len} and vocabulary from model={cfg.model}"
    )

    # Getting the tokenizer
    tokenizer = get_tokenizer(cfg)

    # Mapping function
    def map_fct(sample):
        result = tokenizer(
            sample["prompt"],
            truncation=True,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    # Mapping
    return dataset.map(map_fct)
