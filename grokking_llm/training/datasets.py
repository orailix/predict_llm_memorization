# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import datasets
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizer

from ..utils import TrainingCfg, paths
from ..utils.constants import (
    ARC_MAX_SIZE,
    ETHICS_MAX_SIZE,
    MAX_NUM_MCQ_ANSWER,
    MMLU_MAX_SIZE,
)
from ..utils.hf_hub import (
    DS_ARC,
    DS_ETHICS,
    DS_MMLU,
    MOD_DUMMY_LLAMA,
    MOD_MISTRAL_7B,
    TOK_DUMMY_LLAMA,
)
from .formatting import format_arc, format_ethics, format_label, format_mmlu

# Dataset splits
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
TRAIN_SPLIT_HF = "train_hf"
TEST_SPLIT_HF = "test_hf"

# Disable caching
datasets.disable_caching()


def save_dataset(
    cfg: TrainingCfg = None,
):
    """Saves a dataset so it can be used offline."""
    # Loading full dataset
    if cfg.dataset == DS_ARC:
        args = ("ARC-Challenge",)
    elif cfg.dataset == DS_MMLU:
        args = ("all",)
    elif cfg.dataset == DS_ETHICS:
        args = ()

    ds = datasets.load_dataset(cfg.dataset, *args)
    save_path = paths.hf_home / "saved_datasets" / cfg.dataset
    save_path.mkdir(exist_ok=True, parents=True)
    ds.save_to_disk(str(save_path))


def get_dataset(
    cfg: TrainingCfg = None,
    split: str = TRAIN_SPLIT,
) -> datasets.Dataset:
    """Loads a dataset based on a training configuration.

    Args:
        cfg: A configuration object
        split: One of ["train", "test", "train_hf", "test_hf"]

        "train_hf" and "test_hf" refer to the train and test split as present in HuggingFace.
        "train" and "test" are randomly selected (but always the same) within the "train_hf"
        + "test_hf" splits, with "test" split size being 12.5% ot "train" split size.

    Returns:
        datasets.Dataset: The dataset.
    """

    # Sanity check
    if split not in [TRAIN_SPLIT, TEST_SPLIT, TRAIN_SPLIT_HF, TEST_SPLIT_HF]:
        raise ValueError(
            f"split={split} should be in {[[TRAIN_SPLIT, TEST_SPLIT, TRAIN_SPLIT_HF, TEST_SPLIT_HF]]}"
        )

    # Loading full dataset
    if cfg.dataset == DS_ARC:
        args = ("ARC-Challenge",)
        split_train = "train"
        split_test = "test"
    elif cfg.dataset == DS_MMLU:
        args = ("all",)
        split_train = "auxiliary_train"
        split_test = "test"
    elif cfg.dataset == DS_ETHICS:
        args = ()
        split_train = "train"
        split_test = "test"

    # Train set
    try:
        ds_train = datasets.load_dataset(cfg.dataset, *args, split=split_train)
    except:
        ds = datasets.load_from_disk(paths.hf_home / "saved_datasets" / cfg.dataset)
        ds_train = ds[split_train]
    ds_train = ds_train.add_column("global_index", np.array(range(len(ds_train))))
    if cfg.dataset == DS_MMLU:
        ds_train = ds_train.select(range(MMLU_MAX_SIZE))
    if cfg.dataset == DS_ARC:
        ds_train = ds_train.select(range(ARC_MAX_SIZE))
    if cfg.dataset == DS_ETHICS:
        ds_train = ds_train.select(range(ETHICS_MAX_SIZE))

    # Test set
    try:
        ds_test = datasets.load_dataset(cfg.dataset, *args, split=split_test)
    except:
        ds = datasets.load_from_disk(paths.hf_home / "saved_datasets" / cfg.dataset)
        ds_test = ds[split_test]
    ds_test = ds_test.add_column(
        "global_index", np.array(range(len(ds_train), len(ds_train) + len(ds_test)))
    )
    if cfg.dataset == DS_MMLU:
        ds_test = ds_test.select([])

    # Output
    if split == TRAIN_SPLIT_HF:
        result = ds_train
    elif split == TEST_SPLIT_HF:
        result = ds_test
    else:
        ds_full = datasets.concatenate_datasets([ds_train, ds_test])

        # Selecting
        full_length = len(ds_full)
        test_selector = np.random.RandomState(0).choice(
            range(full_length), int(1 / 9 * full_length), replace=False
        )
        test_selector_set = set(test_selector)
        train_selector = [k for k in range(full_length) if k not in test_selector_set]

        if split == TRAIN_SPLIT:
            result = ds_full.select(train_selector)
        elif split == TEST_SPLIT:
            result = ds_full.select(test_selector)

    # Logging
    logger.info(f"Loading dataset {cfg.dataset} split {split}")

    return result


def format_dataset(
    dataset: datasets.Dataset,
    cfg: TrainingCfg,
    seed: t.Optional[int] = None,
    force_template: bool = False,
) -> datasets.Dataset:
    """Formats a dataset.

    Args:
        dataset: The dataset to be formatted
        cfg: A configuration object
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
    dataset: datasets.Dataset,
    cfg: TrainingCfg,
    split: str,
    *,
    seed: t.Optional[int] = None,
) -> datasets.Dataset:
    """Adds the label at the end of a prompt.

    If some random noise is declared in the config, the label will be randomly flipped:
        - If it is flipped, sample["cls_label_status"] will be set to 0
        - If it is not, sample["cls_label_status"] will be set to 1

    Args:
        dataset: The dataset to which the labels will be added
        cfg: A configuration object
        split: One of ["train", "test"]. If it is the test set, no noise will be added.
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
    label_noise = 0 if split == TEST_SPLIT else cfg.label_noise
    formatting_fct = lambda args: format_label(
        args, random_state=random_state, label_noise=label_noise
    )

    return dataset.map(formatting_fct)


def get_random_split(
    dataset: datasets.Dataset,
    cfg: TrainingCfg,
) -> datasets.Dataset:
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

    # Special case for Dummy Llama, which has no tokenizer
    if cfg.model == MOD_DUMMY_LLAMA:
        model_to_ask = TOK_DUMMY_LLAMA
    else:
        model_to_ask = cfg.model

    # Creating the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_to_ask,
        model_max_length=cfg.max_len,
        padding_side="left",
        truncation_side="left",
        add_eos_token=True,
    )
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )

    return tokenizer


def tokenize_dataset(
    dataset: datasets.Dataset,
    cfg: TrainingCfg,
) -> datasets.Dataset:
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


def add_tokenized_possible_labels(
    dataset: datasets.Dataset,
    cfg: TrainingCfg,
) -> datasets.Dataset:
    """Adds tokenized versions of the possible labels.

    The dataset is expected to be first tokenized, e.g. with `tokenize_dataset` function.

    1. The possible labels are tokenized and inserted in a list stored in dataset["tokenized_possible_labels"]
    This list has size MAX_NUM_MCQ_ANSWER and is padded with value 0.
    2. The index in this list of the label that has been added during formatting is stored
    in dataset["inserted_label_index"]. In case of label noise, this is not necessary the true label.
    """

    # Logging
    logger.info(f"Adding tokenized possible labels to dataset {cfg.dataset}")

    # Preparing the tokenizer
    tokenizer = get_tokenizer(cfg)

    # Mapping function
    def map_fct(sample):
        # We add \n\n to avoid differences e.g. between "_A" and "A" tokens.
        # We get value at index -2 because of the EOS token.
        tokenized_labels = [
            tokenizer.encode("\n\n" + label)[-2]
            for label in sample["possible_cls_labels"]
        ]

        # Clipping
        tokenized_labels = tokenized_labels[:MAX_NUM_MCQ_ANSWER]

        # Padding
        tokenized_labels += (MAX_NUM_MCQ_ANSWER - len(tokenized_labels)) * [0]

        # Index of true label
        inserted_label_index = tokenized_labels.index(sample["input_ids"][-2])

        # Output
        return {
            "tokenized_possible_labels": tokenized_labels,
            "inserted_label_index": inserted_label_index,
        }

    # Mapping
    return dataset.map(map_fct)
