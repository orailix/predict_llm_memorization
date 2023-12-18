"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import copy
import typing as t
from configparser import ConfigParser
from pathlib import Path

import torch
from loguru import logger

from ..utils.constants import *
from ..utils.hf_hub import DS_ARC, DS_ETHICS, DS_MMLU, MOD_LLAMA_7B, MOD_MISTRAL_7B


class TrainingCfg:
    """Represents a training configuration."""

    # ==================== UTILS ====================

    def __repr__(self) -> str:
        return f"""TrainConfig object:
MAIN:
    - model : {self.model}
    - dataset : {self.dataset}
PREPROCESS:
    - max_len: {self.max_len}
    - label_noise: {self.label_noise}
RANDOM SPLIT:
    - split_id: {self.split_id}
    - split_prop: {self.split_prop}
LoRA:
    - lora_r: {self.lora_r}
    - lora_alpha: {self.lora_alpha}
    - lora_dropout: {self.lora_dropout}
DEVICES:
    - accelerator: {self.accelerator}
"""

    def copy(self):
        return copy.copy(self)

    # ==================== BUILD ====================

    @classmethod
    def from_file(cls, path: t.Union[str, Path]):
        """Builds a config object from a config file.

        Args:
            - path: The path to the config file"""

        # PARSING CONFIG
        parser = ConfigParser()
        parser.read(path)

        return cls.from_parser(parser)

    @classmethod
    def from_parser(cls, parser: ConfigParser):
        """Builds a config object from a parsed config file.

        Args:
            - path: The configparser.ConfigParser object representing the parsed config.
        """

        # MAIN CONFIG

        if "main" not in parser:
            raise ValueError("Your config should contain a 'main' section.")
        if "model" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'model' entry."
            )
        if "dataset" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'dataset' entry."
            )

        model = parser["main"]["model"]
        dataset = parser["main"]["dataset"]

        # PREPROCESS CONFIG

        if "preprocess" not in parser:
            raise ValueError("Your config should contain a 'preprocess' section.")
        if "max_len" not in parser["preprocess"]:
            raise ValueError(
                "Section 'preprocess' of your config should contain a 'max_len' entry."
            )
        if "label_noise" not in parser["preprocess"]:
            raise ValueError(
                "Section 'preprocess' of your config should contain a 'label_noise' entry."
            )

        max_len = parser["preprocess"]["max_len"]
        label_noise = parser["preprocess"]["label_noise"]

        # RANDOM SPLIT CONFIG

        if "random_split" not in parser:
            raise ValueError("Your config should contain a 'random_split' section.")
        if "split_id" not in parser["random_split"]:
            raise ValueError(
                "Section 'random_split' of your config should contain a 'split_id' entry."
            )
        if "split_prop" not in parser["random_split"]:
            raise ValueError(
                "Section 'random_split' of your config should contain a 'split_prop' entry."
            )

        split_id = parser["random_split"]["split_id"]
        split_prop = parser["random_split"]["split_prop"]

        # LORA CONFIG

        if "lora" not in parser:
            raise ValueError("Your config should contain a 'lora' section.")
        if "lora_r" not in parser["lora"]:
            raise ValueError(
                "Section 'lora' of your config should contain a 'lora_r' entry."
            )
        if "lora_alpha" not in parser["lora"]:
            raise ValueError(
                "Section 'lora' of your config should contain a 'lora_alpha' entry."
            )
        if "lora_dropout" not in parser["lora"]:
            raise ValueError(
                "Section 'lora' of your config should contain a 'lora_dropout' entry."
            )

        lora_r = parser["lora"]["lora_r"]
        lora_alpha = parser["lora"]["lora_alpha"]
        lora_dropout = parser["lora"]["lora_dropout"]

        # DEVICES CONFIG

        if "devices" not in parser:
            raise ValueError("Your config should contain a 'devices' section.")
        if "accelerator" not in parser["devices"]:
            raise ValueError(
                "Section 'devices' of your config should contain a 'accelerator' entry."
            )

        accelerator = parser["devices"]["accelerator"]

        # OUTPUT

        return cls(
            model=model,
            dataset=dataset,
            max_len=max_len,
            label_noise=label_noise,
            split_id=split_id,
            split_prop=split_prop,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            accelerator=accelerator,
        )

    def __init__(
        self,
        *,
        model: str = TRAIN_CFG_DEFAULT_MODEL,
        dataset: str = TRAIN_CFG_DEFAULT_DATASET,
        max_len: int = TRAIN_CFG_DEFAULT_MAX_LEN,
        label_noise: float = TRAIN_CFG_DEFAULT_NOISE,
        split_id: int = TRAIN_CFG_DEFAULT_SPLIT_ID,
        split_prop: float = TRAIN_CFG_DEFAULT_SPLIT_PROP,
        lora_r: int = TRAIN_CFG_DEFAULT_LORA_R,
        lora_alpha: float = TRAIN_CFG_DEFAULT_LORA_ALPHA,
        lora_dropout: float = TRAIN_CFG_DEFAULT_LORA_DROPOUT,
        accelerator: str = TRAIN_CFG_DEFAULT_ACCELERATOR,
    ):
        """Safely builds a config object from kwargs."""

        # MAIN CONFIG

        if model in [MOD_MISTRAL_7B, MOD_LLAMA_7B]:
            self.model = model
        elif model.lower() == TRAIN_CFG_MISTRAL:
            self.model = MOD_MISTRAL_7B
        elif model.lower() == TRAIN_CFG_LLAMA:
            self.model = MOD_LLAMA_7B
        else:
            raise ValueError(
                f"`model`={model} should be in {[TRAIN_CFG_MISTRAL, TRAIN_CFG_LLAMA, MOD_MISTRAL_7B, MOD_LLAMA_7B]}."
            )

        if dataset in [DS_ARC, DS_ETHICS, DS_MMLU]:
            self.dataset = dataset
        elif dataset.lower() == TRAIN_CFG_ARC:
            self.dataset = DS_ARC
        elif dataset.lower() == TRAIN_CFG_MMLU:
            self.dataset = DS_MMLU
        elif dataset.lower() == TRAIN_CFG_ETHICS:
            self.dataset = DS_ETHICS
        else:
            raise ValueError(
                f"`dataset`={dataset}  should be in {[TRAIN_CFG_ARC, TRAIN_CFG_MMLU, TRAIN_CFG_ETHICS, DS_ARC, DS_ETHICS, DS_MMLU]}."
            )

        # PREPROCESSING CONFIG

        try:
            self.max_len = int(max_len)
            if self.max_len <= 0:
                raise ValueError()
        except ValueError:
            raise ValueError(f"`max_len`={max_len} should be a positive int.")

        try:
            self.label_noise = float(label_noise)
            if self.label_noise < 0 or self.label_noise > 1:
                raise ValueError()
        except ValueError:
            raise ValueError(
                f"`label_noise`={label_noise} should be a float between 0 and 1."
            )

        # RANDOM SPLIT CONFIG

        try:
            self.split_id = int(split_id)
        except ValueError:
            raise ValueError(f"`split_id`={split_id} should be an int.")

        try:
            self.split_prop = float(split_prop)
            if self.split_prop < 0 or self.split_prop > 1.0:
                raise ValueError()
        except ValueError:
            raise ValueError(
                f"`split_prop`={split_prop} should be a float between 0.0 and 1.0."
            )

        # LORA CONFIG

        try:
            self.lora_r = int(lora_r)
            if self.lora_r <= 0:
                raise ValueError()
        except ValueError:
            raise ValueError(f"`lora_r`={lora_r} should be a positive int.")

        try:
            self.lora_alpha = float(lora_alpha)
            if self.lora_alpha <= 0:
                raise ValueError()
        except ValueError:
            raise ValueError(f"`lora_alpha`={lora_alpha} should be a positive float.")

        try:
            self.lora_dropout = float(lora_dropout)
            if self.lora_dropout < 0 or self.lora_dropout > 1:
                raise ValueError()
        except ValueError:
            raise ValueError(
                f"`lora_dropout`={lora_dropout} should be a float between 0 and 1."
            )

        # DEVICE CONFIG

        try:
            self.accelerator = str(accelerator)
        except ValueError:
            raise ValueError(f"`accelerator`={accelerator} should be a string.")

        try:
            # Check for device compatibility
            d = torch.device(self.accelerator)
            torch.rand(1).to(d)
        except RuntimeError as e:
            logger.warning(
                f"Your configuration is not compatible with the following device: {self.accelerator}"
            )
            raise e

        # Special test for device=="cuda"
        if self.accelerator == "cuda" and not torch.cuda.is_available():
            logger.warning(
                f"You selected `cuda` accelerator, but it is not available. CPU will be used instead."
            )
