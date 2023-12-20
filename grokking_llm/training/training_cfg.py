"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import base64
import copy
import functools
import hashlib
import json
import typing as t
from configparser import ConfigParser
from pathlib import Path

import torch
from loguru import logger

from ..utils import paths
from ..utils.constants import *
from ..utils.hf_hub import (
    DS_ARC,
    DS_ETHICS,
    DS_MMLU,
    MOD_DUMMY_LLAMA,
    MOD_LLAMA_7B,
    MOD_MISTRAL_7B,
)

# Check-list for adding a new attribute to the class:
# - [ ] Add it to __repr__
# - [ ] Add a default vaue in ..utils.constants
# - [ ] Add it to get_config_id
# - [ ] Add it to configs/training.cfg ; tests/files/training.cfg ; tests/files/ training_cfg.json
# - [ ] Add it to cls.from_parser
# - [ ] Add it to to_json
# - [ ] Add it to __init__
# - [ ] Add it to tests.test_train_config


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
TRAINING_STATUS:
    - epoch_done: {self.epochs_done}
    - epoch_total: {self.epochs_total}
"""

    def copy(self):
        return copy.copy(self)

    def get_config_id(self) -> str:
        """Gets the config ID of a config (which is a permanent URL-safe string hash)

        The hash is computed based on the attributes of the instance that
        are NOT equal to the default value. Thus, the hash remains the same
        if some new attributes are added to the TrainingCfg class.

        The two only attributes that are not taken into account for this ID are:
            - self.epoch_done: we want the ID to remain the same through training
            - self.epoch_total: we wan the ID to be the same if two cfg has different
            total number of epoch.
        """

        description = ""

        if self.model != TRAIN_CFG_DEFAULT_MODEL:
            description += f"model={self.model};"

        if self.dataset != TRAIN_CFG_DEFAULT_DATASET:
            description += f"dataset={self.dataset};"

        if self.max_len != TRAIN_CFG_DEFAULT_MAX_LEN:
            description += f"max_len={self.max_len};"

        if self.label_noise != TRAIN_CFG_DEFAULT_LABEL_NOISE:
            description += f"label_noise={self.label_noise};"

        if self.split_id != TRAIN_CFG_DEFAULT_SPLIT_ID:
            description += f"split_id={self.split_id};"

        if self.split_prop != TRAIN_CFG_DEFAULT_SPLIT_PROP:
            description += f"split_prop={self.split_prop};"

        if self.lora_r != TRAIN_CFG_DEFAULT_LORA_R:
            description += f"lora_r={self.lora_r};"

        if self.lora_alpha != TRAIN_CFG_DEFAULT_LORA_ALPHA:
            description += f"lora_alpha={self.lora_alpha};"

        if self.lora_dropout != TRAIN_CFG_DEFAULT_LORA_DROPOUT:
            description += f"lora_dropout={self.lora_dropout};"

        if self.accelerator != TRAIN_CFG_DEFAULT_ACCELERATOR:
            description += f"accelerator={self.accelerator};"

        # Persistent, replicable and URL-free hash
        return base64.urlsafe_b64encode(
            hashlib.md5(description.encode("utf-8")).digest()
        ).decode()[:22]

    @property
    @functools.lru_cache
    def output_dir(self) -> Path:
        """The output dir of a training config."""
        return paths.output / self.get_config_id()

    def sync_with_output_dir(self) -> bool:
        """Sync with the output dir of a training config and sync with it.

        - If the output dir does not exist:
            - Creates it
            - Export the training config as output_dir / "training_cfg.json"
        - If it exists:
            - Updates the number of epoch done as the maximum of itself and value on disk
            - Idem for total number of epochs

        Return:
            bool: True if the object has been updated with an existing output dir, else False.
        """

        logger.info(f"Synchronizing config with output dir: {self.output_dir}")
        cfg_export_path = self.output_dir / "training_cfg.json"

        if not self.output_dir.is_dir() or not cfg_export_path.exists():
            logger.debug(f"Config not found on disk.")
            logger.debug(
                f"Creating output dir and exporting config at: {cfg_export_path}"
            )
            self.output_dir.mkdir(parents=True)
            self.to_json(cfg_export_path)

            return False

        else:
            logger.debug(f"Found existing config at: {self.output_dir}")
            disk_config = TrainingCfg.from_json(cfg_export_path)
            if disk_config.get_config_id() != self.get_config_id():
                raise RuntimeError(
                    f"Found a training config file that does not correspond to its config ID: {cfg_export_path}"
                )

            epochs_done_before = self.epochs_done
            epochs_total_before = self.epochs_total
            self.epochs_done = max(self.epochs_done, disk_config.epochs_done)
            logger.debug(
                f"Updating self.epoch_done {epochs_done_before} => {self.epochs_done}"
            )
            self.epochs_total = max(self.epochs_total, disk_config.epochs_total)
            logger.debug(
                f"Updating self.epoch_total {epochs_total_before} => {self.epochs_total}"
            )

            # Exporting
            logger.debug(f"Exporting training config to {cfg_export_path}")
            self.to_json(cfg_export_path)

            return True

    def __eq__(self, __value: object) -> bool:
        """Two instances are equals if all attributes are equals."""

        if not isinstance(__value, TrainingCfg):
            return False
        return (
            self.get_config_id() == __value.get_config_id()
            and self.epochs_done == __value.epochs_done
            and self.epochs_total == __value.epochs_total
        )

    def __hash__(self) -> int:
        return hash(
            self.get_config_id()
            + f";epochs_done={self.epochs_done};epochs_total={self.epochs_total}"
        )

    # ==================== CFG BUILD ====================

    @classmethod
    def from_cfg(cls, path: t.Union[str, Path]):
        """Builds a config object from a config file.

        Args:
            - path: The path to the config file"""

        # PARSING CONFIG
        parser = ConfigParser()
        parser.read(path)

        return cls.from_parser(parser)

    @classmethod
    def from_json(cls, path: t.Union[str, Path]):
        """Builds a config object from a json file.

        Args:
            - path: The path to the json file"""

        # PARSING JSON
        with open(path, "r") as f:
            json_content = json.load(f)

        return cls.from_parser(json_content)

    # ==================== PARSER ====================

    @classmethod
    def from_parser(cls, parser: t.Union[ConfigParser, dict]):
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

        # EPOCHS CONFIG

        if "epochs" not in parser:
            raise ValueError("Your config should contain a 'epochs' section.")
        if "epochs_done" not in parser["epochs"]:
            raise ValueError(
                "Section 'epochs' of your config should contain a 'epochs_done' entry."
            )
        if "epochs_total" not in parser["epochs"]:
            raise ValueError(
                "Section 'epochs' of your config should contain a 'epochs_total' entry."
            )

        epochs_done = parser["epochs"]["epochs_done"]
        epochs_total = parser["epochs"]["epochs_total"]

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
            epochs_done=epochs_done,
            epochs_total=epochs_total,
        )

    # ==================== SAVING ====================

    def to_json(self, path: t.Union[str, Path]) -> None:
        """Saves the config as JSON.

        Args:
            path: The paths to save the config.
        """

        export = {
            "main": {
                "model": self.model,
                "dataset": self.dataset,
            },
            "preprocess": {
                "max_len": self.max_len,
                "label_noise": self.label_noise,
            },
            "random_split": {
                "split_id": self.split_id,
                "split_prop": self.split_prop,
            },
            "lora": {
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            },
            "devices": {
                "accelerator": self.accelerator,
            },
            "epochs": {
                "epochs_done": self.epochs_done,
                "epochs_total": self.epochs_total,
            },
        }

        with open(path, "w") as f:
            json.dump(export, f, sort_keys=True, indent=2)

    # ==================== CLASS BUILD ====================

    def __init__(
        self,
        *,
        model: str = TRAIN_CFG_DEFAULT_MODEL,
        dataset: str = TRAIN_CFG_DEFAULT_DATASET,
        max_len: int = TRAIN_CFG_DEFAULT_MAX_LEN,
        label_noise: float = TRAIN_CFG_DEFAULT_LABEL_NOISE,
        split_id: int = TRAIN_CFG_DEFAULT_SPLIT_ID,
        split_prop: float = TRAIN_CFG_DEFAULT_SPLIT_PROP,
        lora_r: int = TRAIN_CFG_DEFAULT_LORA_R,
        lora_alpha: float = TRAIN_CFG_DEFAULT_LORA_ALPHA,
        lora_dropout: float = TRAIN_CFG_DEFAULT_LORA_DROPOUT,
        accelerator: str = TRAIN_CFG_DEFAULT_ACCELERATOR,
        epochs_done: int = TRAIN_CFG_DEFAULT_EPOCH_DONE,
        epochs_total: int = TRAIN_CFG_DEFAULT_EPOCH_TOTAL,
    ):
        """Safely builds a config object from kwargs."""

        # MAIN CONFIG

        if model in [MOD_MISTRAL_7B, MOD_LLAMA_7B, MOD_DUMMY_LLAMA]:
            self.model = model
        elif model.lower() == TRAIN_CFG_MISTRAL:
            self.model = MOD_MISTRAL_7B
        elif model.lower() == TRAIN_CFG_LLAMA:
            self.model = MOD_LLAMA_7B
        elif model.lower() == TRAIN_CFG_DUMMY_LLAMA:
            logger.info(f"Using dummy Llama model for testing.: {MOD_DUMMY_LLAMA}")
            logger.info("DO NOT USE FOR NON-TESTING PURPOSE")
            self.model = MOD_DUMMY_LLAMA
        else:
            raise ValueError(
                f"`model`={model} should be in {[TRAIN_CFG_MISTRAL, TRAIN_CFG_LLAMA, TRAIN_CFG_DUMMY_LLAMA, MOD_MISTRAL_7B, MOD_LLAMA_7B, MOD_DUMMY_LLAMA]}."
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

        # EPOCHS CONFIG

        try:
            self.epochs_done = int(epochs_done)
            if self.epochs_done < 0:
                raise ValueError()
        except ValueError:
            raise ValueError(f"`epochs_done`={epochs_done} should be a positive int.")

        try:
            self.epochs_total = int(epochs_total)
            if self.epochs_total < self.epochs_done:
                raise ValueError()
        except ValueError:
            raise ValueError(
                f"`epochs_total`={epochs_total} should be an int greater than `epoch_done`={self.epochs_done}"
            )
