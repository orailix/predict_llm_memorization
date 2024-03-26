# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import base64
import copy
import hashlib
import json
import typing as t
from configparser import ConfigParser
from pathlib import Path

import torch
from loguru import logger

from . import paths
from .constants import *
from .hf_hub import (
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
# - [ ] Add it to configs/training.cfg ; tests/files/training.cfg ; tests/files/training_cfg.json
# - [ ] Add it to cls.from_parser
# - [ ] Add it to to_json
# - [ ] Add it to __init__
# - [ ] Add it to tests.test_train_config


SAVING_NAME = "training_cfg.json"


class TrainingCfg:
    """Represents a training configuration."""

    # ==================== UTILS ====================

    def __repr__(self) -> str:
        result = f"""TrainConfig object:
MAIN:
    - model: {self.model}
    - dataset: {self.dataset}
PREPROCESS:
    - max_len: {self.max_len}
    - label_noise: {self.label_noise}
    - data_seed: {self.data_seed}
RANDOM SPLIT:
    - split_id: {self.split_id}
    - split_prop: {self.split_prop}
    - split_test: {self.split_test}
LoRA:
    - lora_r: {self.lora_r}
    - lora_alpha: {self.lora_alpha}
    - lora_dropout: {self.lora_dropout}
DEVICES:
    - accelerator: {self.accelerator}
LOSS:
    - last_token_only: {self.last_token_only}
TRAINING_ARGS:"""

        for key in sorted(self.training_args):
            result += f"\n    - {key}: {self.training_args[key]}"

        return result

    def copy(self):
        return copy.deepcopy(self)

    def get_config_id(self) -> str:
        """Gets the config ID of a config (which is a permanent URL-safe string hash)

        The hash is computed based on the attributes of the instance that
        are NOT equal to the default value. Thus, the hash remains the same
        if some new attributes are added to the TrainingCfg class.

        The attributes that are not taken into account for this ID are:
        - self.accelerator
        - the ones in grokking_llm.utils.constants.TRAINING_ARGS_EXCLUDED_FROM_CONFIG_ID.
        They are excluded because they do not change the training dynamic of the model,
        but enable flexibility for manual adaptation (e.g. add more epochs, modify eval
        batch size, ask the pipeline to resume from existing checkpoint, etc.)
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

        if self.data_seed != TRAIN_CFG_DEFAULT_DATA_SEED:
            description += f"label_noise={self.data_seed};"

        if self.split_id != TRAIN_CFG_DEFAULT_SPLIT_ID:
            description += f"split_id={self.split_id};"

        if self.split_prop != TRAIN_CFG_DEFAULT_SPLIT_PROP:
            description += f"split_prop={self.split_prop};"

        if self.split_test != TRAIN_CFG_DEFAULT_SPLIT_TEST:
            description += f"split_test={self.split_test};"

        if self.lora_r != TRAIN_CFG_DEFAULT_LORA_R:
            description += f"lora_r={self.lora_r};"

        if self.lora_alpha != TRAIN_CFG_DEFAULT_LORA_ALPHA:
            description += f"lora_alpha={self.lora_alpha};"

        if self.lora_dropout != TRAIN_CFG_DEFAULT_LORA_DROPOUT:
            description += f"lora_dropout={self.lora_dropout};"

        if self.last_token_only != TRAIN_CFG_DEFAULT_LAST_TOKEN_ONLY:
            description += f"last_token_only={self.last_token_only};"

        for key in sorted(self.training_args):
            if key in TRAINING_ARGS_EXCLUDED_FROM_CONFIG_ID:
                continue
            description += f"{key}={self.training_args[key]};"

        # Persistent, replicable and URL-free hash
        return base64.urlsafe_b64encode(
            hashlib.md5(description.encode("utf-8")).digest()
        ).decode()[:22]

    def __eq__(self, __value: object) -> bool:
        """Two instances are equals if all attributes are equals."""

        if not isinstance(__value, TrainingCfg):
            return False

        if not self.get_config_id() == __value.get_config_id():
            return False

        for key in TRAINING_ARGS_EXCLUDED_FROM_CONFIG_ID:

            if key in self.training_args:
                if key not in __value.training_args:
                    return False

                if self.training_args[key] != __value.training_args[key]:
                    return False

            if key not in self.training_args:
                if key in __value.training_args:
                    return False

        return True

    def __hash__(self) -> int:
        to_hash = self.get_config_id() + f";accelerator={self.accelerator}"
        for key, value in self.training_args.items():
            to_hash += f";{key}={value}"

        return hash(to_hash)

    # ==================== OUTPUT DIR ====================

    def get_output_dir(self) -> Path:
        """The output dir of a training config.

        Creates the dir and saves the config if it does not exist."""

        result = paths.individual_outputs / self.get_config_id()
        result.mkdir(parents=True, exist_ok=True)
        self.to_json(result / SAVING_NAME)

        return result

    def get_available_checkpoints(self) -> t.List[int]:
        """Gets the list of available checkpoints for a given config.

        Args:
            - cfg: The training config object.

        Returns:
            - List[int]: a sorted list of available checkpoints."""

        output_dir = self.get_output_dir()
        result = []

        for item in output_dir.iterdir():
            if not item.is_dir():
                continue

            dir_name = item.name
            if dir_name[: len("checkpoint-")] != "checkpoint-":
                continue

            try:
                result.append(int(dir_name[len("checkpoint-") :]))
            except ValueError:
                continue

        return sorted(result)

    @property
    def latest_checkpoint(self) -> int:
        return sorted(self.get_available_checkpoints())[-1]

    def get_resume_from_checkpoint_status(self) -> t.Union[bool, str]:
        """Gets the value to pass to trainer.train(resume_from_checkpoint=...)

        - self.training_args["resume_from_checkpoint"] should exists as it is in default values
        - If it is True, returns True if there exist checkpoints in the output dir
        - If it is False, returns False
        - If it is a non-boolean value, returns it (it should be a path to a checkpoint)
        """

        if "resume_from_checkpoint" not in self.training_args:
            raise ValueError(
                "self.training_args['resume_from_checkpoint'] should exists as it is in default values"
            )

        if isinstance(self.training_args["resume_from_checkpoint"], bool):
            if not self.training_args["resume_from_checkpoint"]:
                result = False
            else:
                result = len(self.get_available_checkpoints()) >= 1
        else:
            result = str(self.training_args["resume_from_checkpoint"])

        # Output
        logger.debug(
            f"Using resume_from_checkpoint={result} for config {self.get_config_id()}"
        )
        return result

    # ==================== CFG BUILD ====================

    @classmethod
    def from_file(cls, path: t.Union[str, Path]):
        """Builds a config object from a file.

        Args:
            - path: The path to the file, either .cfg or .json"""

        # PARSING CONFIG
        if type(path) == str:
            path = Path(path)

        if path.suffix == ".cfg":
            return cls.from_cfg(path)
        elif path.suffix == ".json":
            return cls.from_json(path)
        else:
            raise ValueError(f"Expected a .json or .cfg file: {path}")

    @classmethod
    def from_cfg(cls, path: t.Union[str, Path]):
        """Builds a config object from a config file.

        Args:
            - path: The path to the config file"""

        # File exists ?
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")

        # PARSING CONFIG
        parser = ConfigParser()
        parser.read(path)

        return cls.from_parser(parser)

    @classmethod
    def from_json(cls, path: t.Union[str, Path]):
        """Builds a config object from a json file.

        Args:
            - path: The path to the json file"""

        # File exists ?
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")

        # PARSING JSON
        with open(path, "r") as f:
            json_content = json.load(f)

        return cls.from_parser(json_content)

    @classmethod
    def autoconfig(cls, name: t.Union[str, Path]):
        """Automatif build of a config object.

        The method will sequentially try these option and return the result
        of the first one to succeede. If no option work, raise ValueError.

        1. If `name` is the path to a valid `.cfg` or `.json` file, builds
        the result from it.
        2. If `grokking_llm.utils.paths.configs / name` is a valid file,
        builds the result from it.
        3. If `grokking_llm.utils.paths.configs / f'{name}.cfg'` is a valid
        file, builds the result from it.
        4. If `grokking_llm.utils.paths.configs / f'{name}.json'` is a valid
        file, builds the result from it.
        5. If `name` is a at least 4 character length, will look at every dir
        in `grokking_llm.paths.individual_outputs`, and build the output from the config
        of the first dir for which `name` is a prefix of it's config_id

        Args:
            - name: A string describing the config to build.
        """

        if name is None:
            raise TypeError("Cannot build a TrainingCfg from Nonetype.")

        if isinstance(name, Path):
            logger.info(f"Autoconfig `name`: {name} is a valid path, building from it.")
            return cls.from_file(name)

        if Path(name).is_file():
            logger.info(f"Autoconfig `name`: {name} is a valid path, building from it.")
            return cls.from_file(name)

        if (paths.configs / name).exists():
            logger.info(
                f"Autoconfig `name`: {name} is a config in `utils.paths.configs`, building from it."
            )
            return cls.from_file(paths.configs / f"{name}")

        if (paths.configs / f"{name}.cfg").exists():
            logger.info(
                f"Autoconfig `name`: {name}.cfg is a config in `utils.paths.configs`, building from it."
            )
            return cls.from_file(paths.configs / f"{name}.cfg")

        if (paths.configs / f"{name}.json").exists():
            logger.info(
                f"Autoconfig `name`: {name}.json is a config in `utils.paths.configs`, building from it."
            )
            return cls.from_file(paths.configs / f"{name}.json")

        if len(name) >= 4:

            for child in paths.individual_outputs.iterdir():
                if not child.is_dir():
                    continue

                if child.name[: len(name)] == name:
                    logger.info(
                        f"Autoconfig `name`: {name} is the prefix of confi_id {child.name} in `paths.individual_outputs`, building from it."
                    )
                    return cls.from_file(child / "training_cfg.json")

        raise ValueError(f"Autoconfig `name`: {name} cannot be matched to any config.")

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
        if "data_seed" not in parser["preprocess"]:
            raise ValueError(
                "Section 'preprocess' of your config should contain a 'data_seed' entry."
            )

        max_len = parser["preprocess"]["max_len"]
        label_noise = parser["preprocess"]["label_noise"]
        data_seed = parser["preprocess"]["data_seed"]

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
        if "split_test" not in parser["random_split"]:
            raise ValueError(
                "Section 'random_split' of your config should contain a 'split_test' entry."
            )

        split_id = parser["random_split"]["split_id"]
        split_prop = parser["random_split"]["split_prop"]
        split_test = parser["random_split"]["split_test"]

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

        # LOSS

        if "loss" not in parser:
            raise ValueError("Your config should contain a 'loss' section.")
        if "last_token_only" not in parser["loss"]:
            raise ValueError(
                "Section 'loss' of your config should contain a 'last_token_only' entry."
            )

        last_token_only = parser["loss"]["last_token_only"]

        # TRAINING ARGS

        if "training_args" in parser:
            training_args = parser["training_args"]
        else:
            training_args = {}
            logger.info(
                "No 'training_args' section found in your training config, using default values."
            )

        # OUTPUT

        return cls(
            model=model,
            dataset=dataset,
            max_len=max_len,
            label_noise=label_noise,
            data_seed=data_seed,
            split_id=split_id,
            split_prop=split_prop,
            split_test=split_test,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            accelerator=accelerator,
            last_token_only=last_token_only,
            training_args=training_args,
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
                "data_seed": self.data_seed,
            },
            "random_split": {
                "split_id": self.split_id,
                "split_prop": self.split_prop,
                "split_test": self.split_test,
            },
            "lora": {
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "lora_dropout": self.lora_dropout,
            },
            "devices": {
                "accelerator": self.accelerator,
            },
            "loss": {"last_token_only": self.last_token_only},
            "training_args": self.training_args,
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
        data_seed: int = TRAIN_CFG_DEFAULT_DATA_SEED,
        split_id: int = TRAIN_CFG_DEFAULT_SPLIT_ID,
        split_prop: float = TRAIN_CFG_DEFAULT_SPLIT_PROP,
        split_test: t.Union[str, bool] = TRAIN_CFG_DEFAULT_SPLIT_TEST,
        lora_r: int = TRAIN_CFG_DEFAULT_LORA_R,
        lora_alpha: float = TRAIN_CFG_DEFAULT_LORA_ALPHA,
        lora_dropout: float = TRAIN_CFG_DEFAULT_LORA_DROPOUT,
        accelerator: str = TRAIN_CFG_DEFAULT_ACCELERATOR,
        last_token_only: bool = TRAIN_CFG_DEFAULT_LAST_TOKEN_ONLY,
        training_args: dict = TRAIN_CFG_DEFAULT_TRAINING_ARGS,
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

        try:
            self.data_seed = int(data_seed)
        except ValueError:
            raise ValueError(f"`data_seed`={data_seed} should be an int.")

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

        if isinstance(split_test, bool):
            self.split_test = split_test
        elif split_test == "true":
            self.split_test = True
        elif split_test == "false":
            self.split_test = False
        else:
            raise ValueError(
                f"`split_test`={split_test} but it should be 'true' or 'false'."
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
        except (RuntimeError, AssertionError) as e:
            logger.warning(
                f"Your configuration is not compatible with the following device: {self.accelerator}. This is likely to cause errors in yoru pipeline."
            )

        # Special test for device=="cuda"
        if self.accelerator == "cuda" and not torch.cuda.is_available():
            logger.warning(
                f"You selected `cuda` accelerator, but it is not available. CPU will be used instead."
            )

        # LOSS

        if isinstance(last_token_only, bool):
            self.last_token_only = last_token_only
        elif isinstance(last_token_only, str):
            if last_token_only == "false":
                self.last_token_only = False
            elif last_token_only == "true":
                self.last_token_only = True
            else:
                raise ValueError(
                    f"Invalid value for boolean casting of `last_token_only` = {last_token_only} [type={type(last_token_only)}]"
                )
        else:
            raise ValueError(
                f"Invalid value for boolean casting of `last_token_only` = {last_token_only} [type={type(last_token_only)}]"
            )

        # TRAINING ARGS

        self.training_args = TRAIN_CFG_DEFAULT_TRAINING_ARGS.copy()

        for key, value in training_args.items():
            if not isinstance(key, str):
                raise ValueError(f"training_args.{key} key is not a string.")

            if isinstance(value, str):
                try:
                    value = float(value)
                    if int(value) == value:
                        value = int(value)
                except ValueError:
                    pass

                if value == "false":
                    value = False

                if value == "true":
                    value = True

            self.training_args[key] = value
