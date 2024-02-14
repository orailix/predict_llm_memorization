# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import copy
import typing as t
from pathlib import Path

from loguru import logger

from ..training import TrainingCfg
from .deployment_cfg import DeploymentCfg, ParsedSection

MAX_LEN = 1e4


def run_prepare_deploy(config: t.Union[str, Path]) -> None:
    """Initiates a deployment configuration.

    - Builds a DeploymentCfg from the `name` that is provided
    - Parses the section of the DeploymentCfg object
    - Forms the combinations of training configurations
    - Dumps all possible training cfg.
    """

    # Autoconfig
    deployment_cfg = DeploymentCfg.autoconfig(config)

    # General config
    base_config = deployment_cfg.cfg["general"]["base_config"]
    mode = deployment_cfg.cfg["general"]["mode"]
    training_cfg = TrainingCfg.autoconfig(base_config)

    # Parsing sections
    parsed_sections_list = deployment_cfg.get_parsed_section_list()

    # Forming combinations
    if mode == "product":
        possible_training_cfg = product_combinations(training_cfg, parsed_sections_list)
    if mode == "zip":
        possible_training_cfg = zip_combinations(training_cfg, parsed_sections_list)

    # Cleaning
    for child in deployment_cfg.export_dir.iterdir():
        if (
            child.is_file()
            and child.suffix == ".json"
            and "training_cfg_" in child.stem
        ):
            child.unlink()

    # Dump
    for cfg_idx, cfg in enumerate(possible_training_cfg):
        cfg.to_json(deployment_cfg.export_dir / f"training_cfg_{cfg_idx}.json")


def product_combinations(
    training_cfg: TrainingCfg, parsed_sections_list: t.List[ParsedSection]
) -> t.List[TrainingCfg]:
    """Product of parsed sections.

    Similar to the doc product of iterables, the result will be a list
    of TrainingCfg with all possible combination of values from the
    parsed sections.

    Args:
        training_cfg: The base training configuration that will be a template
        for all the combinations
        parsed_sections_list: A list of ParsedSection to form the combinations

    Returns:
        t.List[TrainingCfg]: A list of independent training configurations.
    """

    # Cf. https://docs.python.org/3/library/itertools.html#itertools.product

    # Init result
    result = [copy.deepcopy(training_cfg)]

    for parsed_section in parsed_sections_list:

        tmp = []
        for cfg in result:
            for val in parsed_section.values:
                new_config = copy.deepcopy(cfg)
                set_training_cfg_attr(new_config, parsed_section.name, val)
                tmp.append(new_config)

        result = tmp

    # Output
    return result


def zip_combinations(
    training_cfg: TrainingCfg, parsed_sections_list: t.List[ParsedSection]
):
    """Zip of parsed sections.

    Similar to the `zip` standard function, the result will be a list
    of TrainingCfg with combinations of the parsed sections.

    Args:
        training_cfg: The base training configuration that will be a template
        for all the combinations
        parsed_sections_list: A list of ParsedSection to form the combinations

    Returns:
        t.List[TrainingCfg]: A list of independent training configurations.
    """

    # Init result and expected length
    expected_length = len(parsed_sections_list[0].values)
    result = [copy.deepcopy(training_cfg) for _ in range(expected_length)]

    # Forming combinations
    for parsed_section in parsed_sections_list:
        if len(parsed_section.values) != expected_length:
            raise ValueError(
                f"Linear arrangement expected same length: {len(parsed_section.values)} != {expected_length}"
            )

        for val_idx, val in enumerate(parsed_section.values):
            set_training_cfg_attr(result[val_idx], parsed_section.name, val)

    # Output
    return result


def set_training_cfg_attr(
    training_cfg: TrainingCfg,
    attribute_name: str,
    attribute_value: t.Any,
) -> None:
    """Sets an attribute of a TrainingCfg.

    Examples of attribute name:
        - `lora_dropout`: will modify `training_cfg.lora_dropout`
        - `training_args.weight_decay`: will modify `training_cfg.training_args.weight_decay`

    Args:
        training_cfg: The training configuration to set.
        attribute_name: The name of the attribute to set.
        attribute_value: The value to set."""

    # First case: not modifying the training arguments
    if attribute_name[: len("training_args.")] != "training_args.":
        setattr(training_cfg, attribute_name, attribute_value)

    # Second case: modifying the trainign arguments
    else:
        key = attribute_name[len("training_args.") :]
        training_cfg.training_args[key] = attribute_value
