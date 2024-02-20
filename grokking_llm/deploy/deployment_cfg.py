# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import base64
import configparser
import dataclasses
import hashlib
import typing as t
from pathlib import Path

from loguru import logger

from ..utils import paths
from .utils import DiskStack, ParsedSection


class DeploymentCfg:
    """Class used to represent a deployment configuration."""

    # ==================== BUILDS ====================

    def __init__(self, parsed_config: configparser.ConfigParser):
        # Config
        self.cfg = parsed_config

        # ID and export dir
        self.id = self.get_deployment_id()
        self.export_dir = paths.deployment_configs / self.id
        self.export_dir.mkdir(exist_ok=True)
        with (self.export_dir / "deployment.cfg").open("w") as f:
            self.cfg.write(f)

        # Stacks
        self.stack_all = DiskStack(self.export_dir / "stack_all")
        self.stack_todo_gpu = DiskStack(self.export_dir / "stack_todo_gpu")
        self.stack_todo_cpu = DiskStack(self.export_dir / "stack_todo_cpu")
        self.stack_done_gpu = DiskStack(self.export_dir / "stack_done_gpu")
        self.stack_done_cpu = DiskStack(self.export_dir / "stack_done_cpu")

    def get_deployment_id(self):
        return base64.urlsafe_b64encode(
            hashlib.md5(self.__repr__().encode("utf-8")).digest()
        ).decode()[:22]

    def __repr__(self) -> str:
        desc = ""
        for section_name in sorted(self.cfg.sections()):
            desc += f"[{section_name}]\n"
            for key in sorted(self.cfg[section_name]):
                value = self.cfg[section_name][key]
                value = convert_str_to_int_or_float(value)
                desc += f"{key} = {value}\n"

        return desc

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, DeploymentCfg):
            return False

        return self.id == __value.id

    @classmethod
    def from_file(cls, path: t.Union[str, Path]):
        """Buils an deployment config from a file."""
        cfg = configparser.ConfigParser()
        cfg.read(path)
        return cls(cfg)

    @classmethod
    def autoconfig(cls, name: t.Union[str, Path]):
        """Returns an deployment config.

        - If `name` is an instance of pathlib.Path, builds from it
        - If it is a string referring to a valid file, builds from it
        - If it is a string correspondinf to a file in
            utils.paths.configs, builds from it
        - If it is the prefix (at least four characters) of the hash of a
            deployment config in utils.paths.deployment_configs, builds from it

        Args:
            name: The name to autoconfig

        Returns:
            pathlib.Path: The path to a valid config file
        """

        if isinstance(name, Path):
            logger.info(f"Autoconfig `name`: {name} is a valid path, building from it.")
            return cls.from_file(name)

        if Path(name).exists():
            logger.info(f"Autoconfig `name`: {name} is a valid path, building from it.")
            return cls.from_file(Path(name))

        if (paths.configs / name).exists():
            f"Autoconfig `name`: {name} is a config in `utils.paths.configs`, building from it."
            return cls.from_file(paths.configs / name)

        if (paths.configs / f"{name}.cfg").exists():
            f"Autoconfig `name`: {name}.cfg is a config in `utils.paths.configs`, building from it."
            return cls.from_file(paths.configs / f"{name}.cfg")

        elif len(name) >= 4:

            for child in paths.deployment_configs.iterdir():
                if not child.is_dir():
                    continue

                if child.name[: len(name)] == name:
                    return cls.from_file(child / "deployment.cfg")

        raise ValueError(f"Unfound deployment config: {name}")

    # ==================== PARSING ====================

    def get_parsed_section_list(self) -> t.List[ParsedSection]:
        """Parses the section of the deployment config.

        The sections are parsed as follows:
        - The name of the section will be used as a name for the ParsedSection object
        - if section["mode"] == "range", the section is considered as a range
            and is processed by `process_range_section`
        - if section["mode"] == "list", the section is considered as a list
            ans is processed by `process_list_section`

        Returns:
            A list of ParsedSection
        """
        # Init result
        result = []

        # Iterating over s
        for section in sorted(self.cfg.sections()):

            # Fetching the section object
            if section == "general":
                continue
            section_object = self.cfg[section]

            # Getting list of parsed sections
            if section_object["mode"] == "range":
                result.append(process_range_section(section_object))

            if section_object["mode"] == "list":
                result.append(process_list_section(section_object))

        # Output
        return result


def process_range_section(sec: configparser.SectionProxy) -> ParsedSection:
    """Parses a section of a deployment config in "RANGE" mode.

    The name of the section will be the name of the ParsedSection.
    The section should have three attributes:
        `mode`: It should be "range", or this function should not process this section.
        `start`: The beginning of the range of values (included).
        `step`: The step for defining the values of the range.
        `stop`: The stop value of the range.

    The values included are:
        {start + k*step | start + k*step <= stop} if step > 0
        {start + k*step | start + k*step >= stop} if step < 0

    Args:
        sec: The section

    Returns:
        ParsedSection: The parsed section."""

    # Attributes of the section
    start = float(sec["start"])
    stop = float(sec["stop"])
    step = float(sec["step"])

    # Init result
    result = ParsedSection(name=sec.name, values=[])

    # Step >0 or <0 ?
    if step > 0:
        end_condition = lambda val: val > stop
    elif step < 0:
        end_condition = lambda val: val < stop
    else:
        raise ValueError(f"step==0 in section {sec.name}")

    # Appending values
    while not end_condition(start):
        result.values.append(start)
        start += step
        start = round(start, 12)

    # Output
    return result


def process_list_section(sec: configparser.SectionProxy) -> ParsedSection:
    """Parses a section of a deployment config in "LIST" mode.

    The name of the section will be the name of the ParsedSection.
    The section should have several attributes:
        `mode`: It should be "list", or this function should not process this section.
        `val_1`: The first value
        `val_2`: The second value
        etc.

    Args:
        sec: The section

    Returns:
        ParsedSection: The parsed section."""

    # Init the result
    result = ParsedSection(name=sec.name, values=[])

    # Sorting the keys
    keys = [item for item in sec.keys() if "val_" in item]
    keys = sorted(keys, key=lambda k: int(k[len("val_") :]))

    # Processing the keys
    for key in keys:
        value = convert_str_to_int_or_float(sec[key])
        result.values.append(value)

    # Output
    return result


def convert_str_to_int_or_float(s: str):
    """If the string represents a float, converts it. It it represents an int, converts it.
    Else, does nothing."""
    try:
        s = float(s)
        if int(s) == s:
            s = int(s)
    except ValueError:
        pass

    return s
