# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import configparser
import shutil
from pathlib import Path

import numpy as np
import pytest

from grokking_llm.utils import DeploymentCfg, ParsedSection, paths

# Files
deployment_cfg_path = paths.configs / "deployment_0.cfg"


def test_build():
    cfg = configparser.ConfigParser()
    cfg.read(deployment_cfg_path)

    deployment_cfg = DeploymentCfg(cfg)
    assert isinstance(deployment_cfg.id, str)
    assert isinstance(deployment_cfg.export_dir, Path)
    assert (deployment_cfg.export_dir / "deployment.cfg").is_file()
    assert (
        DeploymentCfg.from_file(deployment_cfg.export_dir / "deployment.cfg")
        == deployment_cfg
    )


def test_deployment_id():
    assert DeploymentCfg.from_file(
        paths.configs / "deployment_0.cfg"
    ) == DeploymentCfg.from_file(paths.configs / "deployment_1.cfg")
    assert DeploymentCfg.from_file(
        paths.configs / "deployment_0.cfg"
    ) != DeploymentCfg.from_file(paths.configs / "deployment_2.cfg")


def test_autoconfig():

    # Cleaning
    for child in paths.deployment_outputs.iterdir():
        if child.is_dir():
            shutil.rmtree(child)

    # Autoconfig -- path
    assert DeploymentCfg.autoconfig(deployment_cfg_path) == DeploymentCfg.from_file(
        deployment_cfg_path
    )
    assert DeploymentCfg.autoconfig(
        str(deployment_cfg_path)
    ) == DeploymentCfg.from_file(deployment_cfg_path)
    assert DeploymentCfg.autoconfig("deployment_0") == DeploymentCfg.from_file(
        deployment_cfg_path
    )
    assert DeploymentCfg.autoconfig("deployment_0.cfg") == DeploymentCfg.from_file(
        deployment_cfg_path
    )

    # Creating output dir for a config
    DeploymentCfg.autoconfig("deployment_0")
    DeploymentCfg.autoconfig("deployment_1")
    cfg = DeploymentCfg.autoconfig("deployment_2")

    assert DeploymentCfg.autoconfig(cfg.id[:4]) == cfg
    assert DeploymentCfg.autoconfig(cfg.id[:6]) == cfg
    assert DeploymentCfg.autoconfig(cfg.id[:8]) == cfg
    assert DeploymentCfg.autoconfig(cfg.id) == cfg

    with pytest.raises(ValueError):
        DeploymentCfg.autoconfig(cfg.id[:3])

    with pytest.raises(ValueError):
        DeploymentCfg.autoconfig(str(np.random.randint(1e6)))

    # Cleaning
    for child in paths.deployment_outputs.iterdir():
        if child.is_dir():
            shutil.rmtree(child)


def test_parse_sections():
    # Build
    parsed_sections = DeploymentCfg.autoconfig("deployment_2").get_parsed_section_list()

    # Sanity checks
    assert len(parsed_sections) == 3
    assert isinstance(parsed_sections[0], ParsedSection)
    assert isinstance(parsed_sections[1], ParsedSection)
    assert isinstance(parsed_sections[2], ParsedSection)

    # Checks on section 0 -- range from 0 to 1
    assert parsed_sections[0].name == "label_noise"
    for val_parsed, val_groundtruth in zip(
        parsed_sections[0].values, [0] + [k * 0.1 for k in range(1, 10)] + [1]
    ):
        assert val_parsed == pytest.approx(val_groundtruth)
        assert type(val_parsed) == type(val_groundtruth)

    # Checks on section 1 -- range from 11 to 1
    assert parsed_sections[1].name == "lora_r"
    for val_parsed, val_groundtruth in zip(
        parsed_sections[1].values, [11 - k for k in range(11)]
    ):
        assert val_parsed == pytest.approx(val_groundtruth)
        assert type(val_parsed) == type(val_groundtruth)

    # Checks on section 2 -- list from 0 to 10
    assert parsed_sections[2].name == "training_args.weight_decay"
    for val_parsed, val_groundtruth in zip(parsed_sections[2].values, range(11)):
        assert val_parsed == pytest.approx(val_groundtruth)
        assert type(val_parsed) == type(val_groundtruth)
