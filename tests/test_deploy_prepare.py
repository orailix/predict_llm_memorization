# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import shutil

import pytest

from grokking_llm.deploy import DeploymentCfg
from grokking_llm.deploy.prepare_deploy import (
    product_combinations,
    run_prepare_deploy,
    zip_combinations,
)
from grokking_llm.training import TrainingCfg


def test_product():
    deployment_cfg = DeploymentCfg.autoconfig("deployment_2")
    parsed_sections_list = deployment_cfg.get_parsed_section_list()
    base_config = deployment_cfg.cfg["general"]["base_config"]
    training_cfg = TrainingCfg.autoconfig(base_config)

    # Product
    possible_training_cfg = product_combinations(training_cfg, parsed_sections_list)

    # Sanity check
    assert len(possible_training_cfg) == 11**3

    # Count values
    count_label_noise = collections.defaultdict(int)
    count_lora_r = collections.defaultdict(int)
    count_weight_decay = collections.defaultdict(int)
    for item in possible_training_cfg:
        assert isinstance(item, TrainingCfg)
        count_label_noise[item.label_noise] += 1
        count_lora_r[item.lora_r] += 1
        count_weight_decay[item.training_args["weight_decay"]] += 1

    assert len(count_lora_r) == len(count_weight_decay) == len(count_label_noise) == 11
    for _, val in count_label_noise.items():
        assert val == 11**2

    for _, val in count_weight_decay.items():
        assert val == 11**2

    for _, val in count_lora_r.items():
        assert val == 11**2


def test_zip():
    deployment_cfg = DeploymentCfg.autoconfig("deployment_2")
    parsed_sections_list = deployment_cfg.get_parsed_section_list()
    base_config = deployment_cfg.cfg["general"]["base_config"]
    training_cfg = TrainingCfg.autoconfig(base_config)

    # Product
    possible_training_cfg = zip_combinations(training_cfg, parsed_sections_list)

    # Sanity check
    assert len(possible_training_cfg) == 11

    # Count values
    for idx, (noise, r, wd) in enumerate(
        zip([k * 0.1 for k in range(11)], [11 - k for k in range(11)], list(range(11)))
    ):
        assert isinstance(possible_training_cfg[idx], TrainingCfg)
        assert possible_training_cfg[idx].label_noise == pytest.approx(noise)
        assert possible_training_cfg[idx].lora_r == pytest.approx(r)
        assert possible_training_cfg[idx].training_args[
            "weight_decay"
        ] == pytest.approx(wd)


def test_prepare_product():
    # Cleaning
    shutil.rmtree(DeploymentCfg.autoconfig("deployment_2").export_dir)

    # Running
    run_prepare_deploy("deployment_2")

    # Deployment cfg
    deployment_cfg = DeploymentCfg.autoconfig("deployment_2")

    # Checking stack_all
    stack_content = []
    while not deployment_cfg.stack_all.empty():
        stack_content.append(deployment_cfg.stack_all.pop())

    assert len(stack_content) == 11**3

    # Counting exports
    count_export = 0
    checked_import = False
    for child in deployment_cfg.export_dir.iterdir():
        if (
            child.is_file()
            and child.suffix == ".json"
            and "training_cfg_" in child.stem
        ):
            count_export += 1

            assert str(child) in stack_content

            if not checked_import:
                TrainingCfg.from_file(child)
                checked_import = True

    assert count_export == 11**3

    # Cleaning
    shutil.rmtree(DeploymentCfg.autoconfig("deployment_2").export_dir)


def test_prepare_zip():
    # Cleaning
    shutil.rmtree(DeploymentCfg.autoconfig("deployment_3").export_dir)

    # Running
    run_prepare_deploy("deployment_3")

    # Deployment cfg
    deployment_cfg = DeploymentCfg.autoconfig("deployment_3")

    # Checking stack_all
    stack_content = []
    while not deployment_cfg.stack_all.empty():
        stack_content.append(deployment_cfg.stack_all.pop())

    assert len(stack_content) == 11

    # Counting exports
    count_export = 0
    checked_import = False
    for child in deployment_cfg.export_dir.iterdir():
        if (
            child.is_file()
            and child.suffix == ".json"
            and "training_cfg_" in child.stem
        ):
            count_export += 1

            assert str(child) in stack_content

            if not checked_import:
                TrainingCfg.from_file(child)
                checked_import = True

    assert count_export == 11

    # Cleaning
    shutil.rmtree(DeploymentCfg.autoconfig("deployment_3").export_dir)


def test_prepare_unfeasible_list():
    with pytest.raises(ValueError):
        run_prepare_deploy("deployment_4")

    # Cleaning
    shutil.rmtree(DeploymentCfg.autoconfig("deployment_4").export_dir)
