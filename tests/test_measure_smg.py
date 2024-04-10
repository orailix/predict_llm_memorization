# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import shutil
import typing as t

import pytest

from grokking_llm.measures_stat import StaticMetricsGroup
from grokking_llm.utils import DeploymentCfg, paths

deployment_cfg_path = paths.configs / "deployment_3.cfg"


class DummyChildClass(StaticMetricsGroup):
    @property
    def metrics_group_name(self) -> str:
        return "dummy_child_class"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["metric_0", "metric_1", "metric_2"]

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:
        return [checkpoint + idx + 1 for idx in range(len(self.metrics_names))]


def test_instantiation():

    # Setup
    deployment_cfg = DeploymentCfg.autoconfig(deployment_cfg_path)

    # Test - Instantiation directly from the ABC should raise TypeError
    with pytest.raises(TypeError):
        metrics_group = StaticMetricsGroup(deployment_cfg)

    # Test - Instantiation fro child class + sanity checks
    metrics_group = DummyChildClass(deployment_cfg)
    assert metrics_group.deployment_cfg == deployment_cfg
    assert metrics_group.deployment_cfg.metrics_dir.is_dir()
    assert metrics_group.deployment_cfg.metrics_dir.parent == deployment_cfg.export_dir
    assert metrics_group.output_file.exists()
    assert metrics_group.metrics_group_name == "dummy_child_class"
    assert metrics_group.metrics_names == ["metric_0", "metric_1", "metric_2"]

    # Test - re-instantiation
    # Edditing the file
    metrics_group.output_file.write_text("Hello world!")
    new_metrics_group = DummyChildClass(deployment_cfg)
    assert new_metrics_group.output_file.read_text() == "Hello world!"

    # Cleaning
    shutil.rmtree(deployment_cfg.export_dir)


def test_measured_checkpoints():

    # Set up
    deployment_cfg = DeploymentCfg.autoconfig(deployment_cfg_path)
    shutil.rmtree(deployment_cfg.metrics_dir)
    metrics_group = DummyChildClass(deployment_cfg)

    # No checkpoints
    assert metrics_group.get_checkpoint_measured() == []

    # Adding one fake checkpoint
    with metrics_group.output_file.open("a") as f:
        f.write("0,-1.0,-1.0,-1.0\n")
    assert metrics_group.get_checkpoint_measured() == [0]

    # Adding a second fake checkpoint
    with metrics_group.output_file.open("a") as f:
        f.write("1,-1.0,-1.0,-1.0\n")
    assert metrics_group.get_checkpoint_measured() == [0, 1]

    # Cleaning
    shutil.rmtree(deployment_cfg.export_dir)


def test_load_metrics():

    # Set up
    deployment_cfg = DeploymentCfg.autoconfig(deployment_cfg_path)
    shutil.rmtree(deployment_cfg.metrics_dir)
    metrics_group = DummyChildClass(deployment_cfg)

    # Creating fake checkpoints
    metrics_group.compute_values(checkpoint=0)
    metrics_group.compute_values(checkpoint=1)

    # Tests
    values = metrics_group.load_metrics_df()
    assert all(values.columns == ["checkpoint"] + metrics_group.metrics_names)
    assert all(values["checkpoint"] == [0, 1])
    assert values["checkpoint"].dtype == int
    assert all(values["metric_0"] == [1.0, 2.0])
    assert values["metric_0"].dtype == float
    assert all(values["metric_1"] == [2.0, 3.0])
    assert values["metric_1"].dtype == float
    assert all(values["metric_2"] == [3.0, 4.0])
    assert values["metric_2"].dtype == float

    # Cleaning
    shutil.rmtree(deployment_cfg.export_dir)


def test_metrics_computation():

    # Set up
    deployment_cfg = DeploymentCfg.autoconfig(deployment_cfg_path)
    shutil.rmtree(deployment_cfg.metrics_dir)
    metrics_group = DummyChildClass(deployment_cfg)

    # Computing for checkpoint 0
    metrics_group.compute_values(checkpoint=0)
    assert metrics_group.output_file.read_text() == (
        "checkpoint,metric_0,metric_1,metric_2\n" "0,1.0,2.0,3.0\n"
    )

    # Computing for checkpoint 2
    metrics_group.compute_values(checkpoint=2)
    assert metrics_group.output_file.read_text() == (
        "checkpoint,metric_0,metric_1,metric_2\n" "0,1.0,2.0,3.0\n" "2,3.0,4.0,5.0\n"
    )

    # Computing for checkpoint 1
    metrics_group.compute_values(checkpoint=1)
    assert metrics_group.output_file.read_text() == (
        "checkpoint,metric_0,metric_1,metric_2\n"
        "0,1.0,2.0,3.0\n"
        "2,3.0,4.0,5.0\n"
        "1,2.0,3.0,4.0\n"
    )

    # Overwritting for additionnal tests
    metrics_group.output_file.write_text(
        "checkpoint,metric_0,metric_1,metric_2\n"
        "0,-1.0,-1.0,-1.0\n"
        "1,-1.0,-1.0,-1.0\n"
        "2,-1.0,-1.0,-1.0\n"
    )

    # Re-computing checkpoint 0 without forcing
    metrics_group.compute_values(checkpoint=0)
    assert metrics_group.output_file.read_text() == (
        "checkpoint,metric_0,metric_1,metric_2\n"
        "0,-1.0,-1.0,-1.0\n"
        "1,-1.0,-1.0,-1.0\n"
        "2,-1.0,-1.0,-1.0\n"
    )

    # Re-computing with forcing
    metrics_group.compute_values(checkpoint=0, recompute_if_exists=True)
    assert metrics_group.output_file.read_text() == (
        "checkpoint,metric_0,metric_1,metric_2\n"
        "1,-1.0,-1.0,-1.0\n"
        "2,-1.0,-1.0,-1.0\n"
        "0,1.0,2.0,3.0\n"
    )

    # Cleaning
    shutil.rmtree(deployment_cfg.export_dir)
