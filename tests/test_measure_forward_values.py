# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import shutil

import pytest
import torch

from grokking_llm.measures import ForwardValues
from grokking_llm.utils import paths


def test_forward_values():

    # Init
    export_path = paths.output / "forward_values"
    if export_path.is_dir():
        shutil.rmtree(export_path)
    export_path.mkdir(exist_ok=True, parents=True)

    # Creating object
    forward_values = ForwardValues(
        name="train_trl",
        num_samples=0,
        vocab_size=32000,
        global_index=torch.Tensor(range(-1, 9)),
        input_ids=torch.Tensor(range(0, 10)),
        tokenized_possible_labels=torch.Tensor(range(1, 11)),
        inserted_label_index=torch.Tensor(range(2, 12)),
        loss_all=torch.Tensor(range(3, 13)),
        loss_asw=torch.Tensor(range(4, 14)),
        mcq_predicted_proba=torch.Tensor(range(5, 15)),
        mcq_states_per_layer={
            6: torch.Tensor(range(6, 16)),
            7: torch.Tensor(range(7, 17)),
        },
    )

    # Saving
    forward_values.save(export_path)
    assert (export_path / "train_trl.safetensors").is_file()

    # Loading
    reloaded = ForwardValues.load(export_path / "train_trl.safetensors")
    assert reloaded.name == "train_trl"
    assert reloaded.num_samples == 0
    assert reloaded.vocab_size == 32000
    assert (reloaded.global_index == torch.Tensor(range(-1, 9))).all()
    assert (reloaded.input_ids == torch.Tensor(range(0, 10))).all()
    assert (reloaded.tokenized_possible_labels == torch.Tensor(range(1, 11))).all()
    assert (reloaded.inserted_label_index == torch.Tensor(range(2, 12))).all()
    assert (reloaded.loss_all == torch.Tensor(range(3, 13))).all()
    assert (reloaded.loss_asw == torch.Tensor(range(4, 14))).all()
    assert (reloaded.mcq_predicted_proba == torch.Tensor(range(5, 15))).all()
    assert sorted(list(reloaded.mcq_states_per_layer)) == [6, 7]
    assert (reloaded.mcq_states_per_layer[6] == torch.Tensor(range(6, 16))).all()
    assert (reloaded.mcq_states_per_layer[7] == torch.Tensor(range(7, 17))).all()

    # Cleaning
    if export_path.is_dir():
        try:
            shutil.rmtree(export_path)
        except OSError:
            pass


def test_concatenation():

    # Creating object
    forward_values_0 = ForwardValues(
        name="train_trl",
        num_samples=0,
        vocab_size=32000,
        global_index=torch.Tensor(range(-1, 9)),
        input_ids=torch.Tensor(range(0, 10)),
        tokenized_possible_labels=torch.Tensor(range(1, 11)),
        inserted_label_index=torch.Tensor(range(2, 12)),
        loss_all=torch.Tensor(range(3, 13)),
        loss_asw=torch.Tensor(range(4, 14)),
        mcq_predicted_proba=torch.Tensor(range(5, 15)),
        mcq_states_per_layer={
            6: torch.Tensor(range(6, 16)),
            7: torch.Tensor(range(7, 17)),
        },
    )

    # Compatible with add
    forward_values_1 = ForwardValues(
        name="train_trl",
        num_samples=0,
        vocab_size=32000,
        global_index=torch.Tensor(range(-1, 9)),
        input_ids=torch.Tensor(range(0, 10)),
        tokenized_possible_labels=torch.Tensor(range(1, 11)),
        inserted_label_index=torch.Tensor(range(2, 12)),
        loss_all=torch.Tensor(range(3, 13)),
        loss_asw=torch.Tensor(range(4, 14)),
        mcq_predicted_proba=torch.Tensor(range(5, 15)),
        mcq_states_per_layer={
            6: torch.Tensor(range(6, 16)),
            7: torch.Tensor(range(7, 17)),
        },
    )

    # Incorrect vocab size
    forward_values_2 = ForwardValues(
        name="train_trl",
        num_samples=0,
        vocab_size=0,
        global_index=torch.Tensor(range(-1, 9)),
        input_ids=torch.Tensor(range(0, 10)),
        tokenized_possible_labels=torch.Tensor(range(1, 11)),
        inserted_label_index=torch.Tensor(range(2, 12)),
        loss_all=torch.Tensor(range(3, 13)),
        loss_asw=torch.Tensor(range(4, 14)),
        mcq_predicted_proba=torch.Tensor(range(5, 15)),
        mcq_states_per_layer={
            6: torch.Tensor(range(6, 16)),
            7: torch.Tensor(range(7, 17)),
        },
    )

    # Incorrect layers
    forward_values_3 = ForwardValues(
        name="train_trl",
        num_samples=0,
        vocab_size=0,
        global_index=torch.Tensor(range(-1, 9)),
        input_ids=torch.Tensor(range(0, 10)),
        tokenized_possible_labels=torch.Tensor(range(1, 11)),
        inserted_label_index=torch.Tensor(range(2, 12)),
        loss_all=torch.Tensor(range(3, 13)),
        loss_asw=torch.Tensor(range(4, 14)),
        mcq_predicted_proba=torch.Tensor(range(5, 15)),
        mcq_states_per_layer={
            6: torch.Tensor(range(6, 16)),
        },
    )

    # Check errors
    with pytest.raises(TypeError):
        ForwardValues.concat(forward_values_0, 0, new_name="test")

    with pytest.raises(TypeError):
        ForwardValues.concat(0, forward_values_0, new_name="test")

    with pytest.raises(ValueError):
        ForwardValues.concat(forward_values_0, forward_values_2, new_name="test")

    with pytest.raises(ValueError):
        ForwardValues.concat(forward_values_0, forward_values_3, new_name="test")

    # Check add
    concatenation = ForwardValues.concat(
        forward_values_0, forward_values_1, new_name="test"
    )
    assert concatenation.name == "test"
    assert concatenation.num_samples == 0
    assert concatenation.vocab_size == 32000
    assert (concatenation.global_index == torch.Tensor(range(-1, 9)).repeat(2)).all()
    assert (concatenation.input_ids == torch.Tensor(range(0, 10)).repeat(2)).all()
    assert (
        concatenation.tokenized_possible_labels == torch.Tensor(range(1, 11)).repeat(2)
    ).all()
    assert (
        concatenation.inserted_label_index == torch.Tensor(range(2, 12)).repeat(2)
    ).all()
    assert (concatenation.loss_all == torch.Tensor(range(3, 13)).repeat(2)).all()
    assert (concatenation.loss_asw == torch.Tensor(range(4, 14)).repeat(2)).all()
    assert (
        concatenation.mcq_predicted_proba == torch.Tensor(range(5, 15)).repeat(2)
    ).all()
    assert sorted(list(concatenation.mcq_states_per_layer)) == [6, 7]
    assert (
        concatenation.mcq_states_per_layer[6] == torch.Tensor(range(6, 16)).repeat(2)
    ).all()
    assert (
        concatenation.mcq_states_per_layer[7] == torch.Tensor(range(7, 17)).repeat(2)
    ).all()
