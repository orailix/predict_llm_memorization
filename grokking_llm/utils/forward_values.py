# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import dataclasses
import typing as t
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

from .training_cfg import TrainingCfg


@dataclasses.dataclass
class ForwardValues:
    """Class used to represent the forward values."""

    # General
    # Name: train_trl / train_rdl / test
    name: str
    num_samples: int
    vocab_size: int

    # Inputs
    global_index: torch.Tensor
    input_ids: torch.Tensor
    tokenized_possible_labels: torch.Tensor
    inserted_label_index: torch.Tensor

    # Outputs
    loss_all: torch.Tensor
    loss_asw: torch.Tensor
    mcq_predicted_proba: torch.Tensor
    mcq_predicted_logits: torch.Tensor
    mcq_states_per_layer: t.Dict[int, torch.Tensor]

    @property
    def mcq_labels(self) -> torch.Tensor:
        return self.tokenized_possible_labels[
            torch.arange(self.num_samples),
            self.inserted_label_index.int(),
        ]

    def __repr__(self) -> str:
        states_size = self.mcq_states_per_layer[
            list(self.mcq_states_per_layer)[0]
        ].size()
        return f"""ForwardValues object:
    - name: {self.name}
    - num_samples: {self.num_samples}
    - vocab_size: {self.vocab_size}
    - global_index: torch.Tensor of size: {self.global_index.size()}
    - input_ids: torch.Tensor of size: {self.input_ids.size()}
    - tokenized_possible_labels: torch.Tensor of size: {self.tokenized_possible_labels.size()}
    - inserted_label_index: torch.Tensor of size: {self.inserted_label_index.size()}
    - loss_all: torch.Tensor of size: {self.loss_all.size()}
    - loss_asw: torch.Tensor of size: {self.loss_asw.size()}
    - mcq_predicted_proba: torch.Tensor of size: {self.mcq_predicted_proba.size()}
    - mcq_predicted_logits: torch.Tensor of size: {self.mcq_predicted_logits.size()}
    - mcq_states_per_layer: {sorted(list(self.mcq_states_per_layer))} -> torch.Tensor of size {states_size}
"""

    def save(self, save_dir: t.Union[str, Path]):

        # Sanity checks
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)

        if not save_dir.is_dir():
            raise FileNotFoundError(
                f"Directory not found for ForwardValues export: {save_dir}"
            )

        # Packing saving tensors
        simple_attributes = dict(
            num_samples=torch.Tensor([self.num_samples]),
            vocab_size=torch.Tensor([self.vocab_size]),
            global_index=self.global_index,
            input_ids=self.input_ids,
            tokenized_possible_labels=self.tokenized_possible_labels,
            inserted_label_index=self.inserted_label_index,
            loss_all=self.loss_all,
            loss_asw=self.loss_asw,
            mcq_predicted_proba=self.mcq_predicted_proba,
            mcq_predicted_logits=self.mcq_predicted_logits,
        )
        mcq_states = {
            f"mcq_states_layer_{layer}": self.mcq_states_per_layer[layer]
            for layer in self.mcq_states_per_layer
        }
        to_save = {**simple_attributes, **mcq_states}

        # Saving
        save_name = save_dir / f"{self.name}.safetensors"
        logger.info(f"Saving ForwardOutputs at {save_name}")
        save_file(to_save, save_name)

    @classmethod
    def load(cls, path: t.Union[str, Path]):

        # Sanity checks
        if not isinstance(path, Path):
            path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"File not found for ForwardValues import: {path}")

        with safe_open(path, framework="pt", device="cpu") as f:
            mcq_states_per_layer = {
                int(key[len("mcq_states_layer_") :]): f.get_tensor(key)
                for key in f.keys()
                if "mcq_states_layer_" in key
            }
            return cls(
                name=path.stem,
                num_samples=int(f.get_tensor("num_samples").item()),
                vocab_size=int(f.get_tensor("vocab_size").item()),
                global_index=f.get_tensor("global_index"),
                input_ids=f.get_tensor("input_ids"),
                tokenized_possible_labels=f.get_tensor("tokenized_possible_labels"),
                inserted_label_index=f.get_tensor("inserted_label_index"),
                loss_all=f.get_tensor("loss_all"),
                loss_asw=f.get_tensor("loss_asw"),
                mcq_predicted_proba=f.get_tensor("mcq_predicted_proba"),
                mcq_predicted_logits=f.get_tensor("mcq_predicted_logits"),
                mcq_states_per_layer=mcq_states_per_layer,
            )

    @classmethod
    def concat(cls, values_0, values_1, new_name: str):

        if not isinstance(values_0, ForwardValues):
            raise TypeError(f"ForwardValues can only be concatenated to ForwardValues.")

        if not isinstance(values_1, ForwardValues):
            raise TypeError(f"ForwardValues can only be concatenated to ForwardValues.")

        if values_0.vocab_size != values_1.vocab_size:
            raise ValueError(
                f"Incompatible vocab size for concatenation: {values_0.vocab_size} "
                f"!= {values_1.vocab_size}"
            )

        if sorted(list(values_0.mcq_states_per_layer)) != sorted(
            list(values_1.mcq_states_per_layer)
        ):
            raise ValueError(
                f"Incompatible layer list for concatenation: {sorted(list(values_0.mcq_states_per_layer))} "
                f"!= {sorted(list(values_1.mcq_states_per_layer))}"
            )

        return cls(
            name=new_name,
            num_samples=values_0.num_samples + values_1.num_samples,
            vocab_size=values_0.vocab_size,
            global_index=torch.cat(
                [values_0.global_index, values_1.global_index], dim=0
            ),
            input_ids=torch.cat([values_0.input_ids, values_1.input_ids], dim=0),
            tokenized_possible_labels=torch.cat(
                [
                    values_0.tokenized_possible_labels,
                    values_1.tokenized_possible_labels,
                ],
                dim=0,
            ),
            inserted_label_index=torch.cat(
                [values_0.inserted_label_index, values_1.inserted_label_index], dim=0
            ),
            loss_all=torch.cat([values_0.loss_all, values_1.loss_all], dim=0),
            loss_asw=torch.cat([values_0.loss_asw, values_1.loss_asw], dim=0),
            mcq_predicted_proba=torch.cat(
                [values_0.mcq_predicted_proba, values_1.mcq_predicted_proba], dim=0
            ),
            mcq_predicted_logits=torch.cat(
                [values_0.mcq_predicted_logits, values_1.mcq_predicted_logits], dim=0
            ),
            mcq_states_per_layer={
                layer: torch.cat(
                    [
                        values_0.mcq_states_per_layer[layer],
                        values_1.mcq_states_per_layer[layer],
                    ],
                    dim=0,
                )
                for layer in values_0.mcq_states_per_layer
            },
        )


def get_forward_values(
    training_cfg: TrainingCfg,
    checkpoint: int,
    name: str,
    enable_compressed: bool = False,
) -> ForwardValues:
    """Gets the forward values.

    Args:
        training_cfg: The config of the model to get the values from
        checkpoint: The checkpoint of the model to get the values from
        name: The name of the forwad values to look for
        enable_compressed: If the forward values are not found, look at a compressed version of them."""

    # Logging
    logger.debug(
        f"Loading forward values {name} from config {training_cfg.get_config_id()} at checkpoint {checkpoint}"
    )

    # Export dir
    forward_export_dir = (
        training_cfg.get_output_dir() / f"checkpoint-{checkpoint}" / "forward_values"
    )

    # First attempt
    load_path = forward_export_dir / f"{name}.safetensors"
    if load_path.is_file():
        return ForwardValues.load(load_path)

    # Second attempt - compressed values
    if enable_compressed:
        load_path = forward_export_dir / f"compressed_{name}.safetensors"
        if load_path.is_file():
            return ForwardValues.load(load_path)

    # Can we replace "on_<config_id>.safetensors" ?
    possible_config_id = name[-22:]
    if possible_config_id == training_cfg.get_config_id():
        new_name = name[:-26]

        # Third attempt
        load_path = forward_export_dir / f"{new_name}.safetensors"
        if load_path.is_file():
            return ForwardValues.load(load_path)

        # Fourth attempt
        if enable_compressed:
            load_path = forward_export_dir / f"compressed_{new_name}.safetensors"
            if load_path.is_file():
                return ForwardValues.load(load_path)

    # If nothing worked...
    logger.warning(
        f"Some ForwardValues were not found. Maybe you forgot to measure forward values before?"
    )
    raise FileNotFoundError(
        f"Unfound forward values: {name} at {training_cfg.get_config_id()} checkpoint {checkpoint} with enable_compressed={enable_compressed}"
    )
