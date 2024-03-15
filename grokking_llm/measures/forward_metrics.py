# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import typing as t

import torch
from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm

from ..training import TrainingCfg, get_model
from ..training.trainer import compute_mcq_last_token_loss
from ..utils.constants import SMI_LAYERS
from .dynamic_metrics_group import DynamicMetricsGroup
from .utils.dataloaders import get_dataloaders_for_measures
from .utils.forward_values import ForwardValues


class ForwardMetrics(DynamicMetricsGroup):
    """Class used to centralize all forward_pass computations."""

    def __init__(
        self, training_cfg: TrainingCfg, target_cfg: t.Optional[TrainingCfg] = None
    ) -> None:

        # Parsing target_cfg
        if target_cfg is None:
            self.target_cfg = training_cfg
            self.target_cfg_name = None
        else:
            self.target_cfg = target_cfg
            self.target_cfg_name = self.target_cfg.get_config_id()

        # Main initialization
        super().__init__(training_cfg)

    @property
    def metrics_group_name(self) -> str:
        if self.target_cfg_name is None:
            return "forward_metrics"
        else:
            return f"forward_metrics_on_{self.target_cfg_name}"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["Done?"]

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

        # Loading model
        logger.info(
            f"Loading model from measure config: {self.training_cfg.get_config_id()}"
        )
        model = get_model(self.training_cfg, at_checkpoint=checkpoint)
        vocab_size = model.config.vocab_size

        # Dataloaders
        logger.info(
            f"Loading datasets from target config: {self.target_cfg.get_config_id()}"
        )
        train_trl_dl, train_rdl_dl, test_all_dl = get_dataloaders_for_measures(
            self.target_cfg
        )

        # Accelerator init
        try:
            accelerator = Accelerator(mixed_precision="fp16")
        except ValueError:
            # In case the accelerator has already been implemented, we cannot change the mixed precision
            accelerator = Accelerator()
        model = accelerator.prepare_model(model, evaluation_mode=True)
        train_trl_dl, train_rdl_dl, test_all_dl = accelerator.prepare(
            train_trl_dl, train_rdl_dl, test_all_dl
        )
        model.eval()

        # Export dir
        forward_export_dir = (
            self.training_cfg.get_output_dir()
            / f"checkpoint-{checkpoint}"
            / "forward_values"
        )
        forward_export_dir.mkdir(exist_ok=True)

        # Iterating over dataloaders
        info_suffix = (
            "" if self.target_cfg_name is None else f"_on_{self.target_cfg_name}"
        )
        for data_loader, info in zip(
            [train_trl_dl, train_rdl_dl, test_all_dl],
            [
                "train_trl" + info_suffix,
                "train_rdl" + info_suffix,
                "test" + info_suffix,
            ],
        ):
            # Logging
            logger.info(f"Computing outputs of the model with dataloader: {info}")

            # Special case for empty dataloader
            if len(data_loader) == 0:
                ForwardValues(
                    name=info,
                    num_samples=0,
                    vocab_size=vocab_size,
                    global_index=torch.empty(),
                    input_ids=torch.empty(),
                    tokenized_possible_labels=torch.empty(),
                    inserted_label_index=torch.empty(),
                    loss_all=torch.empty(),
                    loss_asw=torch.empty(),
                    mcq_predicted_proba=torch.empty(),
                    mcq_states_per_layer=dict(),
                ).save(forward_export_dir)

            # Init containers
            num_sample_count = 0
            global_index_items = []
            input_ids_items = []
            tokenized_possible_labels_items = []
            inserted_label_index_items = []
            loss_all_items = []
            loss_asw_items = []
            mcq_predicted_proba_items = []
            mcq_states_per_layer_items = collections.defaultdict(list)

            # Iterating over dataloader
            for inputs in tqdm(data_loader):

                # Unpacking and pushing to device
                global_index = inputs["global_index"]
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                labels = inputs["labels"]
                tokenized_possible_labels = inputs["tokenized_possible_labels"]
                inserted_label_index = inputs["inserted_label_index"]

                # Batch size, vocab size
                bs = input_ids.size(0)

                # Model forward pass
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                # Saving inputs
                num_sample_count += bs
                global_index_items.append(global_index.cpu())
                input_ids_items.append(input_ids.cpu())
                tokenized_possible_labels_items.append(tokenized_possible_labels.cpu())
                inserted_label_index_items.append(inserted_label_index.cpu())

                # Losses
                loss_all_items.append(outputs["loss"].repeat(bs).view(bs, -1))
                loss_asw_items.append(
                    compute_mcq_last_token_loss(
                        labels, outputs["logits"], vocab_size, reduction="none"
                    ).view(bs, -1)
                )

                # Logits of possible answers
                logits = outputs["logits"]  # Shape (bs, 1024, vocab_size)
                logits_for_mcq_answer = logits[:, -3]  # Shape (bs, vocab_size)
                batch_indices = torch.arange(bs)[:, None]  # Shape (bs, 1)
                index_selector = tokenized_possible_labels.int()  # Shape (bs, 16)
                mcq_logits = logits_for_mcq_answer[
                    batch_indices, index_selector
                ]  # Shape (bs, 16)

                # Setting the logit to -1000 for padding indices
                mcq_logits[index_selector == 0] = -1000

                # Predicted proba
                mcq_predicted_proba_items.append(torch.softmax(mcq_logits, axis=1))

                # MCQ states per layer
                for layer in SMI_LAYERS:
                    mcq_states_per_layer_items[layer].append(
                        outputs["past_key_values"][layer][0][:, :, -3, :]
                        .view(bs, -1)
                        .cpu()
                    )

            # Saving ForwardValues
            forward_values = ForwardValues(
                name=info,
                num_samples=num_sample_count,
                vocab_size=vocab_size,
                input_ids=torch.cat(input_ids_items, dim=0),
                global_index=torch.cat(global_index_items, dim=0),
                tokenized_possible_labels=torch.cat(
                    tokenized_possible_labels_items, dim=0
                ),
                inserted_label_index=torch.cat(inserted_label_index_items, dim=0),
                loss_all=torch.cat(loss_all_items, dim=0),
                loss_asw=torch.cat(loss_asw_items, dim=0),
                mcq_predicted_proba=torch.cat(mcq_predicted_proba_items, dim=0),
                mcq_states_per_layer={
                    layer: torch.cat(mcq_states_per_layer_items[layer], dim=0)
                    for layer in SMI_LAYERS
                },
            )

            # Sanity check
            if forward_values.num_samples != forward_values.input_ids.size(0):
                raise ValueError(
                    f"Inconsistent dataloader size: {forward_values.num_samples} != {forward_values.input_ids.size(0)}"
                )

            # Saving
            forward_values.save(forward_export_dir)

        return [1.0]
