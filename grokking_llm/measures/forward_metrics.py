# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import torch
from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm

from ..training import TrainingCfg, get_model
from .dynamic_metrics_group import DynamicMetricsGroup
from .perf_metrics import PerfMetrics
from .smi_metrics import SmiMetrics
from .utils.dataloaders import get_dataloaders_for_measures


class ForwardMetrics(DynamicMetricsGroup):
    """Class used to centralize all forward_pass computations."""

    def __init__(self, training_cfg: TrainingCfg) -> None:
        super().__init__(training_cfg)
        self.perf_metrics = PerfMetrics(self.training_cfg)
        self.smi_metrics = SmiMetrics(self.training_cfg)

    @property
    def metrics_group_name(self) -> str:
        return "forward_metrics"

    @property
    def metrics_names(self) -> t.List[str]:
        return ["Done?"]

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:
        # Already done ?
        if (
            checkpoint in self.perf_metrics.get_checkpoint_measured()
            and checkpoint in self.smi_metrics.get_checkpoint_measured()
        ):
            return [1.0]

        # Loading model
        model = get_model(self.training_cfg, at_checkpoint=checkpoint)

        # Dataloaders
        train_trl_dl, train_rdl_dl, test_all_dl = get_dataloaders_for_measures(
            self.training_cfg
        )

        # Accelerator
        accelerator = Accelerator(mixed_precision="fp16")
        model = accelerator.prepare_model(model, evaluation_mode=True)
        train_trl_dl, train_rdl_dl, test_all_dl = accelerator.prepare(
            train_trl_dl, train_rdl_dl, test_all_dl
        )
        model.eval()

        # Prepare metrics
        if checkpoint not in self.perf_metrics.get_checkpoint_measured():
            self.perf_metrics.prepare_forward_measure(checkpoint=checkpoint)

        if checkpoint not in self.smi_metrics.get_checkpoint_measured():
            self.smi_metrics.prepare_forward_measure(
                checkpoint=checkpoint,
                len_trl=len(train_trl_dl),
                len_rdl=len(train_rdl_dl),
                len_test=len(test_all_dl),
            )

        # Iterating over dataloaders
        for dl_idx, data_loader, info in zip(
            range(1, 4),
            [train_trl_dl, train_rdl_dl, test_all_dl],
            ["Train -- true labels", "Train -- random labels", "Test"],
        ):
            # Logging
            logger.info(f"Computing outputs of the model with dataloader: {info}")

            if len(data_loader) == 0:
                continue

            for inputs in tqdm(data_loader):

                # Unpacking and pushing to device
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                labels = inputs["labels"]
                tokenized_possible_labels = inputs["tokenized_possible_labels"]
                inserted_label_index = inputs["inserted_label_index"]

                # Batch size
                bs = input_ids.size(0)

                # Model forward pass
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                # Update metrics
                if checkpoint not in self.perf_metrics.get_checkpoint_measured():
                    self.perf_metrics.update_metrics(
                        dl_idx=dl_idx,
                        bs=bs,
                        vocab_size=model.config.vocab_size,
                        labels=labels,
                        tokenized_possible_labels=tokenized_possible_labels,
                        inserted_label_index=inserted_label_index,
                        outputs=outputs,
                    )

                if checkpoint not in self.smi_metrics.get_checkpoint_measured():
                    self.smi_metrics.update_metrics(
                        dl_idx=dl_idx,
                        bs=bs,
                        input_ids=input_ids,
                        outputs=outputs,
                    )

        # Finalize metrics
        if checkpoint not in self.perf_metrics.get_checkpoint_measured():
            self.perf_metrics.finalize_metrics()

        if checkpoint not in self.smi_metrics.get_checkpoint_measured():
            self.smi_metrics.finalize_metrics()

        return [1.0]
