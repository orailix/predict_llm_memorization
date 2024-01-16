# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import collections
import typing as t

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm

from ..training import TrainingCfg, get_model
from .dynamic_metrics_group import DynamicMetricsGroup
from .utils.dataloaders import get_dataloaders_for_measures
from .utils.smi import smi_estimator


class SmiMetrics(DynamicMetricsGroup):
    """Class used to compute Sliced MUtual Information metrics on the models.

    Performance metrics: for each layer (128 in total):
        For dl in [train_all, train_trl, train_rdl, test]:
            - [{dl}_smi_0]
            - [{dl}_smi_1]
            (...)
            - [{dl}_smi_31]

    Where [dl_smi_k] is the sliced mutual information between:
        - The Values of the attention bloc at layer 0 for token N-3
        - The Label of token N-3, which is the answer of the MCQ
    (for dataloader `dl`)

    For SMI computation, we observed that the mutual information between
    random 1-dimensional projection and the label has a mean of about 4.5e-3
    and a standard deviation of about 5.5e-3.

    Thus, by default we take 2000 estimators, leading to a margin at 95% of
    about (1.96 x 5.5e-3)/sqrt(2000) = 2.4e-4   (≈5% of the mean).
    """

    def __init__(
        self, training_cfg: TrainingCfg, n_estimator: int = 2000, n_neighbors: int = 3
    ) -> None:
        self.n_estimator = n_estimator
        self.n_neighbors = n_neighbors
        super().__init__(training_cfg)

    @property
    def metrics_group_name(self) -> str:
        return "smi_metrics"

    @property
    def metrics_names(self) -> t.List[str]:
        return [
            f"{dl}_smi_{layer}"
            for dl in ["train_all", "train_trl", "train_rdl", "test"]
            for layer in range(32)
        ]

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:

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

        # Features: 0=train_trl, 1=train_rdl, 2=test
        # features_per_dl[0][4] = list of features of the 4th layer for train_all
        features_per_dl = [collections.defaultdict(list) for _ in range(3)]
        labels_per_dl = [[] for _ in range(3)]

        # Iterating over dataloaders
        for dl_idx, data_loader, info in zip(
            range(4),
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

                # Batch size
                bs = input_ids.size(0)

                # Model forward pass
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                # Saving features
                for layer in range(32):
                    features_per_dl[dl_idx][layer].append(
                        outputs["past_key_values"][layer][0][:, :, -3, :]
                        .view(bs, -1)
                        .cpu()
                    )

                # Saving labels
                labels_per_dl[dl_idx].append(input_ids[:, -2].cpu())

        # SMI containers
        smi_values = np.zeros((4, 32))

        # SMI -- TRL
        if len(train_trl_dl) != 0:
            # Logging
            logger.info("Computing SMI for: Train -- true labels")

            # Tensors
            X_per_layer = {
                layer: torch.cat(features_list, dim=0)
                for layer, features_list in features_per_dl[0].items()
            }
            y = torch.cat(labels_per_dl[0], dim=0)

            # Logging
            logger.debug(f"X size: {X_per_layer[31].size()}")
            logger.debug(f"y size: {y.size()}")

            # Multiprocessing for each layer
            logger.info("Computing the SMI for each of the 32 layers")
            smi_per_layer = []
            for layer in tqdm(range(32)):
                smi_per_layer.append(
                    smi_estimator(
                        X_per_layer[layer],
                        y,
                        n_estimator=self.n_estimator,
                        n_neighbors=self.n_neighbors,
                    )
                )

            smi_values[1, :] = smi_per_layer
        else:
            logger.info(f"No Train -- true labels samples: SMI=0")

        # SMI -- RDL
        if len(train_trl_dl) != 0:
            # Logging
            logger.info("Computing SMI for: Train -- random labels")

            # Tensors
            X_per_layer = {
                layer: torch.cat(features_list, dim=0)
                for layer, features_list in features_per_dl[1].items()
            }
            y = torch.cat(labels_per_dl[1], dim=0)

            # Logging
            logger.debug(f"X size: {X_per_layer[31].size()}")
            logger.debug(f"y size: {y.size()}")

            # Multiprocessing for each layer
            logger.info("Computing the SMI for each of the 32 layers")
            smi_per_layer = []
            for layer in tqdm(range(32)):
                smi_per_layer.append(
                    smi_estimator(
                        X_per_layer[layer],
                        y,
                        n_estimator=self.n_estimator,
                        n_neighbors=self.n_neighbors,
                    )
                )

            smi_values[2, :] = smi_per_layer
        else:
            logger.info(f"No Train -- random labels samples: SMI=0")

        # SMI -- Test
        if len(train_trl_dl) != 0:
            # Logging
            logger.info("Computing SMI for: Test")

            # Tensors
            X_per_layer = {
                layer: torch.cat(features_list, dim=0)
                for layer, features_list in features_per_dl[2].items()
            }
            y = torch.cat(labels_per_dl[2], dim=0)

            # Logging
            logger.debug(f"X size: {X_per_layer[31].size()}")
            logger.debug(f"y size: {y.size()}")

            # Multiprocessing for each layer
            logger.info("Computing the SMI for each of the 32 layers")
            smi_per_layer = []
            for layer in tqdm(range(32)):
                smi_per_layer.append(
                    smi_estimator(
                        X_per_layer[layer],
                        y,
                        n_estimator=self.n_estimator,
                        n_neighbors=self.n_neighbors,
                    )
                )

            smi_values[3, :] = smi_per_layer
        else:
            logger.info(f"No Test samples: SMI=0")

        # SMI -- Trail all
        if len(train_trl_dl) != 0:
            # Logging
            logger.info("Computing SMI for: Train -- all")

            # Tensors
            X_trl_per_layer = {
                layer: torch.cat(features_list, dim=0)
                for layer, features_list in features_per_dl[0].items()
            }
            X_rdl_per_layer = {
                layer: torch.cat(features_list, dim=0)
                for layer, features_list in features_per_dl[1].items()
            }
            X_per_layer = {
                layer: torch.cat(
                    [X_trl_per_layer[layer], X_rdl_per_layer[layer]], dim=0
                )
                for layer in range(32)
            }
            y_trl = torch.cat(labels_per_dl[0], dim=0)
            y_rdl = torch.cat(labels_per_dl[1], dim=0)
            y = torch.cat([y_trl, y_rdl], dim=0)

            # Logging
            logger.debug(f"X size: {X_per_layer[31].size()}")
            logger.debug(f"y size: {y.size()}")

            # Multiprocessing for each layer
            logger.info("Computing the SMI for each of the 32 layers")
            smi_per_layer = []
            for layer in tqdm(range(32)):
                smi_per_layer.append(
                    smi_estimator(
                        X_per_layer[layer],
                        y,
                        n_estimator=self.n_estimator,
                        n_neighbors=self.n_neighbors,
                    )
                )

            smi_values[0, :] = smi_per_layer
        else:
            logger.info(f"No Train -- all samples: SMI=0")

        # Output
        return [smi_values[dl_idx, layer] for dl_idx in range(4) for layer in range(32)]
