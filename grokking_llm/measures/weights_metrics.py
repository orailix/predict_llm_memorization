# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import numpy as np
from loguru import logger
from safetensors import safe_open
from tqdm import tqdm

from .dynamic_metrics_group import DynamicMetricsGroup


class WeightsMetrics(DynamicMetricsGroup):
    """Class used to compute basic performance metrics on the models.

    Weights metrics: (4x2 = 8 metrics in total)
        Prefix:
        - [frob] The Frobenius norm (sqrt of the sum of squarred coefficients)
        - [nuc] The nuclear norm (sum of the singular values)
        - [Linf] The L infinity norm (max sum of abs of the coefs in a row)
        - [L2] The spectral norm (largest singular value)

        (cf. documentation of numpy.linalg.norm for details about the norm computations)

        Suffix:
        - [norm] The total norm of the LoRA matrices
        - [dist] The distance from the initialization of the LoRA matrices
    """

    @property
    def metrics_group_name(self) -> str:
        return "weights_metrics"

    @property
    def metrics_names(self) -> t.List[str]:
        result = []
        for prefix in "frob", "nuc", "Linf", "L2":
            for suffix in "norm", "dist":
                result.append(f"{prefix}_{suffix}")

        return result

    def metrics_computation_core(self, checkpoint: int) -> t.List[float]:
        # Check that checkpoint 0 has been saved
        if 0 not in self.training_cfg.get_available_checkpoints():
            raise ValueError(
                f"Computation of metrics `{self.metrics_names}` requires checkpoint 0 to be save."
            )

        # Loading safe tensors
        logger.debug(
            f"Loading adapter_model.safetensors for checkpoint 0 at config {self.config_input_id}"
        )
        tensors_checkpoint_0 = safe_open(
            self.training_cfg.get_output_dir()
            / "checkpoint-0"
            / "adapter_model.safetensors",
            framework="torch",
        )
        logger.debug(
            f"Loading adapter_model.safetensors for checkpoint {checkpoint} at config {self.config_input_id}"
        )
        tensors_checkpoint_now = safe_open(
            self.training_cfg.get_output_dir()
            / f"checkpoint-{checkpoint}"
            / "adapter_model.safetensors",
            framework="torch",
        )

        # Init values
        # X axis: frob, nuc, Linf, L2
        # Y axis: norm, dist
        values = np.zeros((4, 2), dtype=float)
        tot_num_params = 0

        logger.debug(f"Computing the norm of each parameter of the adapter")
        for key_0, key_now in tqdm(
            list(
                zip(
                    sorted(tensors_checkpoint_0.keys()),
                    sorted(tensors_checkpoint_now.keys()),
                )
            )
        ):

            # Sanity check
            if key_0 != key_now:
                raise ValueError(
                    f"Inconsistency in key loaded from LoRA adapters safetensors: {key_0} vs {key_now}"
                )

            # Ignoring non_lora parameters
            if "lora" not in key_0:
                continue

            # Fetching tensors
            tensor_0 = tensors_checkpoint_0.get_tensor(key_0)
            tensor_now = tensors_checkpoint_now.get_tensor(key_now)

            # Num of param
            num_params = tensor_0.numel()
            tot_num_params += num_params

            # Computing -- Frobenius norm
            values[0, 0] += num_params * np.linalg.norm(tensor_now, ord="fro")
            values[0, 1] += num_params * np.linalg.norm(
                tensor_0 - tensor_now, ord="fro"
            )

            # Computing -- Nuclear norm
            values[1, 0] += num_params * np.linalg.norm(tensor_now, ord="nuc")
            values[1, 1] += num_params * np.linalg.norm(
                tensor_0 - tensor_now, ord="nuc"
            )

            # Computing -- L infinity norm
            values[2, 0] += num_params * np.linalg.norm(tensor_now, ord=np.inf)
            values[2, 1] += num_params * np.linalg.norm(
                tensor_0 - tensor_now, ord=np.inf
            )

            # Computing -- L 2 norm
            values[3, 0] += num_params * np.linalg.norm(tensor_now, ord=2)
            values[3, 1] += num_params * np.linalg.norm(tensor_0 - tensor_now, ord=2)

        # Averaging
        values /= tot_num_params

        # Output
        return [
            values[norm_name, diff_or_abs]
            for norm_name in range(4)
            for diff_or_abs in range(2)
        ]
