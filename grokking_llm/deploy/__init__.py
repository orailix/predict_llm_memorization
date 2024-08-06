# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

from .clean import (
    run_deploy_clean_forward_values,
    run_deploy_compress_all_forward_values,
)
from .cpu import run_deploy_cpu
from .gpu import run_deploy_gpu
from .prepare import run_deploy_prepare
