# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import warnings

from loguru import logger

from . import paths

MAIN_LOG_PATH = paths.logs / "main.log"

paths.logs.mkdir(exist_ok=True, parents=True)
logger.add(
    paths.logs / "main.log",
    rotation="10 MB",
)

logger.info("Welcome to `grokking_llm` module!")

warnings.filterwarnings(
    "ignore", r".*Using the latest cached version of the module from.*"
)
warnings.filterwarnings(
    "ignore", r"torch.utils._pytree._register_pytree_node is deprecated."
)
warnings.filterwarnings("ignore", r"IProgress not found.")
