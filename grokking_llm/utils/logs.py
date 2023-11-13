"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

from loguru import logger

from . import paths

MAIN_LOG_PATH = paths.logs / "main.log"

paths.logs.mkdir(exist_ok=True, parents=True)
logger.add(
    paths.logs / "main.log",
    rotation="100 MB",
)

logger.info("Welcome to `grokking_llm` module!")
