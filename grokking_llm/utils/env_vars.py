# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import configparser
import os

from . import paths
from .logs import logger

if paths.env_vars_cfg_path.is_file():

    # Logging
    logger.info(
        f"Found an environment var config at {paths.env_vars_cfg_path}, exporting its variables."
    )

    # Main config
    env_vars_cfg_object = configparser.ConfigParser()
    env_vars_cfg_object.optionxform = str
    env_vars_cfg_object.read(paths.env_vars_cfg_path)

    # Exporting
    if "main" not in env_vars_cfg_object.sections():
        raise ValueError("Your environment var config should contain a `main` section.")

    for key, value in env_vars_cfg_object["main"].items():
        if key in os.environ:
            logger.debug(
                f"Skipping env var {key}={value} because {key} was already an env variable"
            )
        else:
            logger.debug(f"Exporting env var {key}={value}")
            os.environ[key] = value

else:
    logger.info(f"No environment var confit found at {paths.env_vars_cfg_path}.")
