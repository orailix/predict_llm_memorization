"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import configparser
from pathlib import Path

RELATIVE_PATH = "relative"
ABSOLUTE_PATH = "absolute"

# Project root
path_root = Path(__file__).parent.parent.parent.parent
path_configs = path_root / "configs"

# Main config
cfg_main = configparser.ConfigParser()
cfg_main.read(path_configs / "main.cfg")

# Path to model hub
if cfg_main["paths.model_hub"]["mode"] == RELATIVE_PATH:
    path_model_hub = path_root / cfg_main["paths.model_hub"]["val"]
elif cfg_main["paths.model_hub"]["mode"] == ABSOLUTE_PATH:
    path_model_hub = Path(cfg_main["paths.model_hub"]["val"])
else:
    raise ValueError(
        f'cfg_main["paths.model_hub"]["mode"] not in {[RELATIVE_PATH, ABSOLUTE_PATH]}'
    )

# Path to data
if cfg_main["paths.data"]["mode"] == RELATIVE_PATH:
    path_data = path_root / cfg_main["paths.data"]["val"]
elif cfg_main["paths.data"]["mode"] == ABSOLUTE_PATH:
    path_data = Path(cfg_main["paths.data"]["val"])
else:
    raise ValueError(
        f'cfg_main["paths.data"]["mode"] not in {[RELATIVE_PATH, ABSOLUTE_PATH]}'
    )

# Path to output
if cfg_main["paths.output"]["mode"] == RELATIVE_PATH:
    path_output = path_root / cfg_main["paths.output"]["val"]
elif cfg_main["paths.output"]["mode"] == ABSOLUTE_PATH:
    path_output = Path(cfg_main["paths.output"]["val"])
else:
    raise ValueError(
        f'cfg_main["paths.output"]["mode"] not in {[RELATIVE_PATH, ABSOLUTE_PATH]}'
    )

# Path to logs
if cfg_main["paths.logs"]["mode"] == RELATIVE_PATH:
    path_logs = path_root / cfg_main["paths.logs"]["val"]
elif cfg_main["paths.logs"]["mode"] == ABSOLUTE_PATH:
    path_logs = Path(cfg_main["paths.logs"]["val"])
else:
    raise ValueError(
        f'cfg_main["paths.logs"]["mode"] not in {[RELATIVE_PATH, ABSOLUTE_PATH]}'
    )
