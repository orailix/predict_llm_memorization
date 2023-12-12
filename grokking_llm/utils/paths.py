"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import configparser
import os
from pathlib import Path

RELATIVE_PATH = "relative"
ABSOLUTE_PATH = "absolute"


# Project root
root = Path(__file__).parent.parent.parent
configs = root / "configs"

# Main config
main_cfg_object = configparser.ConfigParser()
main_cfg_object.read(configs / "main.cfg")

# Path to user directory
if "HOME" in os.environ:
    user_dir = Path(os.environ["HOME"])  # Unix
elif "HOMEPATH" in os.environ:
    user_dir = Path(os.environ["HOMEPATH"])  # Windows
else:
    raise ValueError(
        "User directory found neither in $HOME nor $HOMEPATH. Please declare env variable $HOME manually."
    )

# Path to model hub
if "val" not in main_cfg_object["paths.hf_home"]:
    hf_home = user_dir / ".cache" / "huggingface"
elif main_cfg_object["paths.hf_home"]["mode"] == RELATIVE_PATH:
    hf_home = root / main_cfg_object["paths.hf_home"]["val"]
elif main_cfg_object["paths.hf_home"]["mode"] == ABSOLUTE_PATH:
    hf_home = Path(main_cfg_object["paths.hf_home"]["val"])
else:
    raise ValueError(
        f'main_cfg["paths.hf_home"]["mode"] not in {[RELATIVE_PATH, ABSOLUTE_PATH]}'
    )

# Path to data
if main_cfg_object["paths.data"]["mode"] == RELATIVE_PATH:
    data = root / main_cfg_object["paths.data"]["val"]
elif main_cfg_object["paths.data"]["mode"] == ABSOLUTE_PATH:
    data = Path(main_cfg_object["paths.data"]["val"])
else:
    raise ValueError(
        f'main_cfg["paths.data"]["mode"] not in {[RELATIVE_PATH, ABSOLUTE_PATH]}'
    )

# Path to output
if main_cfg_object["paths.output"]["mode"] == RELATIVE_PATH:
    output = root / main_cfg_object["paths.output"]["val"]
elif main_cfg_object["paths.output"]["mode"] == ABSOLUTE_PATH:
    output = Path(main_cfg_object["paths.output"]["val"])
else:
    raise ValueError(
        f'main_cfg["paths.output"]["mode"] not in {[RELATIVE_PATH, ABSOLUTE_PATH]}'
    )

# Path to logs
if main_cfg_object["paths.logs"]["mode"] == RELATIVE_PATH:
    logs = root / main_cfg_object["paths.logs"]["val"]
elif main_cfg_object["paths.logs"]["mode"] == ABSOLUTE_PATH:
    logs = Path(main_cfg_object["paths.logs"]["val"])
else:
    raise ValueError(
        f'main_cfg["paths.logs"]["mode"] not in {[RELATIVE_PATH, ABSOLUTE_PATH]}'
    )
