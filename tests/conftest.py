# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
from pathlib import Path

from grokking_llm.utils import paths

# Re-directing outputs
paths.output = Path(__file__).parent / "outputs"
paths.configs = Path(__file__).parent / "configs"
paths.deployment_configs = paths.configs / ".deployment_configs"

# To remove a warning from Jupyter
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
