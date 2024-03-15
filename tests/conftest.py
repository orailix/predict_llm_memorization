# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
from pathlib import Path

from grokking_llm.utils import paths

# Re-directing outputs
paths.output = Path(__file__).parent / "output"
paths.individual_outputs = paths.output / "individual"
paths.deployment_outputs = paths.output / "deployment"
paths.configs = Path(__file__).parent / "configs"

# To remove a warning from Jupyter
os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
