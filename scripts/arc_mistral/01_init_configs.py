# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
import sys
from pathlib import Path

sys.path.append("/lustre/fswork/projects/rech/yfw/upp42qa/grokking_llm")

from loguru import logger

from grokking_llm.utils import DeploymentCfg, TrainingCfg

# Training cfg
base_training_cfg_path = Path(__file__).parent / "00_base_training_cfg.json"
base_training_cfg = TrainingCfg.autoconfig(base_training_cfg_path)
base_training_cfg.get_output_dir()

# Deployment cfg
deployment_cfg_path = Path(__file__).parent / "00_deployment.cfg"
deployment_cfg = DeploymentCfg.autoconfig(deployment_cfg_path)

# Logging
logger.info(f"BASE TRAINING ID: {base_training_cfg.get_config_id()}")
logger.info(f"DEPLOYMENT ID: {deployment_cfg.get_deployment_id()}")
