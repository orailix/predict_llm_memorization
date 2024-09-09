# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.
# isort: skip_file

# These modules are necessarily imported first, and in this order
from . import logs
from . import env_vars
from . import hf_hub

# Now, we import the objects that can be accessed with `from grokking_llm.utils import ...`
from .deployment.deployment_cfg import DeploymentCfg
from .deployment.disk_stacks import DiskStack
from .deployment.parsed_section import ParsedSection
from .end_signals import GotEndSignal
from .forward_values import ForwardValues, get_forward_values
from .mahalanobis import mahalanobis_distance
from .pointwise_metrics import (
    LightForwardValues,
    get_logit_gaps_for_pointwise,
    get_losses_for_pointwise,
    get_mia_memo_score,
    get_pointwise_container,
    get_pointwise_layerwise_container,
    get_shadow_forward_values_for_pointwise,
    norm_pdf,
)
from .possible_training_cfg import get_possible_training_cfg
from .smi import get_p_smi_containers, p_smi_estimator, smi_estimator
from .training_cfg import TrainingCfg
