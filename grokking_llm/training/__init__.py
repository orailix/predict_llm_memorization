"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

from .datasets import (
    add_labels,
    format_dataset,
    get_dataset,
    get_random_split,
    get_tokenizer,
    tokenize_dataset,
)
from .models import get_model, get_num_params, save_model
from .trainer import get_trainer
from .training_cfg import TrainingCfg
