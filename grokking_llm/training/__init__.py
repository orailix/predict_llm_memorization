# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

from .datasets import (
    add_labels,
    add_tokenized_possible_labels,
    format_dataset,
    get_dataset,
    get_random_split,
    get_tokenizer,
    save_dataset,
    tokenize_dataset,
)
from .main import run_main_train
from .models import get_model, get_num_params, save_model
from .trainer import compute_mcq_last_token_loss, get_trainer
