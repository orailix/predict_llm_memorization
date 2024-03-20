# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import dataclasses


@dataclasses.dataclass
class ParsedSection:
    """Class representing a parsed section from a deployment config.

    The name of the section will refer to attributes of TrainingCfg. Example:
        - `lora_dropout`
        - `training_args.weight_decay`

    Attributes:
        name (str): The name of the section
        values (list): A list of values it should take
    """

    name: str
    values: list
