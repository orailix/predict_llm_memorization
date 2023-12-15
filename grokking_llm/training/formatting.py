"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import typing as t

import numpy as np

from ..utils.constants import DATASET_BARE_LABEL

MCQ_TEMPLATES = [
    "{question}\n\n{options_}\n",
    "Question: {question}\n\nAnswer:\n\n{options_}\n",
    "Question: {question}\n\nWhat is the correct answer to the question from the following choices?\n\n{options_}\n",
    "Q: {question}\n\nWhat is the correct answer to this question?\n\n{options_}\n",
    "What is the answer?\n\n{question}\n\n{options_}\n",
    "Answer the question.\n\n{question}\n\n{options_}\n",
    "{question}\n\nPick the answer from these options.\n\n{options_}\n",
    "Could you help me with this?\n\n{question}\n\n{options_}\n",
    "{question}\n\n{options_}\n\nPlease solve this MCQ for me.\n",
    "{question}\n\n{options_}\n\nChoose one of the following options.\n",
]

ETHICS_TEMPLATES = [
    "{input}\n\nOPTIONS:\n- 0: Ethical\n- 1: Unethical\n",
    "{input}\n\nOPTIONS:\n- 0: Good\n- 1: Bad\n",
    "{input}\n\nOPTIONS:\n- 0: Innocuous\n- 1: Harmful\n",
    "{input}\n\nOPTIONS:\n- 0: Moral\n- 1: Immoral\n",
    "Did I do the right thing?\n\n{input}\n\nOPTIONS:\n- 0: Yes\n- 1: No\n",
    "What do you think of my behavior?\n\n{input}\n\nOPTIONS:\n- 0: Innocuous\n- 1: Harmful\n",
    "{input}\n\nHow yould you describe what I did?\n\nOPTIONS:\n- 0: Moral\n- 1: Immoral\n",
    "{input}\n\nIs it fine to do this again?\n\nOPTIONS:\n- 0: Yes\n- 1: No\n",
    "{input}\n\nDo you think it could hurt someone?\n\nOPTIONS:\n- 0: No\n- 1: Yes\n",
    "{input}\n\nWould you have recommended me to do this?\n\nOPTIONS:\n- 0: Yes\n- 1: No\n",
]

ETHICS_TEMPLATES_AITA = [
    "{input}\n\nOPTIONS:\n- 0: Moral\n- 1: Immoral\n",
    "{input}\n\nOPTIONS:\n- 0: Ethical\n- 1: Unethical\n",
    "{input}\n\nOPTIONS:\n- 0: No\n- 1: Yes\n",
    "{input}\n\nAITA?\n\nOPTIONS:\n- 0: No\n- 1: Yes\n",
    "{input}\n\nWIBTA?\n\nOPTIONS:\n- 0: No\n- 1: Yes\n",
]


def format_arc(
    sample: dict,
    force_template: bool = False,
    random_state: t.Optional[np.random.RandomState] = None,
) -> dict:
    """Format prompt from ARC dataset.

    Args:
        samples: A sample from the ARC dataset
        force_template: if True, only the first template in the list will be used
        random_state: an optional random state for the sampling of the template

    Returns:
        list: A dict containing the prompt and its expected label.
    """
    # Sampling tempates
    if force_template:
        template = MCQ_TEMPLATES[0]
    else:
        generator = random_state if random_state is not None else np.random
        template = MCQ_TEMPLATES[generator.randint(0, len(MCQ_TEMPLATES))]

    # Options bloc
    options_bloc = "OPTIONS:"
    for option_txt, option_label in zip(
        sample["choices"]["text"], sample["choices"]["label"]
    ):
        options_bloc += f"\n- {option_label}: {option_txt}"

    # Formatting
    return {
        "prompt": template.format(question=sample["question"], options_=options_bloc)
        + "\n",
        "label": str(sample["answerKey"]),
        "possible_labels": sample["choices"]["label"],
        "label_status": DATASET_BARE_LABEL,
    }


def format_mmlu(
    sample: dict,
    force_template: bool = False,
    random_state: t.Optional[np.random.RandomState] = None,
) -> dict:
    """Format prompt from MMLU dataset.

    Args:
        samples: A list of samples from the ARC dataset
        force_template: if True, only the first template in the list will be used
        random_state: an optional random state for the sampling of the template

    Returns:
        list: A list of tuples (prompt, label) containing the prompt and its expected label.
    """

    new_sample = {
        "question": sample["question"],
        "choices": {
            "text": sample["choices"],
            "label": ["0", "1", "2", "3"],
        },
        "answerKey": str(sample["answer"]),
    }

    return format_arc(
        new_sample, force_template=force_template, random_state=random_state
    )


def format_ethics(
    sample: dict,
    force_template: bool = False,
    random_state: t.Optional[np.random.RandomState] = None,
) -> dict:
    """Format prompt from ETHICS dataset.

    Args:
        samples: A samples from the ARC dataset
        force_template: if True, only the first template in the list will be used
        random_state: an optional random state for the sampling of the template

    Returns:
        list: A tuples (prompt, label) containing the prompt and its expected label.
    """

    # Template
    if sample["input"][:4].lower() == "aita" or sample["input"][:5].lower() == "wibta":
        templates_list = ETHICS_TEMPLATES_AITA
    else:
        templates_list = ETHICS_TEMPLATES

    if force_template:
        template = templates_list[0]
    else:
        generator = random_state if random_state is not None else np.random
        idx = generator.randint(0, len(templates_list))
        template = templates_list[idx]

    # Formatting
    return {
        "prompt": template.format(input=sample["input"]) + "\n",
        "label": str(sample["label"]),
        "possible_labels": ["0", "1"],
        "label_status": DATASET_BARE_LABEL,
    }
