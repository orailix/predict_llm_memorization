"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import typing as t

import numpy as np

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
    samples: t.List[dict], force_template: bool = False
) -> t.List[t.Tuple[str, str]]:
    """Format prompt from ARC dataset.

    Args:
        samples: A list of samples from the ARC dataset
        force_template: if True, only the first template in the list will be used

    Returns:
        list: A list of tuples (prompt, label) containing the prompt and its expected label.
    """
    # Sampling tempates
    if force_template:
        templates = [0 for _ in samples]
    else:
        templates = np.random.randint(0, len(MCQ_TEMPLATES), size=len(samples))

    result = []
    for sample, tpl in zip(samples, templates):
        # Options bloc
        options_bloc = "OPTIONS:"
        for option_txt, option_label in zip(
            sample["choices"]["text"], sample["choices"]["label"]
        ):
            options_bloc += f"\n- {option_label}: {option_txt}"

        # Formatting
        result.append(
            (
                MCQ_TEMPLATES[tpl].format(
                    question=sample["question"], options_=options_bloc
                )
                + "\n",
                sample["answerKey"],
            )
        )

        return result


def format_mmlu(
    samples: t.List[dict], force_template: bool = False
) -> t.List[t.Tuple[str, str]]:
    """Format prompt from MMLU dataset.

    Args:
        samples: A list of samples from the ARC dataset
        force_template: if True, only the first template in the list will be used

    Returns:
        list: A list of tuples (prompt, label) containing the prompt and its expected label.
    """

    new_samples = [
        {
            "question": sample["question"],
            "choices": {
                "text": sample["choices"],
                "label": ["0", "1", "2", "3"],
            },
            "answerKey": str(sample["answer"]),
        }
        for sample in samples
    ]
    return format_arc(new_samples, force_template=force_template)


def format_ethics(
    samples: t.List[dict], force_template: bool = False
) -> t.List[t.Tuple[str, str]]:
    """Format prompt from ETHICS dataset.

    Args:
        samples: A list of samples from the ARC dataset
        force_template: if True, only the first template in the list will be used

    Returns:
        list: A list of tuples (prompt, label) containing the prompt and its expected label.
    """

    result = []
    for sample in samples:
        # Template
        if (
            sample["input"][:4].lower() == "aita"
            or sample["input"][:5].lower() == "wibta"
        ):
            templates_list = ETHICS_TEMPLATES_AITA
        else:
            templates_list = ETHICS_TEMPLATES

        if force_template:
            template = templates_list[0]
        else:
            idx = np.random.randint(0, len(templates_list))
            template = templates_list[idx]

        # Formatting
        result.append(
            (
                template.format(input=sample["input"]) + "\n",
                str(sample["label"]),
            )
        )

        return result
