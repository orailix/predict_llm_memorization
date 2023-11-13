"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import os

from loguru import logger

from . import paths

# Model names
MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
MISTRAL_7B_CHAT = "mistralai/Mistral-7B-Instruct-v0.1"
LLAMA_7B = "meta-llama/Llama-2-7b-hf"
LLAMA_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"

# Defining cache location
logger.info(f"Using HuggingFace model hub cache: {paths.hf_cache}")
paths.hf_cache.mkdir(exist_ok=True, parents=True)
os.environ["TRANSFORMERS_CACHE"] = str(paths.hf_cache)


def download_model(name: str) -> None:
    """Downloads a repository and caches it."""

    import huggingface_hub

    logger.info(f"Downloading HF repository: {name}")
    huggingface_hub.snapshot_download(name, cache_dir=paths.hf_cache)


if __name__ == "__main__":

    download_model(LLAMA_7B)
    download_model(MISTRAL_7B)
