"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import os

from loguru import logger

from . import paths

# Model names
MOD_MISTRAL_7B = "mistralai/Mistral-7B-v0.1"
MOD_MISTRAL_7B_CHAT = "mistralai/Mistral-7B-Instruct-v0.1"
MOD_LLAMA_7B = "meta-llama/Llama-2-7b-hf"
MOD_LLAMA_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"

# Dataset names
DS_ARC = "ai2_arc"
DS_ETHICS = "hendrycks/ethics"
DS_MMLU = "cais/mmlu"
DS_E2E = "e2e_nlg"
DS_VIGGO = "GEM/viggo"

# Defining cache location
logger.info(f"Using HuggingFace model hub cache: {paths.hf_cache}")
logger.info(f"To work correcly, this module should be imported before any HuggingFace module.")
paths.hf_cache.mkdir(exist_ok=True, parents=True)
for env_var in ["TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"]:
    os.environ[env_var] = str(paths.hf_cache)
    logger.debug(f"Exporting {env_var}={str(paths.hf_cache)}")


def download_hf(name: str, repo_type: str) -> None:
    """Downloads a repository and caches it.
    
    :name: Name of the repo
    :repo_type: "model" or "dataset".
    """

    import huggingface_hub

    logger.info(f"Downloading HF repository: {name}")
    huggingface_hub.snapshot_download(name, cache_dir=paths.hf_cache, repo_type=repo_type)


if __name__ == "__main__":
    # Models
    download_hf(MOD_LLAMA_7B, repo_type="model")
    download_hf(MOD_LLAMA_7B_CHAT, repo_type="model")
    download_hf(MOD_MISTRAL_7B, repo_type="model")
    download_hf(MOD_MISTRAL_7B_CHAT, repo_type="model")
    
    # Datasets
    download_hf(DS_ETHICS, repo_type="dataset")
    download_hf(DS_ARC, repo_type="dataset")
    download_hf(DS_E2E, repo_type="dataset")
    download_hf(DS_MMLU, repo_type="dataset")
    download_hf(DS_VIGGO, repo_type="dataset")