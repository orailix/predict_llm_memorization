"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

import configparser
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


# Managing HF home
logger.info(f"Setting env variable HF_HOME={paths.hf_home}")
os.environ["HF_HOME"] = paths.hf_home

# HF online / offline cache
if paths.main_cfg_object["internet"]["offline"] == "false":
    logger.info("Config internet.offline = false")
    offline = False
if paths.main_cfg_object["internet"]["offline"] == "true":
    logger.info("Config internet.offline = true")
    logger.debug("Setting env variable HF_HUB_OFFLINE=1")
    os.environ["HF_HUB_OFFLINE"]="1"
    offline = False
    
# Dealing with HF token
if not offline:
    credentials_cfg = paths.configs / 'credentials.cfg'
    logger.debug(f"Looking for HF token at {credentials_cfg} ...")
    if credentials_cfg.exists():
        credential_cfg = configparser.ConfigParser()
        credential_cfg.read(credentials_cfg)
        token = credentials_cfg["huggingface"]["token"]
        logger.info(f"Exporting env variable HF_TOKEN={token[:3]}{(len(token)-8)*'-'}{token[-5:]}")
        os.environ['HF_TOKEN'] = token
    else:
        logger.warning("No credential found. This may prevent you to use LLaMA models.")

if __name__ == "__main__":
    pass