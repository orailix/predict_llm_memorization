import os

from grokking_llm.utils import paths


def test_configs():
    # Check that these entries exist
    paths.main_cfg_object
    paths.root
    paths.hf_cache
    paths.data
    paths.output
    paths.logs


def test_hf_cache_dir():
    assert paths.hf_cache.exists()
    assert os.environ["TRANSFORMERS_CACHE"] == str(paths.hf_cache)
