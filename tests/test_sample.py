import os

from grokking_llm.utils import paths


def test_configs():
    # Check that these entries exist
    paths.main_cfg_object
    paths.root
    paths.hf_home
    paths.data
    paths.output
    paths.logs


def test_hf_home_dir():
    assert paths.hf_home.exists()
    assert os.environ["TRANSFORMERS_CACHE"] == str(paths.hf_home)
