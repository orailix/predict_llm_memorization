"""
`grokking_llm`

Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
Apache Licence v2.0.
"""

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
    assert paths.hf_home.is_dir()
    assert os.environ["HF_HOME"] == str(paths.hf_home)
