from grokking_llm.utils import paths


def test_configs():
    # Check that these entries exist
    paths.main_cfg_object
    paths.root
    paths.model_hub
    paths.data
    paths.output
    paths.logs
