import pytest

from grokking_llm import main


def test_hello_world(sample_args):
    assert main(sample_args) == sample_args
