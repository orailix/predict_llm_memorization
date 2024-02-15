# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import pytest

from grokking_llm.deploy import DiskStack
from grokking_llm.utils import paths


def test_disk_stack():

    # Init
    stack_path = paths.output / "stask_test"

    # Cleaning
    if stack_path.is_file():
        stack_path.unlink()

    # Init stack
    stack = DiskStack(str(stack_path))
    stack = DiskStack(stack_path)

    # Checks
    assert stack.size() == 0
    assert len(stack) == 0
    assert stack.empty()

    # Adding one element
    stack.push("bonjour")
    assert stack.size() == 1
    assert len(stack) == 1
    assert not stack.empty()

    # Adding one element
    stack.push("le monde")
    assert stack.size() == 2
    assert len(stack) == 2
    assert not stack.empty()

    # Adding an illegal element
    with pytest.raises(ValueError):
        stack.push("Hello\nworld!")
    assert stack.size() == 2
    assert len(stack) == 2
    assert not stack.empty()

    # Peeking
    assert stack.peek() == "le monde"
    assert stack.top() == "le monde"
    assert stack.size() == 2
    assert len(stack) == 2
    assert not stack.empty()

    # Poping
    assert stack.pop() == "le monde"
    assert stack.size() == 1
    assert len(stack) == 1
    assert not stack.empty()

    # Poping again
    assert stack.pop() == "bonjour"
    assert stack.size() == 0
    assert len(stack) == 0
    assert stack.empty()

    # Adding a chunk
    stack.push("bonjour")
    stack.push_chunk(["le", "monde"])
    assert stack.size() == 3
    assert len(stack) == 3
    assert not stack.empty()

    # Adding an illegal chunk
    with pytest.raises(ValueError):
        stack.push_chunk(["bonjour\n", "le", "monde"])
    assert stack.size() == 3
    assert len(stack) == 3
    assert not stack.empty()

    # Adding poping one element
    assert stack.pop() == "monde"
    assert stack.size() == 2
    assert len(stack) == 2
    assert not stack.empty()

    # Poping all
    assert stack.pop_all() == ["le", "bonjour"]
    assert stack.size() == 0
    assert len(stack) == 0
    assert stack.empty()

    # Cleaning
    stack.path.unlink()
    stack.lock_path.unlink()
