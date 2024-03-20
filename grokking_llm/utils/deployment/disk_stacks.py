# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from pathlib import Path

from filelock import FileLock


class DiskStack:
    """Class used to represent a stack on disk."""

    def __init__(self, path: t.Union[str, Path]):

        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            path.write_text("")

        # Containers
        self.path = path
        self.lock_path = path.with_suffix(".lock")
        self.lock = FileLock(self.lock_path)

    def size(self):
        with self.lock.acquire():
            content = self.path.read_text()
            return len([item for item in content.split("\n") if item != ""])

    def __len__(self):
        return self.size()

    def push(self, item: str) -> None:
        if not isinstance(item, str):
            item = str(item)

        if "\n" in item:
            raise ValueError("You cannot push items with '\\n' is a DiskStack.")

        with self.lock.acquire():
            self.path.write_text(self.path.read_text() + f"{item}\n")

    def push_chunk(self, items: t.Iterable[str]) -> None:
        to_push = ""
        for item in items:
            if not isinstance(item, str):
                item = str(item)

            if "\n" in item:
                raise ValueError("You cannot push items with '\\n' is a DiskStack.")

            to_push += f"{item}\n"

        with self.lock.acquire():
            self.path.write_text(self.path.read_text() + to_push)

    def pop(self) -> str:
        with self.lock.acquire():
            content = self.path.read_text()
            items = [item for item in content.split("\n") if item != ""]

            if len(items) == 0:
                raise ValueError("Empty stack.")

            last_element = items[-1]
            self.path.write_text("\n".join(items[:-1]) + "\n")
            return last_element

    def pop_all(self) -> t.List[str]:
        with self.lock.acquire():
            content = self.path.read_text()
            items = [item for item in content.split("\n") if item != ""]

            if len(items) == 0:
                raise ValueError("Empty stack.")

            self.path.write_text("")
            return items[-1::-1]

    def empty(self):
        return len(self) == 0

    def top(self):
        with self.lock.acquire():
            content = self.path.read_text()
            items = [item for item in content.split("\n") if item != ""]
            if len(items) == 0:
                raise ValueError("Empty stack.")
            else:
                return items[-1]

    def peek(self):
        return self.top()

    def reset(self):
        with self.lock.acquire():
            self.path.write_text("")
