# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import signal


class GotEndSignal(Exception):
    pass


def signal_sigterm(signum, frame):
    raise GotEndSignal("Got SIGTERM signal.")


def signal_sigint(signum, frame):
    raise GotEndSignal("Got SIGINT signal.")


def signal_sigusr1(signum, frame):
    raise GotEndSignal("Got SIGUSR1 signal.")


signal.signal(signal.SIGTERM, signal_sigterm)
signal.signal(signal.SIGINT, signal_sigint)
signal.signal(signal.SIGUSR1, signal_sigusr1)
