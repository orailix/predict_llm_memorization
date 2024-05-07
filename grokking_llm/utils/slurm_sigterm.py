# `grokking_llm`

# Copyright 2023-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import signal


class GotSigterm(Exception):
    pass


def signal_handler(signum, frame):
    raise GotSigterm("Got SIGTERM signal.")


signal.signal(signal.SIGTERM, signal_handler)
