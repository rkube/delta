# -*- Encoding: UTF-8 -*-

"""Verify that the doggo-dataloader works correctly.

"""

try:
    import mock
except ImportError:
    from unittest import mock

from tests.conftest import doggo_conf
import numpy as np


def test_dataloader_doggo(doggo_conf):
    import sys
    import os
    sys.path.append(os.path.abspath("delta"))
    from delta.sources.loader_doggo import loader_doggo
    
    dognostic = loader_doggo(doggo_conf)

    for batch in dognostic.batch_generator():
        print(type(batch))
            # batch.shape)
        # print()

    assert False
    return True

# End of file test_dataloader_doggo.py