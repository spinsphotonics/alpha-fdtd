# SPDX-License-Identifier: MIT
"""Markers: ``gpu`` tests need one supported GPU, ``gpu2`` tests two.  Without them those tests are skipped (reported
as skipped, with the reason), never passed."""
import functools
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests", "oracle"))
sys.path.insert(0, os.path.join(ROOT, "examples"))


@functools.lru_cache(None)
def gpus():
    """(count, reason): 0, 1 or 2 = how many GPUs a plan can use here (tried, not guessed)."""
    import alpha_fdtd as af
    try:
        af.Plan((16, 16, 16)).close()
    except af.Error as e:
        return 0, str(e)
    try:
        af.Plan((16, 97, 16), ngpus=2).close()
    except af.Error as e:
        return 1, str(e)
    return 2, ""


@pytest.fixture(autouse=True)
def _needs_gpus(request):
    need = 2 if request.node.get_closest_marker("gpu2") else 1 if request.node.get_closest_marker("gpu") else 0
    if need:
        have, why = gpus()
        if have < need:
            pytest.skip(f"needs {need} GPU(s): {why}")
