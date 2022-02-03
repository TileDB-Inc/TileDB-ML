import multiprocessing

import pytest


@pytest.fixture(autouse=True, scope="session")
def ensure_multiprocessing_spawn():
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn")
