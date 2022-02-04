import multiprocessing
import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def set_multiprocessing_start_method():
    if os.name == "posix":
        multiprocessing.set_start_method("forkserver")
