from concurrent.futures import Executor
from typing import Iterator, Sequence

import numpy as np

import tiledb


def parallel_slice(
    executor: Executor, arrays: Sequence[tiledb.Array], batch_size: int, offset: int
) -> Iterator[np.array]:
    """
    Slice multiples arrays in parallel

    param: arrays: The arrays x,y to be sliced in parallel
    param: batch_size: The size of the batch to be sliced
    param: offset: The index of x,y from the start of the arrays
    return: Futures containing the sliced arrays of the parallel runs
    """
    return executor.map(lambda array: array[offset : offset + batch_size], arrays)
