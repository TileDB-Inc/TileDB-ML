# type: ignore
from functools import partial
from typing import Tuple

import tiledb


def run_io_tasks_in_parallel(
    executor, arrays: Tuple[tiledb.Array], batch_size: int, offset: int
):
    """
    Runs the batch_slice_func in parallel
    param: arrays: The arrays x,y to be sliced in parallel
    param: batch_size: The size of the batch to be sliced
    param: offset: The index of x,y from the start of the arrays
    return: Futures containing the sliced arrays of the parallel runs
    """

    batch_slice = partial(batch_slice_fn, batch_size=batch_size, offset=offset)
    running_tasks = executor.map(batch_slice, arrays)
    return running_tasks


def batch_slice_fn(array: tiledb.Array, batch_size: int, offset: int) -> tiledb.Array:
    return array[offset : offset + batch_size]
