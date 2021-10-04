# type: ignore
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Tuple

import tiledb


def batch_slice_fn(
    array: tiledb.DenseArray, batch_size: int, offset: int
) -> tiledb.DenseArray:
    return array[offset : offset + batch_size]


def run_io_tasks_in_parallel(
    arrays: Tuple[tiledb.DenseArray], batch_size: int, offset: int
):
    """
    Creates a Threadpool of 2 workers and runs the batch_slice_func in parallel
    param: arrays: The arrays x,y to be sliced in parallel
    param: batch_size: The size of the batch to be sliced
    param: offset: The index of x,y from the start of the arrays
    return: Futures containing the sliced arrays of the parallel runs
    """
    batch_slice = partial(batch_slice_fn, batch_size=batch_size, offset=offset)
    with ThreadPoolExecutor(max_workers=2) as executor:
        running_tasks = executor.map(batch_slice, arrays)
    return running_tasks
