from concurrent.futures import Future, ThreadPoolExecutor
from typing import List

import numpy as np

import tiledb


def batch_slice_func(
    array: tiledb.DenseArray, batch_size: int, offset: int
) -> tiledb.DenseArray:
    """
    Slice array on batches size segments given an offset
    param: array: The array to be sliced
    param: batch_size: The size of the batch to be sliced
    param: offset: The index of x,y from the start of the arrays
    return: Sliced array
    """
    return array[offset : offset + batch_size]


def run_io_tasks_in_parallel(
    arrays: List[tiledb.DenseArray], batch_size: int, offset: int
) -> List[Future[np.ndarray]]:
    """
    Creates a Threadpool of 2 workers and runs the batch_slice_func in parallel
    param: arrays: The arrays x,y to be sliced in parallel
    param: batch_size: The size of the batch to be sliced
    param: offset: The index of x,y from the start of the arrays
    return: Futures containing the sliced arrays of the parallel runs
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        running_tasks = [
            executor.submit(batch_slice_func, array, batch_size, offset)
            for array in arrays
        ]
    return running_tasks
