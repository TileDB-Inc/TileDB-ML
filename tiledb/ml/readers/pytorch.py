"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

import itertools as it
import math
import random
from typing import Iterable, Iterator, Optional, Sequence, TypeVar

import numpy as np
import torch

import tiledb

from ._batch_utils import SparseTileDBTensorGenerator, tensor_generator


def PyTorchTileDBDataLoader(
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    batch_size: int,
    buffer_bytes: Optional[int] = None,
    shuffle_buffer_size: int = 0,
    x_attrs: Sequence[str] = (),
    y_attrs: Sequence[str] = (),
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Return a DataLoader for loading data from TileDB arrays.

    :param x_array: TileDB array of the features.
    :param y_array: TileDB array of the labels.
    :param batch_size: Size of each batch.
    :param buffer_bytes: Maximum size (in bytes) of memory to allocate for reading
        from each array (default=`tiledb.default_ctx().config()["sm.memory_budget"]`).
    :param shuffle_buffer_size: Number of elements from which this dataset will sample.
    :param x_attrs: Attribute names of x_array.
    :param y_attrs: Attribute names of y_array.
    :param num_workers: how many subprocesses to use for data loading
    """
    dataset = PyTorchTileDBDataset(
        x_array, y_array, buffer_bytes, shuffle_buffer_size, x_attrs, y_attrs
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )


class PyTorchTileDBDataset(torch.utils.data.IterableDataset[Sequence[torch.Tensor]]):
    def __init__(
        self,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        buffer_bytes: Optional[int] = None,
        shuffle_buffer_size: int = 0,
        x_attrs: Sequence[str] = (),
        y_attrs: Sequence[str] = (),
    ):
        super().__init__()
        rows: int = x_array.shape[0]
        if rows != y_array.shape[0]:
            raise ValueError("X and Y arrays must have the same number of rows")

        self._rows = rows
        self._shuffle_buffer_size = shuffle_buffer_size
        self._generator_kwargs = dict(
            x_array=x_array,
            y_array=y_array,
            buffer_bytes=buffer_bytes,
            x_attrs=x_attrs,
            y_attrs=y_attrs,
            sparse_tensor_generator_cls=PyTorchSparseTileDBTensorGenerator,
        )

    def __iter__(self) -> Iterator[Sequence[torch.Tensor]]:
        kwargs = self._generator_kwargs
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            for array_key in "x_array", "y_array":
                if isinstance(kwargs[array_key], tiledb.SparseArray):
                    raise NotImplementedError(
                        "https://github.com/pytorch/pytorch/issues/20248"
                    )
            per_worker = int(math.ceil(self._rows / worker_info.num_workers))
            start_offset = worker_info.id * per_worker
            stop_offset = min(start_offset + per_worker, self._rows)
            kwargs = dict(start_offset=start_offset, stop_offset=stop_offset, **kwargs)

        rows: Iterator[Sequence[torch.Tensor]] = (
            row
            for batch_tensors in tensor_generator(**kwargs)
            for row in zip(*batch_tensors)
        )
        if self._shuffle_buffer_size > 0:
            rows = iter_shuffled(rows, self._shuffle_buffer_size)
        return rows


T = TypeVar("T")


def iter_shuffled(iterable: Iterable[T], buffer_size: int) -> Iterator[T]:
    """
    Shuffle the given iterable with a buffer.

    The buffer with `buffer_size` is filled with elements from the iterable first.
    Then, each item will be yielded from the buffer by reservoir sampling via iterator.

    """
    iterator = iter(iterable)
    buffer = list(it.islice(iterator, buffer_size))
    randrange = random.randrange
    for x in iterator:
        idx = randrange(0, buffer_size)
        yield buffer[idx]
        buffer[idx] = x
    random.shuffle(buffer)
    while buffer:
        yield buffer.pop()


class PyTorchSparseTileDBTensorGenerator(SparseTileDBTensorGenerator[torch.Tensor]):
    @staticmethod
    def _tensor_from_coo(
        data: np.ndarray,
        coords: np.ndarray,
        dense_shape: Sequence[int],
        dtype: np.dtype,
    ) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            torch.tensor(coords).t(), data, dense_shape, requires_grad=False
        )
