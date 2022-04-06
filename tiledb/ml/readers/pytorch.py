"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

import itertools as it
import math
import random
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, TypeVar

import numpy as np
import sparse
import torch

import tiledb

from ._buffer_utils import get_attr_names, get_buffer_size
from ._tensor_gen import TileDBSparseTensorGenerator, tensor_generator


def PyTorchTileDBDataLoader(
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    *,
    batch_size: int,
    buffer_bytes: Optional[int] = None,
    shuffle_buffer_size: int = 0,
    prefetch: int = 2,
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
    :param prefetch: Number of samples loaded in advance by each worker. Not applicable
        (and should not be given) when `num_workers` is 0.
    :param shuffle_buffer_size: Number of elements from which this dataset will sample.
    :param x_attrs: Attribute names of x_array.
    :param y_attrs: Attribute names of y_array.
    :param num_workers: how many subprocesses to use for data loading
    """
    x_schema = x_array.schema
    y_schema = y_array.schema
    if not x_attrs:
        x_attrs = get_attr_names(x_schema)
    if not y_attrs:
        y_attrs = get_attr_names(y_schema)

    return torch.utils.data.DataLoader(
        dataset=PyTorchTileDBDataset(
            x_array, y_array, buffer_bytes, shuffle_buffer_size, x_attrs, y_attrs
        ),
        batch_size=batch_size,
        prefetch_factor=prefetch,
        num_workers=num_workers,
        collate_fn=CompositeCollator(
            (
                np_arrays_collate
                if not is_sparse
                else torch.utils.data.dataloader.default_collate
            )
            for is_sparse in it.chain(
                it.repeat(x_schema.sparse, len(x_attrs)),
                it.repeat(y_schema.sparse, len(y_attrs)),
            )
        ),
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

        if not x_attrs:
            x_attrs = get_attr_names(x_array.schema)
        if not y_attrs:
            y_attrs = get_attr_names(y_array.schema)

        self._rows = rows
        self._shuffle_buffer_size = shuffle_buffer_size
        self._generator_kwargs = dict(
            x_array=x_array,
            y_array=y_array,
            x_buffer_size=get_buffer_size(x_array, x_attrs, buffer_bytes),
            y_buffer_size=get_buffer_size(y_array, y_attrs, buffer_bytes),
            x_attrs=x_attrs,
            y_attrs=y_attrs,
            sparse_generator_cls=PyTorchSparseTensorGenerator,
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


class CompositeCollator:
    """
    A callable for collating "rows" of data into Tensors.

    Each data "column" is collated to a torch.Tensor by a different collator function.
    Finally, the collated columns are returned as a sequence of torch.Tensors.
    """

    def __init__(self, collators: Iterable[Callable[[Sequence[Any]], torch.Tensor]]):
        self._collators = tuple(collators)

    def __call__(self, rows: Sequence[Sequence[Any]]) -> Sequence[torch.Tensor]:
        columns = list(zip(*rows))
        assert len(columns) == len(self._collators)
        return [collator(column) for collator, column in zip(self._collators, columns)]


def np_arrays_collate(arrays: Sequence[np.ndarray]) -> torch.Tensor:
    # Specialized version of default_collate for collating Numpy arrays
    # Faster than `torch.as_tensor(arrays)` (https://github.com/pytorch/pytorch/pull/51731)
    # and `torch.stack([torch.as_tensor(array) for array in arrays]])`
    return torch.as_tensor(np.stack(arrays))


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


class PyTorchSparseTensorGenerator(TileDBSparseTensorGenerator[torch.Tensor]):
    @staticmethod
    def _tensor_from_coo(coo: sparse.COO) -> torch.Tensor:
        return torch.sparse_coo_tensor(coo.coords, coo.data, coo.shape)
