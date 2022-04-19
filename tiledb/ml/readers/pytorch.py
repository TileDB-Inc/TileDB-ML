"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

import itertools as it
import math
import operator
import random
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, TypeVar, Union

import numpy as np
import scipy.sparse
import sparse
import torch

try:
    # torch >= 1.10
    sparse_csr_tensor = torch.sparse_csr_tensor
except AttributeError:
    # torch < 1.10
    sparse_csr_tensor = torch._sparse_csr_tensor


import tiledb

from ._buffer_utils import get_attr_names, get_buffer_size
from ._tensor_gen import TileDBNumpyGenerator, TileDBSparseGenerator

TensorSequence = Sequence[Union[np.ndarray, sparse.COO, scipy.sparse.csr_matrix]]
TensorGenerator = Union[TileDBNumpyGenerator, TileDBSparseGenerator]


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
    csr: bool = True,
) -> torch.utils.data.DataLoader:
    """Return a DataLoader for loading data from TileDB arrays.

    :param x_array: TileDB array of the features.
    :param y_array: TileDB array of the labels.
    :param batch_size: Size of each batch.
    :param buffer_bytes: Maximum size (in bytes) of memory to allocate for reading
        from each array (default=`tiledb.default_ctx().config()["sm.memory_budget"]`).
    :param shuffle_buffer_size: Number of elements from which this dataset will sample.
    :param prefetch: Number of samples loaded in advance by each worker. Not applicable
        (and should not be given) when `num_workers` is 0.
    :param x_attrs: Attribute names of x_array.
    :param y_attrs: Attribute names of y_array.
    :param num_workers: how many subprocesses to use for data loading. 0 means that the
        data will be loaded in the main process. Note: yielded batches may be shuffled
        even if `shuffle_buffer_size` is zero when `num_workers` > 1.
    :param csr: For sparse 2D arrays, whether to return CSR tensors instead of COO.
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
        collate_fn=_CompositeCollator(
            it.chain(
                it.repeat(_get_tensor_collator(x_array, csr), len(x_attrs)),
                it.repeat(_get_tensor_collator(y_array, csr), len(y_attrs)),
            )
        ),
    )


class PyTorchTileDBDataset(torch.utils.data.IterableDataset[TensorSequence]):
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

        self._x_gen = _get_tensor_generator(x_array, x_attrs)
        self._y_gen = _get_tensor_generator(y_array, y_attrs)
        self._x_buffer_size = get_buffer_size(x_array, x_attrs, buffer_bytes)
        self._y_buffer_size = get_buffer_size(y_array, y_attrs, buffer_bytes)
        self._rows = rows
        self._shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self) -> Iterator[TensorSequence]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            for gen in self._x_gen, self._y_gen:
                if isinstance(gen, TileDBSparseGenerator):
                    raise NotImplementedError(
                        "https://github.com/pytorch/pytorch/issues/20248"
                    )
            per_worker = int(math.ceil(self._rows / worker_info.num_workers))
            start = worker_info.id * per_worker
            stop = min(start + per_worker, self._rows)
        else:
            start = 0
            stop = self._rows

        def iter_rows(
            gen: TensorGenerator, buffer_size: int
        ) -> Iterator[TensorSequence]:
            for batch_tensors in gen.iter_tensors(buffer_size, start, stop):
                for row in zip(*batch_tensors):
                    yield row

        rows: Iterator[TensorSequence] = (
            (*x_row, *y_row)
            for x_row, y_row in zip(
                iter_rows(self._x_gen, self._x_buffer_size),
                iter_rows(self._y_gen, self._y_buffer_size),
            )
        )
        if self._shuffle_buffer_size > 0:
            rows = _iter_shuffled(rows, self._shuffle_buffer_size)
        return rows


def _get_tensor_generator(array: tiledb.Array, attrs: Sequence[str]) -> TensorGenerator:
    if not array.schema.sparse:
        return TileDBNumpyGenerator(array, attrs)
    elif array.ndim == 2:
        return TileDBSparseGenerator(array, attrs, operator.methodcaller("tocsr"))
    else:
        return TileDBSparseGenerator(array, attrs, lambda x: x)


def _get_tensor_collator(
    array: tiledb.Array, csr: bool
) -> Callable[[TensorSequence], torch.Tensor]:
    if not array.schema.sparse:
        return _ndarray_collate
    elif array.ndim != 2:
        return _sparse_coo_collate
    elif csr:
        return _csr_collate
    else:
        return _csr_to_coo_collate


class _CompositeCollator:
    """
    A callable for collating "rows" of data into Tensors.

    Each data "column" is collated to a torch.Tensor by a different collator function.
    Finally, the collated columns are returned as a sequence of torch.Tensors.
    """

    def __init__(self, collators: Iterable[Callable[[Sequence[Any]], torch.Tensor]]):
        self._collators = tuple(collators)

    def __call__(self, rows: Sequence[Sequence[Any]]) -> Sequence[torch.Tensor]:
        columns = tuple(zip(*rows))
        assert len(columns) == len(self._collators)
        return [collator(column) for collator, column in zip(self._collators, columns)]


def _ndarray_collate(arrays: Sequence[np.ndarray]) -> torch.Tensor:
    """Collate multiple Numpy arrays to a torch.Tensor with strided layout."""
    # Specialized version of default_collate for collating Numpy arrays
    # Faster than `torch.as_tensor(arrays)` (https://github.com/pytorch/pytorch/pull/51731)
    # and `torch.stack([torch.as_tensor(array) for array in arrays]])`
    return torch.from_numpy(np.stack(arrays))


def _sparse_coo_collate(arrays: Sequence[sparse.COO]) -> torch.Tensor:
    """Collate multiple sparse.COO arrays to a torch.Tensor with sparse_coo layout."""
    stacked = sparse.stack(arrays)
    return torch.sparse_coo_tensor(stacked.coords, stacked.data, stacked.shape)


def _csr_to_coo_collate(arrays: Sequence[scipy.sparse.csr_matrix]) -> torch.Tensor:
    """Collate multiple Scipy CSR matrices to a torch.Tensor with sparse_coo layout."""
    stacked = scipy.sparse.vstack(arrays).tocoo()
    coords = np.stack((stacked.row, stacked.col))
    return torch.sparse_coo_tensor(coords, stacked.data, stacked.shape)


def _csr_collate(arrays: Sequence[scipy.sparse.csr_matrix]) -> torch.Tensor:
    """Collate multiple Scipy CSR matrices to a torch.Tensor with sparse_csr layout."""
    stacked = scipy.sparse.vstack(arrays)
    return sparse_csr_tensor(
        torch.from_numpy(stacked.indptr),
        torch.from_numpy(stacked.indices),
        stacked.data,
        stacked.shape,
    )


_T = TypeVar("_T")


def _iter_shuffled(iterable: Iterable[_T], buffer_size: int) -> Iterator[_T]:
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
