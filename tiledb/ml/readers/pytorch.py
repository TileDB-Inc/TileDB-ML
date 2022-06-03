"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

import itertools
import random
from math import ceil
from operator import methodcaller
from typing import (
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import scipy.sparse
import sparse
import torch

try:
    # torch>=1.10
    sparse_csr_tensor = torch.sparse_csr_tensor
except AttributeError:
    # torch=1.9
    sparse_csr_tensor = torch._sparse_csr_tensor

import tiledb

from ._tensor_gen import TileDBNumpyGenerator, TileDBSparseGenerator
from ._tensor_schema import TensorSchema, iter_slices

TensorLikeSequence = Union[
    Sequence[np.ndarray], Sequence[sparse.COO], Sequence[scipy.sparse.csr_matrix]
]
TensorLikeOrSequence = Union[
    np.ndarray, sparse.COO, scipy.sparse.csr_matrix, TensorLikeSequence
]
XY = Tuple[TensorLikeOrSequence, TensorLikeOrSequence]


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
    x_key_dim: Union[int, str] = 0,
    y_key_dim: Union[int, str] = 0,
    num_workers: int = 0,
    csr: bool = True,
) -> torch.utils.data.DataLoader:
    """Return a DataLoader for loading data from TileDB arrays.

    :param x_array: TileDB array of the features.
    :param y_array: TileDB array of the labels.
    :param batch_size: Size of each batch.
    :param buffer_bytes: Maximum size (in bytes) of memory to allocate for reading from
        each array. This is bounded by the `sm.memory_budget` config parameter of the
        array context for dense arrays and `py.init_buffer_bytes` (or 10 MB if unset) for
        sparse arrays. These bounds are also used as the default memory budget.
    :param shuffle_buffer_size: Number of elements from which this dataset will sample.
    :param prefetch: Number of samples loaded in advance by each worker. Not applicable
        (and should not be given) when `num_workers` is 0.
    :param x_attrs: Attribute and/or dimension names of the x_array to read. Defaults to
        all attributes.
    :param y_attrs: Attribute and/or dimension names of the y_array to read. Defaults to
        all attributes.
    :param x_key_dim: Name or index of the key dimension of x_array.
    :param y_key_dim: Name or index of the key dimension of y_array.
    :param num_workers: how many subprocesses to use for data loading. 0 means that the
        data will be loaded in the main process. Note: yielded batches may be shuffled
        even if `shuffle_buffer_size` is zero when `num_workers` > 1.
    :param csr: For sparse 2D arrays, whether to return CSR tensors instead of COO.
    """
    x_schema = TensorSchema(x_array, x_key_dim, x_attrs)
    y_schema = TensorSchema(y_array, y_key_dim, y_attrs)
    x_schema.ensure_equal_keys(y_schema)
    return torch.utils.data.DataLoader(
        dataset=_PyTorchTileDBDataset(
            x_array=x_array,
            y_array=y_array,
            x_schema=x_schema,
            y_schema=y_schema,
            buffer_bytes=buffer_bytes,
            shuffle_buffer_size=shuffle_buffer_size,
        ),
        batch_size=batch_size,
        prefetch_factor=prefetch,
        num_workers=num_workers,
        worker_init_fn=_worker_init,
        collate_fn=_CompositeCollator(
            _get_tensor_collator(x_array, csr, len(x_schema.fields)),
            _get_tensor_collator(y_array, csr, len(y_schema.fields)),
        ),
    )


class _PyTorchTileDBDataset(torch.utils.data.IterableDataset[XY]):
    def __init__(
        self,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        x_schema: TensorSchema,
        y_schema: TensorSchema,
        buffer_bytes: Optional[int] = None,
        shuffle_buffer_size: int = 0,
    ):
        super().__init__()
        self._x_schema = x_schema
        self._y_schema = y_schema
        self._start_key = self._x_schema.start_key
        self._stop_key = self._x_schema.stop_key
        self._num_keys = self._x_schema.num_keys
        self._x_gen = _get_tensor_generator(x_array, self._x_schema)
        self._y_gen = _get_tensor_generator(y_array, self._y_schema)
        self._buffer_bytes = buffer_bytes
        self._shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self) -> Iterator[XY]:
        rows: Iterator[XY] = zip(self._iter_rows(True), self._iter_rows(False))
        if self._shuffle_buffer_size > 0:
            rows = _iter_shuffled(rows, self._shuffle_buffer_size)
        return rows

    def _iter_rows(self, is_x: bool) -> Iterator[TensorLikeOrSequence]:
        if is_x:
            schema, gen = self._x_schema, self._x_gen
        else:
            schema, gen = self._y_schema, self._y_gen
        key_dim_slices = schema.partition_key_dim(
            self._buffer_bytes, self._start_key, self._stop_key
        )
        if len(schema.fields) == 1:
            return (row for tensor in gen(key_dim_slices) for row in tensor)
        else:
            return (row for tensors in gen(key_dim_slices) for row in zip(*tensors))


def _worker_init(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    for gen in dataset._x_gen, dataset._y_gen:
        if isinstance(gen, TileDBSparseGenerator):
            raise NotImplementedError("https://github.com/pytorch/pytorch/issues/20248")
    per_worker = ceil(dataset._num_keys / worker_info.num_workers)
    partitions = list(iter_slices(dataset._start_key, dataset._stop_key, per_worker))
    assert len(partitions) == worker_info.num_workers
    dataset._start_key = partitions[worker_id].start
    dataset._stop_key = partitions[worker_id].stop


def _get_tensor_generator(
    array: tiledb.Array, schema: TensorSchema
) -> Union[
    TileDBNumpyGenerator,
    TileDBSparseGenerator[sparse.COO],
    TileDBSparseGenerator[scipy.sparse.csr_matrix],
]:
    if not array.schema.sparse:
        return TileDBNumpyGenerator(array, schema)
    elif array.ndim == 2:
        return TileDBSparseGenerator(array, schema, from_coo=methodcaller("tocsr"))
    else:
        return TileDBSparseGenerator(array, schema, from_coo=lambda x: x)


_SingleCollator = Callable[[TensorLikeSequence], torch.Tensor]


class _CompositeCollator:
    """
    A callable for collating "rows" of data by a separate collator for each "column".
    Returns the collated columns collected into a tuple.
    """

    def __init__(self, *collators: _SingleCollator):
        self._collators = collators

    def __call__(self, rows: Sequence[TensorLikeSequence]) -> Sequence[torch.Tensor]:
        columns = tuple(zip(*rows))
        collators = self._collators
        assert len(columns) == len(collators)
        return tuple(collator(column) for collator, column in zip(collators, columns))


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


def _get_tensor_collator(
    array: tiledb.Array, csr: bool, num_fields: int
) -> Union[_SingleCollator, _CompositeCollator]:
    if not array.schema.sparse:
        collator = _ndarray_collate
    elif array.ndim != 2:
        collator = _sparse_coo_collate
    elif csr:
        collator = _csr_collate
    else:
        collator = _csr_to_coo_collate

    if num_fields == 1:
        return collator
    else:
        return _CompositeCollator(*itertools.repeat(collator, num_fields))


_T = TypeVar("_T")


def _iter_shuffled(iterable: Iterable[_T], buffer_size: int) -> Iterator[_T]:
    """
    Shuffle the given iterable with a buffer.

    The buffer with `buffer_size` is filled with elements from the iterable first.
    Then, each item will be yielded from the buffer by reservoir sampling via iterator.

    """
    iterator = iter(iterable)
    buffer = list(itertools.islice(iterator, buffer_size))
    randrange = random.randrange
    for x in iterator:
        idx = randrange(0, buffer_size)
        yield buffer[idx]
        buffer[idx] = x
    random.shuffle(buffer)
    while buffer:
        yield buffer.pop()
