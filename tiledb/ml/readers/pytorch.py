"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

import itertools
import random
from operator import methodcaller
from typing import Callable, Iterable, Iterator, Sequence, Tuple, TypeVar, Union

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
from tiledb.ml.readers.types import ArrayParams

from ._tensor_schema import DenseTensorSchema, SparseTensorSchema, TensorSchema

TensorLikeSequence = Union[
    Sequence[np.ndarray], Sequence[sparse.COO], Sequence[scipy.sparse.csr_matrix]
]
TensorLikeOrSequence = Union[
    np.ndarray, sparse.COO, scipy.sparse.csr_matrix, TensorLikeSequence
]
XY = Tuple[TensorLikeOrSequence, TensorLikeOrSequence]


def PyTorchTileDBDataLoader(
    x_params: ArrayParams,
    y_params: ArrayParams,
    *,
    batch_size: int,
    shuffle_buffer_size: int = 0,
    prefetch: int = 2,
    num_workers: int = 0,
    csr: bool = True,
) -> torch.utils.data.DataLoader:
    """Return a DataLoader for loading data from TileDB arrays.

    :param x_params: TileDB ArrayParams of the features.
    :param y_params: TileDB ArrayParams of the labels.
    :param batch_size: Size of each batch.
    :param shuffle_buffer_size: Number of elements from which this dataset will sample.
    :param prefetch: Number of samples loaded in advance by each worker. Not applicable
        (and should not be given) when `num_workers` is 0.
    :param num_workers: how many subprocesses to use for data loading. 0 means that the
        data will be loaded in the main process. Note: when `num_workers` > 1
        yielded batches may be shuffled even if `shuffle_buffer_size` is zero.
    :param csr: For sparse 2D arrays, whether to return CSR tensors instead of COO.
    """
    x_schema = _get_tensor_schema(x_params)
    y_schema = _get_tensor_schema(y_params)
    if not x_schema.key_range.equal_values(y_schema.key_range):
        raise ValueError(
            f"X and Y arrays have different key range: {x_schema.key_range} != {y_schema.key_range}"
        )

    return torch.utils.data.DataLoader(
        dataset=_PyTorchTileDBDataset(
            x_schema=x_schema,
            y_schema=y_schema,
            shuffle_buffer_size=shuffle_buffer_size,
        ),
        batch_size=batch_size,
        prefetch_factor=prefetch,
        num_workers=num_workers,
        worker_init_fn=_worker_init,
        collate_fn=_CompositeCollator(
            _get_tensor_collator(x_params.array, csr, len(x_schema.fields)),
            _get_tensor_collator(y_params.array, csr, len(y_schema.fields)),
        ),
    )


class _PyTorchTileDBDataset(torch.utils.data.IterableDataset[XY]):
    def __init__(
        self,
        x_schema: TensorSchema,
        y_schema: TensorSchema,
        shuffle_buffer_size: int = 0,
    ):
        super().__init__()
        self.x_schema = x_schema
        self.y_schema = y_schema
        self.key_range = x_schema.key_range
        self._shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self) -> Iterator[XY]:
        rows: Iterator[XY] = zip(
            self._iter_rows(self.x_schema), self._iter_rows(self.y_schema)
        )
        if self._shuffle_buffer_size > 0:
            rows = _iter_shuffled(rows, self._shuffle_buffer_size)
        return rows

    def _iter_rows(self, schema: TensorSchema) -> Iterator[TensorLikeOrSequence]:
        max_weight = schema.max_partition_weight
        key_subranges = self.key_range.partition_by_weight(max_weight)
        batches: Iterable[TensorLikeOrSequence] = schema.iter_tensors(key_subranges)
        if len(schema.fields) == 1:
            return (tensor for batch in batches for tensor in batch)
        else:
            return (tensors for batch in batches for tensors in zip(*batch))


def _worker_init(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.x_schema.sparse or dataset.y_schema.sparse:
        raise NotImplementedError("https://github.com/pytorch/pytorch/issues/20248")
    key_ranges = list(dataset.key_range.partition_by_count(worker_info.num_workers))
    dataset.key_range = key_ranges[worker_id]


def _get_tensor_schema(array_params: ArrayParams) -> TensorSchema:
    if not array_params.array.schema.sparse:
        return DenseTensorSchema(array_params)
    elif array_params.array.ndim == 2:
        return SparseTensorSchema(array_params, methodcaller("tocsr"))
    else:
        return SparseTensorSchema(array_params)


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
