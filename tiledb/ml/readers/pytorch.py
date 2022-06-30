"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

import itertools
import random
from operator import methodcaller
from typing import Callable, Iterable, Iterator, Sequence, Tuple, TypeVar, Union

import numpy as np
import scipy.sparse
import sparse
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    # torch>=1.10
    sparse_csr_tensor = torch.sparse_csr_tensor
except AttributeError:
    # torch=1.9
    sparse_csr_tensor = torch._sparse_csr_tensor

import tiledb

from ._tensor_schema import DenseTensorSchema, SparseTensorSchema, TensorSchema
from .types import ArrayParams

Tensor = Union[np.ndarray, sparse.COO, scipy.sparse.csr_matrix]
TensorSequence = Union[
    Sequence[np.ndarray], Sequence[sparse.COO], Sequence[scipy.sparse.csr_matrix]
]
TensorOrSequence = Union[Tensor, TensorSequence]
OneOrMoreTensorsOrSequences = Union[TensorOrSequence, Tuple[TensorOrSequence, ...]]


def PyTorchTileDBDataLoader(
    *array_params: ArrayParams,
    batch_size: int,
    shuffle_buffer_size: int = 0,
    prefetch: int = 2,
    num_workers: int = 0,
    csr: bool = True,
) -> DataLoader:
    """Return a DataLoader for loading data from TileDB arrays.

    :param array_params: One or more `ArrayParams` instances, one per TileDB array.
    :param batch_size: Size of each batch.
    :param shuffle_buffer_size: Number of elements from which this dataset will sample.
    :param prefetch: Number of samples loaded in advance by each worker. Not applicable
        (and should not be given) when `num_workers` is 0.
    :param num_workers: how many subprocesses to use for data loading. 0 means that the
        data will be loaded in the main process. Note: when `num_workers` > 1
        yielded batches may be shuffled even if `shuffle_buffer_size` is zero.
    :param csr: For sparse 2D arrays, whether to return CSR tensors instead of COO.
    """
    schemas = tuple(map(_get_tensor_schema, array_params))
    collators = tuple(
        _get_tensor_collator(params.array, csr, len(schema.fields))
        for params, schema in zip(array_params, schemas)
    )
    collate_fn = _CompositeCollator(*collators) if len(collators) > 1 else collators[0]

    return DataLoader(
        dataset=_PyTorchTileDBDataset(schemas, shuffle_buffer_size=shuffle_buffer_size),
        batch_size=batch_size,
        prefetch_factor=prefetch,
        num_workers=num_workers,
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
    )


class _PyTorchTileDBDataset(IterableDataset[OneOrMoreTensorsOrSequences]):
    def __init__(self, schemas: Sequence[TensorSchema], shuffle_buffer_size: int = 0):
        super().__init__()
        key_range = schemas[0].key_range
        if not all(key_range.equal_values(schema.key_range) for schema in schemas[1:]):
            raise ValueError(f"All arrays must have the same key range: {key_range}")
        self.schemas = schemas
        self.key_range = key_range
        self._shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self) -> Iterator[OneOrMoreTensorsOrSequences]:
        rows: Iterator[OneOrMoreTensorsOrSequences]
        it_rows = tuple(map(self._iter_rows, self.schemas))
        rows = zip(*it_rows) if len(it_rows) > 1 else it_rows[0]
        if self._shuffle_buffer_size > 0:
            rows = _iter_shuffled(rows, self._shuffle_buffer_size)
        return rows

    def _iter_rows(self, schema: TensorSchema) -> Iterator[TensorOrSequence]:
        max_weight = schema.max_partition_weight
        key_subranges = self.key_range.partition_by_weight(max_weight)
        batches: Iterable[TensorOrSequence] = schema.iter_tensors(key_subranges)
        if len(schema.fields) == 1:
            return (tensor for batch in batches for tensor in batch)
        else:
            return (tensors for batch in batches for tensors in zip(*batch))


def _worker_init(worker_id: int) -> None:
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    if any(schema.sparse for schema in dataset.schemas):
        raise NotImplementedError("https://github.com/pytorch/pytorch/issues/20248")
    key_ranges = list(dataset.key_range.partition_by_count(worker_info.num_workers))
    dataset.key_range = key_ranges[worker_id]


def _get_tensor_schema(array_params: ArrayParams) -> TensorSchema:
    if not array_params.array.schema.sparse:
        return DenseTensorSchema.from_array_params(array_params)
    elif array_params.array.ndim == 2:
        return SparseTensorSchema.from_array_params(array_params, methodcaller("tocsr"))
    else:
        return SparseTensorSchema.from_array_params(array_params)


_SingleCollator = Callable[[TensorSequence], torch.Tensor]


class _CompositeCollator:
    """
    A callable for collating "rows" of data by a separate collator for each "column".
    Returns the collated columns collected into a tuple.
    """

    def __init__(self, *collators: _SingleCollator):
        self._collators = collators

    def __call__(self, rows: Sequence[TensorSequence]) -> Sequence[torch.Tensor]:
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
