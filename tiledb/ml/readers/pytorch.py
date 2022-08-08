"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

import itertools
import random
from dataclasses import dataclass
from operator import methodcaller
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import scipy.sparse
import sparse
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ._ranges import InclusiveRange
from ._tensor_schema import TensorKind, TensorSchema
from .types import ArrayParams

Tensor = Union[np.ndarray, sparse.COO, scipy.sparse.csr_matrix]
TensorSequence = Union[
    Sequence[np.ndarray], Sequence[sparse.COO], Sequence[scipy.sparse.csr_matrix]
]
TensorOrSequence = Union[Tensor, TensorSequence]
OneOrMoreTensorsOrSequences = Union[TensorOrSequence, Tuple[TensorOrSequence, ...]]


def PyTorchTileDBDataLoader(
    *all_array_params: ArrayParams,
    shuffle_buffer_size: int = 0,
    **kwargs: Dict[str, Any],
) -> DataLoader:
    """Return a DataLoader for loading data from TileDB arrays.

    :param all_array_params: One or more `ArrayParams` instances, one per TileDB array.
    :param shuffle_buffer_size: Number of elements from which this dataset will sample.
    **kwargs: Should contain all parameters for PyTorch Dataloader. At the moment TileDB-ML can support ONLY the
    following PyTorch Dataloader arguments:
        batch_size: How many samples per batch to load (default: ``1``).
        prefetch_factor: Number of batches loaded in advance by each worker. Not applicable (and should not be
        given) when `num_workers` is 0.
        num_workers: How many subprocesses to use for data loading. 0 means that the data will be loaded in the main
        process. Note: when `num_workers` > 1 yielded batches may be shuffled even if `shuffle_buffer_size` is zero.
        persistent_workers: If ``True``, the data loader will not shutdown the worker processes after a dataset has
        been consumed once. This allows to maintain the workers `Dataset` instances alive. (default: ``False``)
        timeout: if positive, the timeout value for collecting a batch from workers. Should always be non-negative.
        (default: ``0``)
        drop_last: Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by the
        batch size. If ``False`` and the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)

    Users should NOT pass (TileDB-ML either doesn't support or implements internally the corresponding functionality)
    the following arguments: 'shuffle', 'sampler', 'batch_sampler', 'worker_init_fn' and 'collate_fn'.
    """
    schemas = tuple(
        array_params.to_tensor_schema(_transforms) for array_params in all_array_params
    )
    key_range = schemas[0].key_range
    if not all(key_range.equal_values(schema.key_range) for schema in schemas[1:]):
        raise ValueError(f"All arrays must have the same key range: {key_range}")

    if kwargs.get("num_workers") and any(
        schema.kind is not TensorKind.DENSE for schema in schemas
    ):
        raise NotImplementedError("https://github.com/pytorch/pytorch/issues/20248")

    collators = tuple(map(_get_tensor_collator, schemas))
    collate_fn = _CompositeCollator(*collators) if len(collators) > 1 else collators[0]

    return DataLoader(
        dataset=_PyTorchTileDBDataset(schemas, key_range, shuffle_buffer_size),
        worker_init_fn=_worker_init,
        collate_fn=collate_fn,
        **kwargs,
    )


@dataclass
class _PyTorchTileDBDataset(IterableDataset[OneOrMoreTensorsOrSequences]):
    schemas: Sequence[TensorSchema[Tensor]]
    key_range: InclusiveRange[Any, int]
    shuffle_buffer_size: int

    def __iter__(self) -> Iterator[OneOrMoreTensorsOrSequences]:
        it_rows = tuple(map(self._iter_rows, self.schemas))
        rows = zip(*it_rows) if len(it_rows) > 1 else it_rows[0]
        if self.shuffle_buffer_size > 0:
            rows = _iter_shuffled(rows, self.shuffle_buffer_size)
        return rows

    def _iter_rows(self, schema: TensorSchema[Tensor]) -> Iterator[TensorOrSequence]:
        max_weight = schema.max_partition_weight
        key_subranges = self.key_range.partition_by_weight(max_weight)
        batches = schema.iter_tensors(key_subranges)
        if schema.num_fields == 1:
            return (tensor for batch in batches for tensor in batch)
        else:
            return (tensors for batch in batches for tensors in zip(*batch))


def _worker_init(worker_id: int) -> None:
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    key_ranges = tuple(dataset.key_range.partition_by_count(worker_info.num_workers))
    dataset.key_range = key_ranges[worker_id]


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


def _ragged_ndarray_collate(arrays: Sequence[np.ndarray]) -> torch.Tensor:
    """Collate multiple 1D Numpy arrays of possibly different size to a NestedTensor."""
    return torch.nested_tensor(tuple(map(torch.from_numpy, arrays)))


def _coo_collate(arrays: Sequence[sparse.COO]) -> torch.Tensor:
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
    return torch.sparse_csr_tensor(
        torch.from_numpy(stacked.indptr),
        torch.from_numpy(stacked.indices),
        stacked.data,
        stacked.shape,
    )


def _get_tensor_collator(
    schema: TensorSchema[Tensor],
) -> Union[_SingleCollator, _CompositeCollator]:
    if schema.kind is TensorKind.DENSE:
        collator = _ndarray_collate
    elif schema.kind is TensorKind.RAGGED:
        collator = _ragged_ndarray_collate
    elif schema.kind is TensorKind.SPARSE_COO:
        if len(schema.shape) != 2:
            collator = _coo_collate
        else:
            collator = _csr_to_coo_collate
    elif schema.kind is TensorKind.SPARSE_CSR:
        if len(schema.shape) != 2:
            raise ValueError("SPARSE_CSR is supported only for 2D tensors")
        collator = _csr_collate
    else:
        assert False, schema.kind

    num_fields = schema.num_fields
    if num_fields == 1:
        return collator
    else:
        return _CompositeCollator(*itertools.repeat(collator, num_fields))


_transforms: Mapping[TensorKind, Union[Callable[[Any], Any], bool]] = {
    TensorKind.DENSE: True,
    TensorKind.SPARSE_COO: methodcaller("to_sparse_array"),
    TensorKind.SPARSE_CSR: methodcaller("to_sparse_array"),
    TensorKind.RAGGED: hasattr(torch, "nested_tensor"),
}


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
    yield from buffer
