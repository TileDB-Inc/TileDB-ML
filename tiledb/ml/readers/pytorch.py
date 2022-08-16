"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

from functools import partial
from operator import methodcaller
from typing import Any, Callable, Iterator, Mapping, Sequence, Tuple, Union

import numpy as np
import scipy.sparse
import sparse
import torch
from torch.utils.data import DataLoader, IterDataPipe
from torchdata.datapipes.iter import IterableWrapper

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
    **kwargs: Any,
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

    datapipe_for_key_range = partial(_get_unbatched_datapipe, schemas)
    num_workers = kwargs.get("num_workers", 0)
    if num_workers:
        if any(schema.kind is not TensorKind.DENSE for schema in schemas):
            raise NotImplementedError("https://github.com/pytorch/pytorch/issues/20248")

        # partition the key range into `num_workers` subkey ranges of roughly equal weight
        worker_key_ranges = tuple(key_range.partition_by_count(num_workers))
        # create a datapipe for these partitions
        datapipe = IterableWrapper(worker_key_ranges, deepcopy=False)
        # shard the datapipe so that each worker gets exactly one partition
        datapipe = datapipe.sharding_filter()
        # read and unbatch the tensors for each partition
        datapipe = datapipe.flatmap(datapipe_for_key_range)
    else:
        # create a datapipe that reads and unbatches the tensors for the whole key range
        datapipe = datapipe_for_key_range(key_range)

    # shuffle the unbatched rows if shuffle_buffer_size > 0
    if shuffle_buffer_size:
        # load the rows to be shuffled
        # don't batch them (batch_size=None) or collate them (collate_fn=_identity)
        row_loader = DataLoader(
            datapipe, num_workers=num_workers, batch_size=None, collate_fn=_identity
        )
        # create a new datapipe for these rows
        datapipe = IterableWrapper(iter(row_loader), deepcopy=False)
        # shuffle the datapipe items
        datapipe = datapipe.shuffle(buffer_size=shuffle_buffer_size)
        # run the shuffling on this process, not on workers
        kwargs["num_workers"] = 0

    # determine the collate function that transforms sequences of rows into `torch.Tensor`s
    collators = tuple(map(_get_tensor_collator, schemas))
    collate_fn = _CompositeCollator(*collators) if len(collators) > 1 else collators[0]

    # return the DataLoader for the final datapipe
    return DataLoader(datapipe, collate_fn=collate_fn, **kwargs)


def _identity(x: Any) -> Any:
    return x


def _get_unbatched_datapipe(
    schemas: Sequence[TensorSchema[Tensor]],
    key_range: InclusiveRange[Any, int],
) -> IterDataPipe[OneOrMoreTensorsOrSequences]:
    """Return a datapipe over unbatched rows for the given schemas and key range.

    If `len(schemas) == 1`, each item of the datapipe is either a single `Tensor` or a
    sequence of `Tensor`s, depending on `schemas[0].num_fields`.
    If `len(schemas) > 1`, each item of the datapipe is a tuple of (`Tensor` or sequence
    of `Tensor`s, depending on `schema.num_fields`), one for each schema.
    """
    schema_dps = [
        IterableWrapper(_unbatch_tensors(schema, key_range), deepcopy=False)
        for schema in schemas
    ]
    dp = schema_dps.pop(0)
    if schema_dps:
        dp = dp.zip(*schema_dps)
    return dp


def _unbatch_tensors(
    schema: TensorSchema[Tensor], key_range: InclusiveRange[Any, int]
) -> Iterator[TensorOrSequence]:
    """
    Generate batches of `Tensor`s for the given schema and key range and then unbatch them
    into single "rows". Each row is either a single `Tensor` (if schema.num_fields == 1)
    or a sequence of `Tensor`s (if schema.num_fields > 1).
    """
    batches = schema.iter_tensors(
        key_range.partition_by_weight(schema.max_partition_weight)
    )
    if schema.num_fields > 1:
        # convert batches of columns to batches of rows
        batches = (zip(*batch) for batch in batches)
    # flatten batches of rows
    return (row for batch in batches for row in batch)


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
    """Return a callable for collating a sequence into a tensor based on the given schema."""
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
        return _CompositeCollator(*(collator,) * num_fields)


_transforms: Mapping[TensorKind, Union[Callable[[Any], Any], bool]] = {
    TensorKind.DENSE: True,
    TensorKind.SPARSE_COO: methodcaller("to_sparse_array"),
    TensorKind.SPARSE_CSR: methodcaller("to_sparse_array"),
    TensorKind.RAGGED: hasattr(torch, "nested_tensor"),
}
