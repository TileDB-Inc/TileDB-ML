"""Functionality for loading data from TileDB arrays to the PyTorch Dataloader API."""

from functools import partial
from typing import Any, Callable, Iterator, Sequence, Tuple, Union

import numpy as np
import scipy.sparse
import sparse
import torchdata
from torch.utils.data import DataLoader, IterDataPipe
from torchdata.datapipes.iter import IterableWrapper

from ._pytorch_collators import Collator
from ._tensor_schema import TensorSchema
from ._tensor_schema.ranges import InclusiveRange
from .types import ArrayParams, TensorKind

TensorLike = Union[np.ndarray, sparse.COO, scipy.sparse.csr_matrix]
TensorLikeOrTuple = Union[TensorLike, Tuple[TensorLike, ...]]


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
    is_batched = kwargs.get("batch_size", 1) is not None

    schemas = []
    for array_params in all_array_params:
        schema = array_params.tensor_schema
        # unbatched 3D arrays generate 2D tensors so they can be converted to CSR
        # anything else with ndim>=3 cannot
        if schema.kind is TensorKind.SPARSE_CSR:
            ndim = len(schema.shape)
            if ndim > 3 or ndim == 3 and is_batched:
                raise ValueError(f"Cannot generate CSR tensors for {ndim}D array")
        schemas.append(schema)

    key_range = schemas[0].key_range
    if not all(key_range.equal_values(schema.key_range) for schema in schemas[1:]):
        raise ValueError(f"All arrays must have the same key range: {key_range}")

    datapipe_for_key_range = partial(_get_unbatched_datapipe, schemas)
    num_workers = kwargs.get("num_workers", 0)
    if num_workers:
        if torchdata.__version__ < "0.4":
            raise NotImplementedError("torchdata>=0.4 required for multiple workers")
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
        datapipe = DeferredIterableIterDataPipe(iter, row_loader)
        # shuffle the datapipe items
        datapipe = datapipe.shuffle(buffer_size=shuffle_buffer_size)
        # run the shuffling on this process, not on workers
        kwargs["num_workers"] = 0

    # construct an appropriate collate function
    collator = Collator.from_schemas(*schemas)
    kwargs["collate_fn"] = collator.collate if is_batched else collator.convert

    # return the DataLoader for the final datapipe
    return DataLoader(datapipe, **kwargs)


class DeferredIterableIterDataPipe(IterDataPipe):
    """Wraps a callable that returns an iterable object to create an IterDataPipe."""

    def __init__(self, func: Callable[..., Iterator[Any]], *args: Any, **kwargs: Any):
        self._callable = partial(func, *args, **kwargs)

    def __iter__(self) -> Iterator[Any]:
        return self._callable()


def _identity(x: Any) -> Any:
    return x


def _get_unbatched_datapipe(
    schemas: Sequence[TensorSchema[TensorLike]],
    key_range: InclusiveRange[Any, int],
) -> IterDataPipe[Union[TensorLikeOrTuple, Tuple[TensorLikeOrTuple, ...]]]:
    """Return a datapipe over unbatched rows for the given schemas and key range.

    If `len(schemas) == 1`, each item of the datapipe is either a single `TensorLike`
    or a sequence of `TensorLike`s, depending on `schemas[0].num_fields`.
    If `len(schemas) > 1`, each item of the datapipe is a tuple of (`TensorLike` or
    sequence of `TensorLike`s, depending on `schema.num_fields`), one for each schema.
    """
    schema_dps = [
        DeferredIterableIterDataPipe(_unbatch_tensors, schema, key_range)
        for schema in schemas
    ]
    dp = schema_dps.pop(0)
    if schema_dps:
        dp = dp.zip(*schema_dps)
    return dp


def _unbatch_tensors(
    schema: TensorSchema[TensorLike], key_range: InclusiveRange[Any, int]
) -> Iterator[TensorLikeOrTuple]:
    """
    Generate batches of `TensorLike`s for the given schema and key range and then unbatch
    them into single "rows".

    If `schema.num_fields == 1`, each "row" is a single `TensorLike`
    If `schema.num_fields > 1`, each "row" is a sequence of `TensorLike`s
    """
    batches = schema.iter_tensors(
        key_range.partition_by_weight(schema.max_partition_weight)
    )
    if schema.num_fields > 1:
        # convert batches of columns to batches of rows
        batches = (zip(*batch) for batch in batches)
    # flatten batches of rows
    return (row for batch in batches for row in batch)
