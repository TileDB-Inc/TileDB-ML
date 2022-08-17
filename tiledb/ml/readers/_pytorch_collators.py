from typing import Callable, Sequence, Union

import numpy as np
import scipy.sparse
import sparse
import torch

from ._tensor_schema import TensorKind, TensorSchema

TensorLike = Union[np.ndarray, sparse.COO, scipy.sparse.csr_matrix]
SingleCollator = Callable[[Sequence[TensorLike]], torch.Tensor]


class CompositeCollator:
    """
    A callable for collating "rows" of data by a separate collator for each "column".
    Returns the collated columns collected into a tuple.
    """

    def __init__(self, *collators: SingleCollator):
        self._collators = collators

    def __call__(self, rows: Sequence[Sequence[TensorLike]]) -> Sequence[torch.Tensor]:
        columns = tuple(zip(*rows))
        collators = self._collators
        assert len(columns) == len(collators)
        return tuple(collator(column) for collator, column in zip(collators, columns))


Collator = Union[SingleCollator, CompositeCollator]


def get_schemas_collator(schemas: Sequence[TensorSchema[TensorLike]]) -> Collator:
    collators = tuple(map(_get_schema_collator, schemas))
    return CompositeCollator(*collators) if len(collators) > 1 else collators[0]


def _get_schema_collator(schema: TensorSchema[TensorLike]) -> Collator:
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
        return CompositeCollator(*(collator,) * num_fields)


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
