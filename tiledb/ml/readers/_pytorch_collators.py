from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

import numpy as np
import scipy.sparse
import sparse
import torch

from ._tensor_schema import TensorKind, TensorSchema

T = TypeVar("T")


class Collator(ABC, Generic[T]):
    @abstractmethod
    def collate(self, batch: Sequence[T]) -> torch.Tensor:
        """
        Collate a batch of objects of type `T` into a `torch.Tensor` with an
        additional  outer dimension - batch size.
        """

    @classmethod
    def from_schemas(cls, schemas: Sequence[TensorSchema[Any]]) -> Collator[Any]:
        """
        Return an appropriate Collator for collating instances generated by the
        `iter_tensors()` method of the given schemas.
        """
        collators = tuple(map(cls.from_schema, schemas))
        return RowCollator(*collators) if len(collators) > 1 else collators[0]

    @classmethod
    def from_schema(cls, schema: TensorSchema[Any]) -> Collator[Any]:
        """
        Return an appropriate Collator for collating instances generated by the
        `iter_tensors()` method of the given schema.
        """
        collator: Collator[Any]
        if schema.kind is TensorKind.DENSE:
            collator = NDArrayCollator()
        elif schema.kind is TensorKind.RAGGED:
            collator = RaggedNDArrayCollator()
        elif schema.kind is TensorKind.SPARSE_COO:
            if len(schema.shape) != 2:
                collator = SparseCOOCollator()
            else:
                collator = ScipySparseCSRToCOOCollator()
        elif schema.kind is TensorKind.SPARSE_CSR:
            if len(schema.shape) != 2:
                raise ValueError("SPARSE_CSR is supported only for 2D tensors")
            collator = ScipySparseCSRCollator()
        else:
            assert False, schema.kind

        num_fields = schema.num_fields
        return RowCollator(*(collator,) * num_fields) if num_fields > 1 else collator


class RowCollator(Collator[Sequence[Any]]):
    """
    Collator of "row" tuples.

    - All rows must have the same length.
    - The i-th element of every row (i.e. the i-th "column") must have the same type.
    - The i-th column values are collated by the i-th collator given in the constructor.
    """

    def __init__(self, *column_collators: Collator[Any]):
        self._column_collators = column_collators

    def collate(self, batch: Sequence[Sequence[Any]]) -> Sequence[torch.Tensor]:
        columns = tuple(zip(*batch))
        assert len(columns) == len(self._column_collators)
        return tuple(
            collator.collate(column)
            for collator, column in zip(self._column_collators, columns)
        )


class NDArrayCollator(Collator[np.ndarray]):
    """Numpy array collator to `torch.Tensor` with strided layout."""

    def collate(self, batch: Sequence[np.ndarray]) -> torch.Tensor:
        # Faster than `torch.as_tensor(arrays)` (https://github.com/pytorch/pytorch/pull/51731)
        # and `torch.stack([torch.as_tensor(array) for array in arrays]])`
        return torch.from_numpy(np.stack(batch))


class RaggedNDArrayCollator(Collator[np.ndarray]):
    """
    Collator of 1D Numpy arrays with possibly different size to a `torch.NestedTensor`.
    """

    def collate(self, batch: Sequence[np.ndarray]) -> torch.Tensor:
        return torch.nested_tensor(tuple(map(torch.from_numpy, batch)))


class SparseCOOCollator(Collator[sparse.COO]):
    """Collator of sparse.COO arrays to a `torch.Tensor` with `sparse_coo` layout."""

    def collate(self, batch: Sequence[sparse.COO]) -> torch.Tensor:
        stacked = sparse.stack(batch)
        return torch.sparse_coo_tensor(stacked.coords, stacked.data, stacked.shape)


class ScipySparseCSRCollator(Collator[scipy.sparse.csr_matrix]):
    """
    Collator of `scipy.sparse.csr_matrix`s to a `torch.Tensor` with `sparse_csr` layout.
    """

    def collate(self, batch: Sequence[scipy.sparse.csr_matrix]) -> torch.Tensor:
        stacked = scipy.sparse.vstack(batch)
        return torch.sparse_csr_tensor(
            torch.from_numpy(stacked.indptr),
            torch.from_numpy(stacked.indices),
            stacked.data,
            stacked.shape,
        )


class ScipySparseCSRToCOOCollator(Collator[scipy.sparse.csr_matrix]):
    """
    Collator of `scipy.sparse.csr_matrix`s to a `torch.Tensor` with `sparse_coo` layout.
    """

    def collate(self, batch: Sequence[scipy.sparse.csr_matrix]) -> torch.Tensor:
        stacked = scipy.sparse.vstack(batch).tocoo()
        coords = np.stack((stacked.row, stacked.col))
        return torch.sparse_coo_tensor(coords, stacked.data, stacked.shape)
