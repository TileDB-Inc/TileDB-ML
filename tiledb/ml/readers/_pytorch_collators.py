from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy as np
import scipy.sparse
import sparse
import torch

from ._tensor_schema import TensorKind, TensorSchema


def get_collate_fn(
    schemas: Sequence[TensorSchema[Any]], is_batched: bool
) -> Callable[[Any], torch.Tensor]:
    """
    Return an appropriate callable to be used as the `collate_fn` parameter of Dataloader.
    """
    collators = tuple(map(Collator.from_schema, schemas))
    collator = RowCollator(*collators) if len(collators) > 1 else collators[0]
    return collator.collate if is_batched else collator.convert


T = TypeVar("T")


class Collator(ABC, Generic[T]):
    """Mapper of instances of type `T` to `torch.Tensor`s"""

    @abstractmethod
    def convert(self, value: T) -> torch.Tensor:
        """Convert a tensor-like object of type `T` into a `torch.Tensor`."""

    @abstractmethod
    def collate(self, batch: Sequence[T]) -> torch.Tensor:
        """
        Collate a batch of tensor-like objects of type `T` into a `torch.Tensor`
        with an additional outer dimension - batch size.
        """

    @staticmethod
    def from_schema(schema: TensorSchema[Any]) -> Collator[Any]:
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

    def convert(self, value: Sequence[Any]) -> Sequence[torch.Tensor]:
        assert len(value) == len(self._column_collators)
        return tuple(
            collator.convert(value)
            for collator, value in zip(self._column_collators, value)
        )

    def collate(self, batch: Sequence[Sequence[Any]]) -> Sequence[torch.Tensor]:
        columns = tuple(zip(*batch))
        assert len(columns) == len(self._column_collators)
        return tuple(
            collator.collate(column)
            for collator, column in zip(self._column_collators, columns)
        )


class NDArrayCollator(Collator[np.ndarray]):
    """Collator of Numpy arrays of fixed shape"""

    def convert(self, value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value)

    def collate(self, batch: Sequence[np.ndarray]) -> torch.Tensor:
        return self.convert(np.stack(batch))


class RaggedNDArrayCollator(NDArrayCollator):
    """Collator of 1D Numpy arrays with possibly different size"""

    def collate(self, batch: Sequence[np.ndarray]) -> torch.Tensor:
        return torch.nested_tensor(tuple(map(torch.from_numpy, batch)))


class SparseCOOCollator(Collator[sparse.COO]):
    """Collator of sparse.COO arrays to a `torch.Tensor` with `sparse_coo` layout."""

    def convert(self, value: sparse.COO) -> torch.Tensor:
        return torch.sparse_coo_tensor(value.coords, value.data, value.shape)

    def collate(self, batch: Sequence[sparse.COO]) -> torch.Tensor:
        return self.convert(sparse.stack(batch))


class ScipySparseCSRCollator(Collator[scipy.sparse.csr_matrix]):
    """
    Collator of `scipy.sparse.csr_matrix`s to a `torch.Tensor` with `sparse_csr` layout.
    """

    def convert(self, value: scipy.sparse.csr_matrix) -> torch.Tensor:
        return torch.sparse_csr_tensor(
            value.indptr, value.indices, value.data, value.shape
        )

    def collate(self, batch: Sequence[scipy.sparse.csr_matrix]) -> torch.Tensor:
        return self.convert(scipy.sparse.vstack(batch))


class ScipySparseCSRToCOOCollator(ScipySparseCSRCollator):
    """
    Collator of `scipy.sparse.csr_matrix`s to a `torch.Tensor` with `sparse_coo` layout.
    """

    def convert(self, value: scipy.sparse.csr_matrix) -> torch.Tensor:
        coo = value.tocoo()
        coords = np.stack((coo.row, coo.col))
        return torch.sparse_coo_tensor(coords, coo.data, coo.shape)
