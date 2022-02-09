from abc import ABC, abstractmethod
from typing import Any, Generic, Mapping, Sequence, Tuple, TypeVar

import numpy as np
import scipy.sparse as sp

import tiledb

Tensor = TypeVar("Tensor")


class BaseBatch(ABC, Generic[Tensor]):
    """
    Base class used for getting tensors from batches of buffers read from a TileDB array.
    """

    def __init__(self, attrs: Sequence[str]):
        """Initialize this instance with the attributes to create tensors for each batch.

        :param attrs: Sequence of attribute names.
        """
        self._attrs = attrs

    @abstractmethod
    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        """Set the current buffer from which subsequent batches are to be read.

        :param buffer: Mapping of attribute names to numpy arrays.
        :param offset: Start offset of the buffer in the TileDB array.
        """

    @abstractmethod
    def set_batch_slice(self, batch_slice: slice) -> None:
        """Set the current batch as a slice of the set buffer.

        Must be called after `set_buffer_offset`.

        :param batch_slice: Slice of the buffer to be used as the current batch.
        """

    @abstractmethod
    def get_tensors(self, idx: Any = Ellipsis) -> Tuple[Tensor, ...]:
        """
        Get a tuple of tensors for the current batch, one tensor per attribute
        given in the constructor.

        Must be called after `set_batch_slice`.

        :param idx: Optional indexer for the current batch (e.g. to get a permutation).
        """

    @abstractmethod
    def __len__(self) -> int:
        """Get the size (i.e. number of rows) of the current batch.

        Must be called after `set_batch_slice`.
        """


class BaseDenseBatch(BaseBatch[Tensor]):
    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        self._buffer = buffer

    def set_batch_slice(self, batch_slice: slice) -> None:
        _ensure_attr(self, "_buffer", "set_buffer_offset() not called")
        self._attr_batches = [self._buffer[attr][batch_slice] for attr in self._attrs]

    def get_tensors(self, idx: Any = Ellipsis) -> Tuple[Tensor, ...]:
        _ensure_attr(self, "_attr_batches", "set_batch_slice() not called")
        if idx is Ellipsis:
            iter_attr_batches = iter(self._attr_batches)
        else:
            iter_attr_batches = (attr_batch[idx] for attr_batch in self._attr_batches)
        return tuple(map(self._tensor_from_numpy, iter_attr_batches))

    def __len__(self) -> int:
        _ensure_attr(self, "_attr_batches", "set_batch_slice() not called")
        return len(self._attr_batches[0])

    @staticmethod
    @abstractmethod
    def _tensor_from_numpy(data: np.ndarray) -> Tensor:
        """Convert a numpy array to a Tensor"""


class BaseSparseBatch(BaseBatch[Tensor]):
    def __init__(
        self, attrs: Sequence[str], schema: tiledb.ArraySchema, batch_size: int
    ):
        super().__init__(attrs)
        domain = schema.domain
        self._row_dim = domain.dim(0).name
        self._col_dim = domain.dim(1).name
        self._dense_shape = (batch_size, domain.shape[1])
        self._schema = schema

    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        # COO to CSR transformation for batching and row slicing
        row = buffer[self._row_dim]
        col = buffer[self._col_dim]
        # Normalize indices: We want the coords indices to be in the [0, batch_size]
        # range. If we do not normalize the sparse tensor is being created but with a
        # dimension [0, max(coord_index)], which is overkill
        row_size_norm = row.max() - row.min() + 1
        col_size_norm = col.max() + 1
        self._buffer_csr = sp.csr_matrix(
            (buffer[self._attrs[0]], (row - offset, col)),
            shape=(row_size_norm, col_size_norm),
        )

    def set_batch_slice(self, batch_slice: slice) -> None:
        _ensure_attr(self, "_buffer_csr", "set_buffer_offset() not called")
        self._batch_csr = self._buffer_csr[batch_slice]

    def get_tensors(self, idx: Any = Ellipsis) -> Tuple[Tensor, ...]:
        _ensure_attr(self, "_batch_csr", "set_batch_slice() not called")
        if idx is not Ellipsis:
            raise NotImplementedError(
                "within_batch_shuffle not implemented for sparse arrays"
            )
        batch_coo = self._batch_csr.tocoo()
        coords = np.stack((batch_coo.row, batch_coo.col), axis=-1)
        return tuple(
            self._tensor_from_coo(
                batch_coo.data, coords, self._dense_shape, self._schema.attr(attr).dtype
            )
            for attr in self._attrs
        )

    def __len__(self) -> int:
        _ensure_attr(self, "_batch_csr", "set_batch_slice() not called")
        # return number of non-zero rows
        return int((self._batch_csr.getnnz(axis=1) > 0).sum())

    def __bool__(self) -> bool:
        _ensure_attr(self, "_batch_csr", "set_batch_slice() not called")
        # faster version of __len__() > 0
        return len(self._batch_csr.data) > 0

    @staticmethod
    @abstractmethod
    def _tensor_from_coo(
        data: np.ndarray,
        coords: np.ndarray,
        dense_shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> Tensor:
        """Convert a scipy.sparse.coo_matrix to a Tensor"""


def _ensure_attr(obj: Any, attr: str, message: str) -> None:
    if not hasattr(obj, attr):
        raise RuntimeError(message)
