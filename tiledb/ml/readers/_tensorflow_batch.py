from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple, Union

import numpy as np
import scipy.sparse
import tensorflow as tf

import tiledb


def TensorflowBatch(
    array: tiledb.Array, attrs: Sequence[str], batch_size: int
) -> Union[TensorflowDenseBatch, TensorflowSparseBatch]:
    if isinstance(array, tiledb.DenseArray):
        return TensorflowDenseBatch(attrs)
    if isinstance(array, tiledb.SparseArray):
        return TensorflowSparseBatch(attrs, array.schema, batch_size)
    raise TypeError(f"Unsupported type: {type(array)}")


class TensorflowDenseBatch:
    def __init__(self, attrs: Sequence[str]):
        self._attrs = attrs

    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        self._buffer = {attr: buffer[attr] for attr in self._attrs}

    def set_batch_slice(self, batch_slice: slice) -> None:
        _ensure_attr(self, "_buffer", "set_buffer_offset() not called")
        self._batch = {attr: data[batch_slice] for attr, data in self._buffer.items()}

    def get_tensors(self, idx: Any = Ellipsis) -> Tuple[tf.Tensor, ...]:
        _ensure_attr(self, "_batch", "set_batch_slice() not called")
        return tuple(
            tf.convert_to_tensor(attr_batch[idx]) for attr_batch in self._batch.values()
        )

    def __len__(self) -> int:
        _ensure_attr(self, "_batch", "set_batch_slice() not called")
        return len(next(iter(self._batch.values())))


class TensorflowSparseBatch:
    def __init__(
        self, attrs: Sequence[str], schema: tiledb.ArraySchema, batch_size: int
    ):
        domain = schema.domain
        self._row_dim = domain.dim(0).name
        self._col_dim = domain.dim(1).name
        self._dense_shape = (batch_size, domain.shape[1])
        self._schema = schema
        self._attrs = attrs

    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        # COO to CSR transformation for batching and row slicing
        row = buffer[self._row_dim]
        col = buffer[self._col_dim]
        # Normalize indices: We want the coords indices to be in the [0, batch_size]
        # range. If we do not normalize the sparse tensor is being created but with a
        # dimension [0, max(coord_index)], which is overkill
        row_size_norm = row.max() - row.min() + 1
        col_size_norm = col.max() + 1
        self._buffer_csr = scipy.sparse.csr_matrix(
            (buffer[self._attrs[0]], (row - offset, col)),
            shape=(row_size_norm, col_size_norm),
        )

    def set_batch_slice(self, batch_slice: slice) -> None:
        _ensure_attr(self, "_buffer_csr", "set_buffer_offset() not called")
        self._batch_csr = self._buffer_csr[batch_slice]

    def get_tensors(self, idx: Any = Ellipsis) -> Tuple[tf.SparseTensor, ...]:
        _ensure_attr(self, "_batch_csr", "set_batch_slice() not called")
        if idx is not Ellipsis:
            raise NotImplementedError(
                "within_batch_shuffle not implemented for sparse arrays"
            )
        batch_coo = self._batch_csr.tocoo()
        coords = np.stack((batch_coo.row, batch_coo.col), axis=-1)
        return tuple(
            tf.SparseTensor(
                indices=tf.constant(coords, dtype=tf.int64),
                values=tf.constant(batch_coo.data, dtype=self._schema.attr(attr).dtype),
                dense_shape=self._dense_shape,
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


def _ensure_attr(obj: Any, attr: str, message: str) -> None:
    if not hasattr(obj, attr):
        raise RuntimeError(message)
