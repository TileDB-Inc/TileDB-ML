from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, Mapping, Sequence, Tuple, Union

import numpy as np
import scipy.sparse
import tensorflow as tf

import tiledb

Tensor = Union[tf.Tensor, tf.SparseTensor]


def dense_dense_generator(
    x_array: tiledb.DenseArray,
    y_array: tiledb.DenseArray,
    x_attrs: Sequence[str],
    y_attrs: Sequence[str],
    offsets: range,
    batch_size: int,
    buffer_size: int,
    batch_shuffle: bool = False,
    within_batch_shuffle: bool = False,
) -> Iterator[Tuple[tf.Tensor, ...]]:
    """
    Generator for yielding training batches.

    :param x_array: An opened TileDB array which contains features.
    :param y_array: An opened TileDB array which contains labels.
    :param x_attrs: Attribute names of x_array.
    :param y_attrs: Attribute names of y_array.
    :param offsets: Row start offsets in x_array/y_array.
    :param batch_size: Size of batch, i.e., number of rows returned per call.
    :param batch_shuffle: True for shuffling batches.
    :param within_batch_shuffle: True for shuffling records in each batch.
    :return: An iterator of x and y batches.
    """
    x_batch = DenseBatch(x_attrs)
    y_batch = DenseBatch(y_attrs)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            x_batch.set_buffer_offset(x_buffer, offset)
            y_batch.set_buffer_offset(y_buffer, offset)
            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch.set_batch_slice(batch_slice)
                y_batch.set_batch_slice(batch_slice)
                if within_batch_shuffle:
                    idx = np.arange(len(x_batch))
                    np.random.shuffle(idx)
                else:
                    idx = Ellipsis

                yield x_batch.get_tensors(idx) + y_batch.get_tensors(idx)


def sparse_sparse_generator(
    x_array: tiledb.SparseArray,
    y_array: tiledb.SparseArray,
    x_attrs: Sequence[str],
    y_attrs: Sequence[str],
    offsets: range,
    batch_size: int,
    buffer_size: int,
    batch_shuffle: bool,
    within_batch_shuffle: bool,
) -> Iterator[Tuple[tf.SparseTensor, ...]]:
    """
    Generator for yielding training batches.

    :param x_array: An opened TileDB array which contains features.
    :param y_array: An opened TileDB array which contains labels.
    :param x_attrs: Attribute names of x_array.
    :param y_attrs: Attribute names of y_array.
    :param offsets: Row start offsets in x_array/y_array.
    :param batch_size: Size of batch, i.e., number of rows returned per call.
    :param batch_shuffle: True for shuffling batches.
    :param within_batch_shuffle: True for shuffling records in each batch.
    :return: An iterator of x and y batches.
    """
    x_batch = SparseBatch(x_array, x_attrs, batch_size)
    y_batch = SparseBatch(y_array, y_attrs, batch_size)
    # https://github.com/tensorflow/tensorflow/issues/44565
    with ThreadPoolExecutor(max_workers=2) as executor:
        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            x_batch.set_buffer_offset(x_buffer, offset)
            y_batch.set_buffer_offset(y_buffer, offset)
            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch.set_batch_slice(batch_slice)
                y_batch.set_batch_slice(batch_slice)
                if len(x_batch) != len(y_batch):
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be of "
                        "equal domain extent inside the batch"
                    )
                if x_batch:
                    yield x_batch.get_tensors() + y_batch.get_tensors()


def sparse_dense_generator(
    x_array: tiledb.SparseArray,
    y_array: tiledb.DenseArray,
    x_attrs: Sequence[str],
    y_attrs: Sequence[str],
    offsets: range,
    batch_size: int,
    buffer_size: int,
    batch_shuffle: bool,
    within_batch_shuffle: bool,
) -> Iterator[Tuple[Tensor, ...]]:
    """
    Generator for yielding training batches.

    :param x_array: An opened TileDB array which contains features.
    :param y_array: An opened TileDB array which contains labels.
    :param x_attrs: Attribute names of x_array.
    :param y_attrs: Attribute names of y_array.
    :param offsets: Row start offsets in x_array/y_array.
    :param batch_size: Size of batch, i.e., number of rows returned per call.
    :param batch_shuffle: True for shuffling batches.
    :param within_batch_shuffle: True for shuffling records in each batch.
    :return: An iterator of x and y batches.
    """
    x_batch = SparseBatch(x_array, x_attrs, batch_size)
    y_batch = DenseBatch(y_attrs)
    # https://github.com/tensorflow/tensorflow/issues/44565
    with ThreadPoolExecutor(max_workers=2) as executor:
        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            x_batch.set_buffer_offset(x_buffer, offset)
            y_batch.set_buffer_offset(y_buffer, offset)
            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch.set_batch_slice(batch_slice)
                y_batch.set_batch_slice(batch_slice)
                if len(x_batch) != len(y_batch):
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be of "
                        "equal domain extent inside the batch"
                    )
                if x_batch:
                    yield x_batch.get_tensors() + y_batch.get_tensors()


class DenseBatch:
    def __init__(self, attrs: Sequence[str]):
        self._attrs = attrs

    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        self._buffer = buffer

    def set_batch_slice(self, batch_slice: slice) -> None:
        if not hasattr(self, "_buffer"):
            raise RuntimeError("set_buffer_offset() not called")
        self._batch = {attr: self._buffer[attr][batch_slice] for attr in self._attrs}

    def get_tensors(self, idx: Any = Ellipsis) -> Tuple[tf.Tensor, ...]:
        if not hasattr(self, "_batch"):
            raise RuntimeError("set_batch_slice() not called")
        return tuple(
            tf.convert_to_tensor(attr_batch[idx]) for attr_batch in self._batch.values()
        )

    def __len__(self) -> int:
        if not hasattr(self, "_batch"):
            raise RuntimeError("set_batch_slice() not called")
        return len(next(iter(self._batch.values())))


class SparseBatch:
    def __init__(
        self,
        array: tiledb.SparseArray,
        attrs: Sequence[str],
        batch_size: int,
    ):
        self._schema = array.schema
        self._attrs = attrs
        self._dense_shape = (batch_size, array.schema.domain.shape[1])

    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        # COO to CSR transformation for batching and row slicing
        dim = self._schema.domain.dim
        row = buffer[dim(0).name]
        col = buffer[dim(1).name]
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
        if not hasattr(self, "_buffer_csr"):
            raise RuntimeError("set_buffer_offset() not called")
        self._batch_csr = self._buffer_csr[batch_slice]

    def get_tensors(self) -> Tuple[tf.SparseTensor, ...]:
        if not hasattr(self, "_batch_csr"):
            raise RuntimeError("set_batch_slice() not called")
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
        if not hasattr(self, "_batch_csr"):
            raise RuntimeError("set_batch_slice() not called")
        # return number of non-zero rows
        return int((self._batch_csr.getnnz(axis=1) > 0).sum())

    def __bool__(self) -> bool:
        if not hasattr(self, "_batch_csr"):
            raise RuntimeError("set_batch_slice() not called")
        return len(self._batch_csr.data) > 0


def _iter_batch_offsets(
    buffer_size: int, batch_size: int, batch_shuffle: bool
) -> Iterator[int]:
    # Split the buffer_size into batch_size chunks
    batch_offsets = np.arange(0, buffer_size, batch_size)
    # Shuffle offsets in case we need batch shuffling
    if batch_shuffle:
        np.random.shuffle(batch_offsets)
    return iter(batch_offsets)
