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
    x_batch_factory = DenseBatchFactory(x_attrs)
    y_batch_factory = DenseBatchFactory(y_attrs)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            x_batch_factory.set_buffer_offset(x_buffer, offset)
            y_batch_factory.set_buffer_offset(y_buffer, offset)
            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch = x_batch_factory.get_batch(batch_slice)
                y_batch = y_batch_factory.get_batch(batch_slice)
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
    x_batch_factory = SparseBatchFactory(x_array, x_attrs, batch_size)
    y_batch_factory = SparseBatchFactory(y_array, y_attrs, batch_size)
    # https://github.com/tensorflow/tensorflow/issues/44565
    with ThreadPoolExecutor(max_workers=2) as executor:
        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            x_batch_factory.set_buffer_offset(x_buffer, offset)
            y_batch_factory.set_buffer_offset(y_buffer, offset)
            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch = x_batch_factory.get_batch(batch_slice)
                y_batch = y_batch_factory.get_batch(batch_slice)
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
    x_batch_factory = SparseBatchFactory(x_array, x_attrs, batch_size)
    y_batch_factory = DenseBatchFactory(y_attrs)
    # https://github.com/tensorflow/tensorflow/issues/44565
    with ThreadPoolExecutor(max_workers=2) as executor:
        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            x_batch_factory.set_buffer_offset(x_buffer, offset)
            y_batch_factory.set_buffer_offset(y_buffer, offset)
            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch = x_batch_factory.get_batch(batch_slice)
                y_batch = y_batch_factory.get_batch(batch_slice)
                if len(x_batch) != len(y_batch):
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be of "
                        "equal domain extent inside the batch"
                    )
                if x_batch:
                    yield x_batch.get_tensors() + y_batch.get_tensors()


class DenseBatch:
    def __init__(
        self, buffer: Mapping[str, np.ndarray], attrs: Sequence[str], batch_slice: slice
    ):
        self._attr_to_batch = {attr: buffer[attr][batch_slice] for attr in attrs}

    def __len__(self) -> int:
        return len(next(iter(self._attr_to_batch.values())))

    def get_tensors(self, idx: Any = Ellipsis) -> Tuple[tf.Tensor, ...]:
        return tuple(
            tf.convert_to_tensor(batch[idx]) for batch in self._attr_to_batch.values()
        )


class DenseBatchFactory:
    def __init__(self, attrs: Sequence[str]):
        self._attrs = attrs

    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        self._buffer = buffer

    def get_batch(self, batch_slice: slice) -> DenseBatch:
        return DenseBatch(self._buffer, self._attrs, batch_slice)


class SparseBatch:
    def __init__(
        self,
        batch_csr: scipy.sparse.csr_matrix,
        array: tiledb.Array,
        attrs: Sequence[str],
        dense_shape: Tuple[int, ...],
    ):
        self._batch_csr = batch_csr
        self._schema = array.schema
        self._attrs = attrs
        self._dense_shape = dense_shape

    def __bool__(self) -> bool:
        return len(self._batch_csr.data) > 0

    def __len__(self) -> int:
        # return number of non-zero rows
        return int((self._batch_csr.getnnz(axis=1) > 0).sum())

    def get_tensors(self) -> Tuple[tf.SparseTensor, ...]:
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


class SparseBatchFactory:
    def __init__(
        self,
        array: tiledb.SparseArray,
        attrs: Sequence[str],
        batch_size: int,
    ):
        self._array = array
        self._attrs = attrs
        self._dense_shape = (batch_size, array.schema.domain.shape[1])

    def set_buffer_offset(self, buffer: Mapping[str, np.ndarray], offset: int) -> None:
        # COO to CSR transformation for batching and row slicing
        dim = self._array.schema.domain.dim
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

    def get_batch(self, batch_slice: slice) -> SparseBatch:
        return SparseBatch(
            self._buffer_csr[batch_slice], self._array, self._attrs, self._dense_shape
        )


def _iter_batch_offsets(
    buffer_size: int, batch_size: int, batch_shuffle: bool
) -> Iterator[int]:
    # Split the buffer_size into batch_size chunks
    batch_offsets = np.arange(0, buffer_size, batch_size)
    # Shuffle offsets in case we need batch shuffling
    if batch_shuffle:
        np.random.shuffle(batch_offsets)
    return iter(batch_offsets)
