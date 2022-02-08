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
    with ThreadPoolExecutor(max_workers=2) as executor:
        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch = DenseBatch(x_buffer, x_attrs, batch_slice)
                y_batch = DenseBatch(y_buffer, y_attrs, batch_slice)

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
    # https://github.com/tensorflow/tensorflow/issues/44565
    with ThreadPoolExecutor(max_workers=2) as executor:
        x_dense_shape = (batch_size, x_array.schema.domain.shape[1])
        y_dense_shape = (batch_size, y_array.schema.domain.shape[1])

        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            # COO to CSR transformation for batching and row slicing
            x_buffer_csr = _to_csr(x_array, x_attrs[0], x_buffer, offset)
            y_buffer_csr = _to_csr(y_array, y_attrs[0], y_buffer, offset)

            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch = SparseBatch(
                    x_buffer_csr[batch_slice], x_attrs, x_array, x_dense_shape
                )
                if not x_batch:
                    return

                y_batch = SparseBatch(
                    y_buffer_csr[batch_slice], y_attrs, y_array, y_dense_shape
                )
                if len(x_batch) != len(y_batch):
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be of "
                        "equal domain extent inside the batch"
                    )

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
    # https://github.com/tensorflow/tensorflow/issues/44565
    with ThreadPoolExecutor(max_workers=2) as executor:
        x_dense_shape = (batch_size, x_array.schema.domain.shape[1])

        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            # COO to CSR transformation for batching and row slicing
            x_buffer_csr = _to_csr(x_array, x_attrs[0], x_buffer, offset)

            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch = SparseBatch(
                    x_buffer_csr[batch_slice], x_attrs, x_array, x_dense_shape
                )
                if not x_batch:
                    return

                y_batch = DenseBatch(y_buffer, y_attrs, batch_slice)
                if len(x_batch) != len(y_batch):
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be of "
                        "equal domain extent inside the batch"
                    )

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


class SparseBatch:
    def __init__(
        self,
        batch_csr: scipy.sparse.csr_matrix,
        attrs: Sequence[str],
        array: tiledb.Array,
        dense_shape: Tuple[int, ...],
    ):
        self._batch_csr = batch_csr
        self._attrs = attrs
        self._schema = array.schema
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


def _iter_batch_offsets(
    buffer_size: int, batch_size: int, batch_shuffle: bool
) -> Iterator[int]:
    # Split the buffer_size into batch_size chunks
    batch_offsets = np.arange(0, buffer_size, batch_size)
    # Shuffle offsets in case we need batch shuffling
    if batch_shuffle:
        np.random.shuffle(batch_offsets)
    return iter(batch_offsets)


def _to_csr(
    array: tiledb.Array,
    attr: str,
    buffer: Mapping[str, np.ndarray],
    offset: int,
) -> scipy.sparse.csr_matrix:
    """
    :param array: The matrix on which the transformation will have effect
    :param buffer: The buffered slice of the matrix to be batched
    :param offset: The starting offset of the buffered slice
    :returns A CSR representation of the buffered slice of the matrix
    """
    dim = array.schema.domain.dim
    row = buffer[dim(0).name]
    col = buffer[dim(1).name]
    # Normalize indices for torch.sparse.Tensor We want the coords indices in every
    # iteration to be in the range of [0, self.batch_size] so the torch.sparse.Tensors
    # can be created batch-wise. If we do not normalize the sparse tensor is being
    # created but with a dimension [0, max(coord_index)], which is overkill
    row_size_norm = row.max() - row.min() + 1
    col_size_norm = col.max() + 1
    return scipy.sparse.csr_matrix(
        (buffer[attr], (row - offset, col)), shape=(row_size_norm, col_size_norm)
    )
