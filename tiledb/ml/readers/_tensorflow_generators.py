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
    x_attribute_names: Sequence[str],
    y_attribute_names: Sequence[str],
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
    :param x_attribute_names: Attribute names of x_array.
    :param y_attribute_names: Attribute names of y_array.
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
                x_batch = {
                    attr: data[batch_offset : batch_offset + batch_size]
                    for attr, data in x_buffer.items()
                }
                y_batch = {
                    attr: data[batch_offset : batch_offset + batch_size]
                    for attr, data in y_buffer.items()
                }

                if within_batch_shuffle:
                    # We get batch length based on the first attribute
                    # because last batch might be smaller than the batch size
                    idx = np.arange(x_batch[x_attribute_names[0]].shape[0])
                    np.random.shuffle(idx)
                else:
                    idx = Ellipsis

                # Yield the next training batch
                x_tensors = _get_dense_tensors(x_batch, x_attribute_names, idx)
                y_tensors = _get_dense_tensors(y_batch, y_attribute_names, idx)
                yield x_tensors + y_tensors


def sparse_sparse_generator(
    x_array: tiledb.SparseArray,
    y_array: tiledb.SparseArray,
    x_attribute_names: Sequence[str],
    y_attribute_names: Sequence[str],
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
    :param x_attribute_names: Attribute names of x_array.
    :param y_attribute_names: Attribute names of y_array.
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
            x_buffer_csr = _to_csr(x_array, x_attribute_names[0], x_buffer, offset)
            y_buffer_csr = _to_csr(y_array, y_attribute_names[0], y_buffer, offset)

            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                x_batch_csr = x_buffer_csr[batch_offset : batch_offset + batch_size]
                if x_batch_csr.data.size == 0:
                    return

                y_batch_csr = y_buffer_csr[batch_offset : batch_offset + batch_size]

                # Keep row records number for cross-check between x_array and y_array
                # batches. Last index excluded shows to empty
                samples_num_x = np.unique(x_batch_csr.indptr[:-1]).size
                samples_num_y = np.unique(y_batch_csr.indptr[:-1]).size
                if samples_num_x != samples_num_y:
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be of "
                        "equal domain extent inside the batch"
                    )

                # Transform back to COO for torch.sparse_coo_tensor to digest
                x_tensors = _get_sparse_tensors(
                    x_batch_csr, x_array, x_attribute_names, x_dense_shape
                )
                y_tensors = _get_sparse_tensors(
                    y_batch_csr, y_array, y_attribute_names, y_dense_shape
                )
                yield x_tensors + y_tensors


def sparse_dense_generator(
    x_array: tiledb.SparseArray,
    y_array: tiledb.DenseArray,
    x_attribute_names: Sequence[str],
    y_attribute_names: Sequence[str],
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
    :param x_attribute_names: Attribute names of x_array.
    :param y_attribute_names: Attribute names of y_array.
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
            x_buffer_csr = _to_csr(x_array, x_attribute_names[0], x_buffer, offset)

            for batch_offset in _iter_batch_offsets(
                buffer_size, batch_size, batch_shuffle
            ):
                x_batch_csr = x_buffer_csr[batch_offset : batch_offset + batch_size]
                if x_batch_csr.data.size == 0:
                    return

                y_batch = {
                    attr: data[batch_offset : batch_offset + batch_size]
                    for attr, data in y_buffer.items()
                }

                # Keep row records number for cross-check between x_array and y_array
                # batches. Last index excluded shows to empty
                samples_num_x = np.unique(x_batch_csr.indptr[:-1]).size
                # for the check slice the row dimension of y_array dense array
                samples_num_y = y_batch[y_attribute_names[0]].shape[0]
                if samples_num_x != samples_num_y:
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be of "
                        "equal domain extent inside the batch"
                    )

                # Transform back to COO for torch.sparse_coo_tensor to digest
                x_tensors = _get_sparse_tensors(
                    x_batch_csr, x_array, x_attribute_names, x_dense_shape
                )
                y_tensors = _get_dense_tensors(y_batch, y_attribute_names)
                yield x_tensors + y_tensors


def _iter_batch_offsets(
    buffer_size: int, batch_size: int, batch_shuffle: bool
) -> Iterator[int]:
    # Split the buffer_size into batch_size chunks
    batch_offsets = np.arange(0, buffer_size, batch_size)
    # Shuffle offsets in case we need batch shuffling
    if batch_shuffle:
        np.random.shuffle(batch_offsets)
    return iter(batch_offsets)


def _get_dense_tensors(
    batch: Mapping[str, np.ndarray], attrs: Sequence[str], idx: Any = Ellipsis
) -> Tuple[tf.Tensor, ...]:
    return tuple(tf.convert_to_tensor(batch[attr][idx]) for attr in attrs)


def _get_sparse_tensors(
    batch_csr: scipy.sparse.csr_matrix,
    array: tiledb.SparseArray,
    attrs: Sequence[str],
    dense_shape: Tuple[int, ...],
) -> Tuple[tf.SparseTensor, ...]:
    batch_coo = batch_csr.tocoo()
    coords = np.stack((batch_coo.row, batch_coo.col), axis=-1)
    return tuple(
        tf.SparseTensor(
            indices=tf.constant(coords, dtype=tf.int64),
            values=tf.constant(batch_coo.data, dtype=array.schema.attr(attr).dtype),
            dense_shape=dense_shape,
        )
        for attr in attrs
    )


def _to_csr(
    array: tiledb.Array,
    attribute_names: str,
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
        (
            buffer[attribute_names],
            (row - offset, col),
        ),
        shape=(row_size_norm, col_size_norm),
    )
