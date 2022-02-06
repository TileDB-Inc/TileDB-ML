from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Mapping, Sequence, Tuple, Union

import numpy as np
import scipy.sparse
import tensorflow as tf

import tiledb


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
) -> Iterator[Tuple[np.ndarray, ...]]:
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

            # Split the buffer_size into batch_size chunks
            batch_offsets = np.arange(0, buffer_size, batch_size)

            # Shuffle offsets in case we need batch shuffling
            if batch_shuffle:
                np.random.shuffle(batch_offsets)

            for batch_offset in batch_offsets:
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
                    rand_permutation = np.arange(x_batch[x_attribute_names[0]].shape[0])

                    np.random.shuffle(rand_permutation)

                    # Yield the next training batch
                    yield tuple(
                        x_batch[attr][rand_permutation] for attr in x_attribute_names
                    ) + tuple(
                        y_batch[attr][rand_permutation] for attr in y_attribute_names
                    )
                else:
                    # Yield the next training batch
                    yield tuple(x_batch[attr] for attr in x_attribute_names) + tuple(
                        y_batch[attr] for attr in y_attribute_names
                    )


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
    x_shape = x_array.schema.domain.shape[1:]
    y_shape = y_array.schema.domain.shape[1:]

    # https://github.com/tensorflow/tensorflow/issues/44565
    with ThreadPoolExecutor(max_workers=2) as executor:
        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )

            # COO to CSR transformation for batching and row slicing
            x_buffer_csr = to_csr(x_array, x_attribute_names[0], x_buffer, offset)
            y_buffer_csr = to_csr(y_array, y_attribute_names[0], y_buffer, offset)

            # Split the buffer_size into batch_size chunks
            batch_offsets = np.arange(0, buffer_size, batch_size)

            # Shuffle offsets in case we need batch shuffling
            if batch_shuffle:
                np.random.shuffle(batch_offsets)

            for batch_offset in batch_offsets:
                x_batch = x_buffer_csr[batch_offset : batch_offset + batch_size]

                if x_batch.data.size == 0:
                    return

                y_batch = y_buffer_csr[batch_offset : batch_offset + batch_size]

                # Keep row records number for cross-check between x_array and y_array batches
                # Last index excluded shows to empty
                samples_num_x = x_batch.indptr[:-1]

                # Transform back to COO for torch.sparse_coo_tensor to digest
                x_batch_coo = x_batch.tocoo()
                x_coords = np.stack((x_batch_coo.row, x_batch_coo.col), axis=-1)

                # Keep row records number for cross-check between x_array and y_array batches
                # Last index excluded shows to empty
                samples_num_y = y_batch.indptr[:-1]

                # Transform back to COO for torch.sparse_coo_tensor to digest
                y_batch_coo = y_batch.tocoo()
                y_coords = np.stack((y_batch_coo.row, y_batch_coo.col), axis=-1)

                if np.unique(samples_num_x).size != np.unique(samples_num_y).size:
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be of "
                        "equal domain extent inside the batch"
                    )

                yield tuple(
                    tf.SparseTensor(
                        indices=tf.constant(x_coords, dtype=tf.int64),
                        values=tf.constant(
                            x_batch_coo.data, dtype=x_array.schema.attr(attr).dtype
                        ),
                        dense_shape=(batch_size, x_shape[0]),
                    )
                    for attr in x_attribute_names
                ) + tuple(
                    tf.SparseTensor(
                        indices=tf.constant(y_coords, dtype=tf.int64),
                        values=tf.constant(
                            y_batch_coo.data, dtype=y_array.schema.attr(attr).dtype
                        ),
                        dense_shape=(batch_size, y_shape[0]),
                    )
                    for attr in y_attribute_names
                )


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
) -> Iterator[Tuple[Union[tf.SparseTensor, np.ndarray], ...]]:
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
    x_shape = x_array.schema.domain.shape[1:]

    # https://github.com/tensorflow/tensorflow/issues/44565
    with ThreadPoolExecutor(max_workers=1) as executor:
        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )

            # COO to CSR transformation for batching and row slicing
            x_buffer_csr = to_csr(x_array, x_attribute_names[0], x_buffer, offset)

            # Split the buffer_size into batch_size chunks
            batch_offsets = np.arange(0, buffer_size, batch_size)

            # Shuffle offsets in case we need batch shuffling
            if batch_shuffle:
                np.random.shuffle(batch_offsets)

            for batch_offset in batch_offsets:
                x_batch = x_buffer_csr[batch_offset : batch_offset + batch_size]

                if x_batch.data.size == 0:
                    return

                y_batch = {
                    attr: data[batch_offset : batch_offset + batch_size]
                    for attr, data in y_buffer.items()
                }

                # Keep row records number for cross-check between x_array and y_array batches
                # Last index excluded shows to empty
                samples_num_x = x_batch.indptr[:-1]

                # Transform back to COO for torch.sparse_coo_tensor to digest
                x_batch_coo = x_batch.tocoo()
                x_coords = np.stack((x_batch_coo.row, x_batch_coo.col), axis=-1)

                # for the check slice the row dimension of y_array dense array
                if (
                    np.unique(samples_num_x).size
                    != y_batch[y_attribute_names[0]].shape[0]
                ):
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be of "
                        "equal domain extent inside the batch"
                    )

                yield tuple(
                    tf.SparseTensor(
                        indices=tf.constant(x_coords, dtype=tf.int64),
                        values=tf.constant(
                            x_batch_coo.data, dtype=x_array.schema.attr(attr).dtype
                        ),
                        dense_shape=(batch_size, x_shape[0]),
                    )
                    for attr in x_attribute_names
                ) + tuple(y_batch[attr] for attr in y_attribute_names)


def to_csr(
    array: tiledb.Array,
    attribute_names: str,
    buffer: Mapping[str, np.array],
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
