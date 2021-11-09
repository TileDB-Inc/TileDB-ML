"""Functionality for loading data directly from TileDB arrays into the Tensorflow Data API."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Iterator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op

import tiledb
from tiledb.ml._parallel_utils import run_io_tasks_in_parallel


class TensorflowTileDBSparseDataset(tf.data.Dataset):
    """
    Class that implements all functionality needed to load sparse data from TileDB directly to the
    Tensorflow Data API, by employing generators.
    """

    # We have to track the following issues on *working with sparse input*
    # and *convert SparseTensor to Tensor* respectively.
    # TODO: TF https://github.com/tensorflow/tensorflow/issues/47532
    # TODO: TF https://github.com/tensorflow/tensorflow/issues/47931

    def __new__(
        cls,
        x_array: tiledb.SparseArray,
        y_array: tiledb.Array,
        batch_size: int,
        buffer_size: Optional[int],
        batch_shuffle: bool = False,
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
    ) -> tf.data.Dataset:
        """
        Return a Tensorflow Dataset object which loads data from TileDB arrays by
        employing a generator.

        :param x_array: Array that contains features.
        :param y_array: Array that contains labels.
        :param batch_size: The size of the batch that the implemented _generator method will return.
        :param batch_shuffle: True if we want to shuffle batches.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        """
        if type(x_array) is tiledb.DenseArray:
            raise TypeError(
                "TensorflowTileDBSparseDataset class should be used with tiledb.SparseArray representation"
            )

        # Check that x and y have the same number of rows
        if x_array.schema.domain.shape[0] != y_array.schema.domain.shape[0]:
            raise ValueError(
                "X and Y should have the same number of rows, i.e., the 1st dimension "
                "of TileDB arrays X, Y should be of equal domain extent."
            )

        # If no attribute names are passed explicitly, return all attributes
        if not x_attribute_names:
            x_attribute_names = [
                x_array.schema.attr(idx).name for idx in range(x_array.schema.nattr)
            ]

        if not y_attribute_names:
            y_attribute_names = [
                y_array.schema.attr(idx).name for idx in range(y_array.schema.nattr)
            ]

        # Set the buffer_size appropriately and check its size
        buffer_size_checked = buffer_size or batch_size
        if buffer_size_checked < batch_size:
            raise ValueError("Buffer size should be geq to the batch size.")

        # Get number of observations
        rows = x_array.schema.domain.shape[0]

        # Get x and y shapes
        x_shape = (None,) + x_array.schema.domain.shape[1:]
        y_shape = (None,) + y_array.schema.domain.shape[1:]

        if isinstance(y_array, tiledb.SparseArray):
            # Signatures for x and y
            x_signature = tuple(
                tf.SparseTensorSpec(
                    shape=x_shape, dtype=x_array.schema.attr(attr).dtype
                )
                for attr in x_attribute_names
            )
            y_signature = tuple(
                tf.SparseTensorSpec(
                    shape=y_shape, dtype=y_array.schema.attr(attr).dtype
                )
                for attr in y_attribute_names
            )

            generator_ = partial(
                cls._generator_sparse_sparse,
                x=x_array,
                y=y_array,
                x_attribute_names=x_attribute_names,
                y_attribute_names=y_attribute_names,
                rows=rows,
                batch_size=batch_size,
                buffer_size=buffer_size_checked,
                batch_shuffle=batch_shuffle,
            )

            return dataset_ops.Dataset.from_generator(
                generator=generator_,
                output_signature=x_signature + y_signature,
            )
        else:
            # Signatures for x and y
            x_signature = tuple(
                tf.SparseTensorSpec(
                    shape=x_shape, dtype=x_array.schema.attr(attr).dtype
                )
                for attr in x_attribute_names
            )
            y_signature = tuple(
                tf.TensorSpec(shape=y_shape, dtype=y_array.schema.attr(attr).dtype)
                for attr in y_attribute_names
            )

            generator_ = partial(
                cls._generator_sparse_dense,
                x=x_array,
                y=y_array,
                x_attribute_names=x_attribute_names,
                y_attribute_names=y_attribute_names,
                rows=rows,
                batch_size=batch_size,
                buffer_size=buffer_size_checked,
                batch_shuffle=batch_shuffle,
            )

            return dataset_ops.Dataset.from_generator(
                generator=generator_,
                output_signature=x_signature + y_signature,
            )

    @staticmethod
    def __check_row_dims(
        x_row_idx: np.ndarray, y_row_idx: np.ndarray, sparse: bool
    ) -> None:
        """
        Check the row dimensionality of x,y in case y is sparse or not

        :param x_row_idx: The row indices x_coords of x Sparse Array of the dimension
            that is being batched
        :param y_row_idx: if y is sparse array, the row indices y_coords of the dimension
            that is being batched. If y is dense array, data of y
        :raises ValueError: If unique coords idx of x and y mismatch (both-sparse) or
            when unique coords idx of x mismatch y elements when y is Dense
        """
        if np.unique(x_row_idx).size != (
            np.unique(y_row_idx).size if sparse else y_row_idx.shape[0]
        ):
            raise ValueError(
                "X and Y should have the same number of rows, i.e., the 1st dimension "
                "of TileDB arrays X, Y should be of equal domain extent inside the batch."
            )

    @classmethod
    def __to_csr(
        cls,
        array: tiledb.Array,
        attribute_names: str,
        buffer: Mapping[str, np.array],
        offset: int,
    ) -> csr_matrix:
        """
        :param array_id: The matrix on which the transformation will have effect 'X' for x_array and 'Y' for y_array
        :param buffer: The buffered slice of the matrix to be batched
        :param offset: The starting offset of the buffered slice
        :returns A CSR representation of the buffered slice of the matrix
        """

        dim = array.schema.domain.dim
        row = buffer[dim(0).name]
        col = buffer[dim(1).name]
        # Normalise indices for torch.sparse.Tensor We want the coords indices in every iteration to be
        # in the range of [0, self.batch_size] so the torch.sparse.Tensors can be created batch-wise. If
        # we do not normalise the sparse tensor is being created but with a dimension [0,
        # max(coord_index)], which is overkill
        row_size_norm = row.max() - row.min() + 1
        col_size_norm = col.max() + 1
        buffer_csr = csr_matrix(
            (
                buffer[attribute_names],
                (row - offset, col),
            ),
            shape=(row_size_norm, col_size_norm),
        )
        return buffer_csr

    @classmethod
    def _generator_sparse_sparse(
        cls,
        x: tiledb.Array,
        y: tiledb.Array,
        x_attribute_names: Sequence[str],
        y_attribute_names: Sequence[str],
        rows: int,
        batch_size: int,
        buffer_size: Optional[int],
        batch_shuffle: bool,
    ) -> Iterator[Tuple[tf.sparse.SparseTensor, ...]]:
        """
        Generator for yielding training batches.

        :param x: An opened TileDB array which contains features.
        :param y: An opened TileDB array which contains labels.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        :param rows: The number of observations in x, y datasets.
        :param batch_size: Size of batch, i.e., number of rows returned per call.
        :param batch_shuffle: True if we want to shuffle batches.
        :return: An iterator of x and y batches.
        """
        x_shape = x.schema.domain.shape[1:]
        y_shape = y.schema.domain.shape[1:]

        offsets = np.arange(0, rows, buffer_size)

        # Loop over batches
        # https://github.com/tensorflow/tensorflow/issues/44565
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in offsets:
                x_buffer, y_buffer = run_io_tasks_in_parallel(
                    executor,
                    (x, y),
                    buffer_size,
                    offset,
                )

                # COO to CSR transformation for batching and row slicing
                x_buffer_csr = cls.__to_csr(x, x_attribute_names[0], x_buffer, offset)
                y_buffer_csr = cls.__to_csr(y, y_attribute_names[0], y_buffer, offset)

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

                    # Keep row records number for cross-check between X and Y batches. Last index excluded shows to
                    # empty
                    samples_num_x = x_batch.indptr[:-1]

                    # Transform back to COO for torch.sparse_coo_tensor to digest
                    x_batch_coo = x_batch.tocoo()
                    x_coords = np.stack((x_batch_coo.row, x_batch_coo.col), axis=-1)

                    # Keep row records number for cross-check between X and Y batches. Last index excluded shows
                    # to empty
                    samples_num_y = y_batch.indptr[:-1]

                    # Transform back to COO for torch.sparse_coo_tensor to digest
                    y_batch_coo = y_batch.tocoo()
                    y_coords = np.stack((y_batch_coo.row, y_batch_coo.col), axis=-1)

                    cls.__check_row_dims(samples_num_x, samples_num_y, sparse=True)

                    yield tuple(
                        tf.sparse.SparseTensor(
                            indices=constant_op.constant(x_coords, dtype=tf.int64),
                            values=constant_op.constant(
                                x_batch_coo.data, dtype=x.schema.attr(attr).dtype
                            ),
                            dense_shape=(batch_size, x_shape[0]),
                        )
                        for attr in x_attribute_names
                    ) + tuple(
                        tf.sparse.SparseTensor(
                            indices=constant_op.constant(y_coords, dtype=tf.int64),
                            values=constant_op.constant(
                                y_batch_coo.data, dtype=y.schema.attr(attr).dtype
                            ),
                            dense_shape=(batch_size, y_shape[0]),
                        )
                        for attr in y_attribute_names
                    )

    @classmethod
    def _generator_sparse_dense(
        cls,
        x: tiledb.Array,
        y: tiledb.Array,
        x_attribute_names: Sequence[str],
        y_attribute_names: Sequence[str],
        rows: int,
        batch_size: int,
        buffer_size: Optional[int],
        batch_shuffle: bool,
    ) -> Iterator[Tuple[Union[tf.sparse.SparseTensor, np.ndarray], ...]]:
        """
        Generator for yielding training batches.

        :param x: An opened TileDB array which contains features.
        :param y: An opened TileDB array which contains labels.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        :param rows: The number of observations in x, y datasets.
        :param batch_size: Size of batch, i.e., number of rows returned per call.
        :param batch_shuffle: True if we want to shuffle batches.
        :return: An iterator of x and y batches.
        """
        x_shape = x.schema.domain.shape[1:]

        offsets = np.arange(0, rows, buffer_size)

        # Loop over batches
        # https://github.com/tensorflow/tensorflow/issues/44565
        with ThreadPoolExecutor(max_workers=1) as executor:
            for offset in offsets:
                x_buffer, y_buffer = run_io_tasks_in_parallel(
                    executor,
                    (x, y),
                    buffer_size,
                    offset,
                )

                # COO to CSR transformation for batching and row slicing
                x_buffer_csr = cls.__to_csr(x, x_attribute_names[0], x_buffer, offset)

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

                    # Keep row records number for cross-check between X and Y batches. Last index excluded shows to
                    # empty
                    samples_num_x = x_batch.indptr[:-1]

                    # Transform back to COO for torch.sparse_coo_tensor to digest
                    x_batch_coo = x_batch.tocoo()
                    x_coords = np.stack((x_batch_coo.row, x_batch_coo.col), axis=-1)

                    # for the check slice the row dimension of y dense array
                    cls.__check_row_dims(
                        samples_num_x, y_batch[y_attribute_names[0]], sparse=False
                    )

                    yield tuple(
                        tf.sparse.SparseTensor(
                            indices=constant_op.constant(x_coords, dtype=tf.int64),
                            values=constant_op.constant(
                                x_batch_coo.data, dtype=x.schema.attr(attr).dtype
                            ),
                            dense_shape=(batch_size, x_shape[0]),
                        )
                        for attr in x_attribute_names
                    ) + tuple(y_batch[attr] for attr in y_attribute_names)
