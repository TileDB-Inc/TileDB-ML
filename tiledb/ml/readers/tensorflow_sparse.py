"""Functionality for loading data directly from TileDB arrays into the Tensorflow Data API."""
import tiledb
import tensorflow as tf
import numpy as np
from functools import partial
from tensorflow.python.framework import constant_op
from tensorflow.python.data.ops import dataset_ops


class TensorflowTileDBSparseDataset(tf.data.Dataset):
    """
    Class that implements all functionality needed to load sparse data from TileDB directly to the
    Tensorflow Data API, by employing generators.
    """

    def __new__(cls, x_array: tiledb.Array, y_array: tiledb.Array, batch_size: int):
        """
        Returns a Tensorflow Dataset object which loads data from TileDB arrays by employing a generator.
        :param x_array: TileDB Sparse Array. Array that contains features.
        :param y_array: TileDB Dense/Sparse Array. Array that contains labels.
        :param batch_size: Integer. The size of the batch that the implemented _generator method will return.
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

        # Get number of observations
        rows = x_array.schema.domain.shape[0]

        # Get x and y shapes
        x_shape = (None,) + x_array.schema.domain.shape[1:]
        y_shape = (None,) + y_array.schema.domain.shape[1:]

        # Get x and y data types
        x_dtype = x_array.schema.attr(0).dtype
        y_dtype = y_array.schema.attr(0).dtype

        if isinstance(y_array, tiledb.SparseArray):
            generator_ = partial(
                cls._generator_sparse_sparse,
                x=x_array,
                y=y_array,
                rows=rows,
                batch_size=batch_size,
            )
            return dataset_ops.Dataset.from_generator(
                generator=generator_,
                output_signature=(
                    tf.SparseTensorSpec(shape=x_shape, dtype=x_dtype),
                    tf.SparseTensorSpec(shape=y_shape, dtype=y_dtype),
                ),
            )
        else:
            generator_ = partial(
                cls._generator_sparse_dense,
                x=x_array,
                y=y_array,
                rows=rows,
                batch_size=batch_size,
            )
            return dataset_ops.Dataset.from_generator(
                generator=generator_,
                output_signature=(
                    tf.SparseTensorSpec(shape=x_shape, dtype=x_dtype),
                    tf.TensorSpec(shape=y_shape, dtype=y_dtype),
                ),
            )

    @staticmethod
    def __check_row_dims(x_row_idx: np.array, y_row_idx: np.array, sparse: bool):
        """
        Check the row dimensionality of x,y in case y is sparse or not

        Parameters:

            x_row_idx (np.array): Expects the row indices x_coords of x Sparse Array of the
            dimension that is being batched

            y_row_idx (np.array): if y Sparse Array -> Expects the row indices y_coords of the
            dimension that is being batched else if y is Dense Array -> data of y

        Raises:
            ValueError: If unique coords idx of x and y mismatch (both-sparse) or
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
    def _generator_sparse_sparse(
        cls, x: tiledb.Array, y: tiledb.Array, rows: int, batch_size: int
    ) -> tuple:
        """
        A generator function that yields the next training batch.
        :param x: TileDB Sparse array. An opened TileDB array which contains features.
        :param y: TileDB Sparse array. An opened TileDB array which contains labels.
        :param rows: Integer. The number of observations in x, y datasets.
        :param batch_size: Integer. Size of batch, i.e., number of rows returned per call.
        :return: Tuple. Tuple that contains x and y batches.
        """

        x_shape = x.schema.domain.shape[1:]
        y_shape = y.schema.domain.shape[1:]

        x_dtype = x.schema.attr(0).dtype
        y_dtype = y.schema.attr(0).dtype

        # Loop over batches
        # https://github.com/tensorflow/tensorflow/issues/44565
        for offset in range(0, rows, batch_size):
            y_batch = y[offset : offset + batch_size]
            x_batch = x[offset : offset + batch_size]

            # TODO: Both for dense case support multiple attributes
            values_y = list(y_batch.items())[0][1]
            values_x = list(x_batch.items())[0][1]

            # Transform to TF COO format y data
            y_data = np.array(values_y).ravel()
            x_data = np.array(values_x).ravel()

            y_coords = []
            for i in range(0, y.schema.domain.ndim):
                dim_name = y.schema.domain.dim(i).name
                y_coords.append(np.array(y_batch[dim_name]))

            # Transform to TF COO format x data
            x_coords = []
            for i in range(0, x.schema.domain.ndim):
                dim_name = x.schema.domain.dim(i).name
                x_coords.append(np.array(x_batch[dim_name]))

            # Normalise indices for torch.sparse.Tensor We want the coords indices in every iteration
            # to be in the range of [0, self.batch_size] so the torch.sparse.Tensors can be created batch-wise.
            # If we do not normalise the sparse tensor is being created but with a dimension [0, max(coord_index)],
            # which is overkill
            x_coords[0] -= x_coords[0].min()
            y_coords[0] -= y_coords[0].min()

            cls.__check_row_dims(x_coords[0], y_coords[0], sparse=True)

            yield tf.sparse.SparseTensor(
                indices=constant_op.constant(list(zip(*x_coords)), dtype=tf.int64),
                values=constant_op.constant(x_data, dtype=x_dtype),
                dense_shape=(batch_size, x_shape[0]),
            ), tf.sparse.SparseTensor(
                indices=constant_op.constant(list(zip(*y_coords)), dtype=tf.int64),
                values=constant_op.constant(y_data, dtype=y_dtype),
                dense_shape=(batch_size, y_shape[0]),
            )

    @classmethod
    def _generator_sparse_dense(
        cls, x: tiledb.Array, y: tiledb.Array, rows: int, batch_size: int
    ) -> tuple:
        """
        A generator function that yields the next training batch.
        :param x: TileDB Sparse array. An opened TileDB array which contains features.
        :param y: TileDB Dense array. An opened TileDB array which contains labels.
        :param rows: Integer. The number of observations in x, y datasets.
        :param batch_size: Integer. Size of batch, i.e., number of rows returned per call.
        :return: Tuple. Tuple that contains x and y batches.
        """
        x_shape = x.schema.domain.shape[1:]
        x_dtype = x.schema.attr(0).dtype

        # Loop over batches
        # https://github.com/tensorflow/tensorflow/issues/44565
        for offset in range(0, rows, batch_size):
            y_batch = y[offset : offset + batch_size]
            x_batch = x[offset : offset + batch_size]

            # TODO: Both for dense case support multiple attributes
            values_y = list(y_batch.items())[0][1]
            values_x = list(x_batch.items())[0][1]

            x_coords = []
            for i in range(0, x.schema.domain.ndim):
                dim_name = x.schema.domain.dim(i).name
                x_coords.append(np.array(x_batch[dim_name]))

            # Normalise indices for torch.sparse.Tensor We want the coords indices in every iteration
            # to be in the range of [0, self.batch_size] so the torch.sparse.Tensors can be created batch-wise.
            # If we do not normalise the sparse tensor is being created but with a dimension [0, max(coord_index)],
            # which is overkill
            x_coords[0] -= x_coords[0].min()

            # for the check slice the row dimension of y dense array
            cls.__check_row_dims(x_coords[0], values_y, sparse=False)

            x_data = np.array(values_x).flatten()

            yield tf.sparse.SparseTensor(
                indices=constant_op.constant(list(zip(*x_coords)), dtype=tf.int64),
                values=constant_op.constant(x_data, dtype=x_dtype),
                dense_shape=(batch_size, x_shape[0]),
            ), values_y
