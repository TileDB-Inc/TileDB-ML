"""Functionality for loading data directly from TileDB arrays into the Tensorflow Data API."""
import dataclasses

import tiledb
import tensorflow as tf
import numpy as np
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
        For optimal reads from a TileDB array, it is recommended to set the batch size equal to the tile extent of the
        dimension we query (here, we always query the first dimension of a TileDB array) in order to get a slice (batch)
        of the data. For example, in case the tile extent of the first dimension of a TileDB array (x or y) is equal to
        32, it's recommended to set batch_size=32. Any batch size will work, but in case it's not equal the tile extent
        of the first dimension of the TileDB array, you won't achieve highest read speed. For more details on tiles,
        tile extent and indices in TileDB, please check here:
        https://docs.tiledb.com/main/solutions/tiledb-embedded/performance-tips/choosing-tiling-and-cell-layout#dense-arrays
        """
        cls.x = x_array
        cls.y = y_array

        # Check that x and y have the same number of rows
        if x_array.schema.domain.shape[0] != y_array.schema.domain.shape[0]:
            raise Exception(
                "X and Y should have the same number of rows, i.e., the 1st dimension "
                "of TileDB arrays X, Y should be of equal domain extent."
            )

        # Get number of observations
        rows = x_array.schema.domain.shape[0]

        # Get x and y shapes
        cls.x_shape = (None,) + x_array.schema.domain.shape[1:]
        cls.y_shape = (None,) + y_array.schema.domain.shape[1:]

        # Get x and y data types
        cls.x_dtype = x_array.schema.attr(0).dtype
        cls.y_dtype = y_array.schema.attr(0).dtype

        if isinstance(y_array, tiledb.SparseArray):
            return dataset_ops.Dataset.from_generator(
                generator=cls._generator_sparse_sparse,
                output_signature=(
                    tf.SparseTensorSpec(shape=cls.x_shape, dtype=cls.x_dtype),
                    tf.SparseTensorSpec(shape=cls.y_shape, dtype=cls.y_dtype)
                ),
                args=(rows, batch_size)
            )
        else:
            return dataset_ops.Dataset.from_generator(
                generator=cls._generator_sparse_dense,
                output_signature=(
                    tf.SparseTensorSpec(shape=cls.x_shape, dtype=cls.x_dtype),
                    tf.TensorSpec(shape=cls.y_shape, dtype=cls.y_dtype)
                ),
                args=(rows, batch_size)
            )

    @classmethod
    def _generator_sparse_sparse(cls, rows: int, batch_size: int) -> tuple:
        """
        A generator function that yields the next training batch.
        :param x: TileDB array. An opened TileDB array which contains features.
        :param y: TileDB array. An opened TileDB array which contains labels.
        :param rows: Integer. The number of observations in x, y datasets.
        :param batch_size: Integer. Size of batch, i.e., number of rows returned per call.
        :return: Tuple. Tuple that contains x and y batches.
        """
        # Loop over batches
        # https://github.com/tensorflow/tensorflow/issues/44565
        for offset in range(0, rows, batch_size):
            y_batch = cls.y[offset: offset + batch_size]
            x_batch = cls.x[offset: offset + batch_size]

            #TODO: Both for dense case support multiple attributes
            values_y = list(y_batch.items())[0][1]
            values_x = list(x_batch.items())[0][1]

            # Transform to TF COO format y data
            y_data = np.array(values_y).flatten()
            y_coords = []
            for i in range(0, cls.y.schema.domain.ndim):
                dim_name = cls.y.schema.domain.dim(i).name
                y_coords.append(np.array(y_batch[dim_name]))

            # Transform to TF COO format x data
            x_coords = []
            for i in range(0, cls.x.schema.domain.ndim):
                dim_name = cls.x.schema.domain.dim(i).name
                x_coords.append(np.array(x_batch[dim_name]))

            x_data = np.array(values_x).flatten()

            if x_data.shape[0] != y_data.shape[0]:
                raise Exception(
                    "X and Y should have the same number of rows, i.e., the 1st dimension "
                    "of TileDB arrays X, Y should be of equal domain extent inside the batch"
                )

            yield tf.sparse.SparseTensor(indices=constant_op.constant(list(zip(*x_coords)), dtype=tf.int64),
                                         values=constant_op.constant(x_data, dtype=tf.float32),
                                         dense_shape=(batch_size, cls.x_shape[1])), \
                  tf.sparse.SparseTensor(indices=constant_op.constant(list(zip(*y_coords)), dtype=tf.int64),
                                         values=constant_op.constant(y_data, dtype=cls.y_dtype),
                                         dense_shape=(batch_size, cls.y_shape[1]))

    @classmethod
    def _generator_sparse_dense(cls, rows: int, batch_size: int) -> tuple:
        """
        A generator function that yields the next training batch.
        :param x: TileDB array. An opened TileDB array which contains features.
        :param y: TileDB array. An opened TileDB array which contains labels.
        :param rows: Integer. The number of observations in x, y datasets.
        :param batch_size: Integer. Size of batch, i.e., number of rows returned per call.
        :return: Tuple. Tuple that contains x and y batches.
        """

        # Loop over batches
        # https://github.com/tensorflow/tensorflow/issues/44565
        for offset in range(0, rows, batch_size):
            y_batch = cls.y[offset: offset + batch_size]
            x_batch = cls.x[offset: offset + batch_size]

            #TODO: Both for dense case support multiple attributes
            values_y = list(y_batch.items())[0][1]
            values_x = list(x_batch.items())[0][1]

            x_coords = []
            for i in range(0, cls.x.schema.domain.ndim):
                dim_name = cls.x.schema.domain.dim(i).name
                x_coords.append(np.array(x_batch[dim_name]))

            x_data = np.array(values_x).flatten()

            # Detects empty lines due to sparsity inside the batch
            x_sparse_real_batch_size = x_coords[0].max(axis=0) - x_coords[0].min(axis=0)
            if values_y.shape[0] - x_sparse_real_batch_size != 1:
                raise Exception(
                    "X and Y should have the same number of rows, i.e., the 1st dimension "
                    "of TileDB arrays X, Y should be of equal domain extent inside the batch"
                )

            yield tf.sparse.SparseTensor(indices=constant_op.constant(list(zip(*x_coords)), dtype=tf.int64),
                                         values=constant_op.constant(x_data, dtype=tf.float32),
                                         dense_shape=(batch_size, cls.x_shape[1])), \
                  values_y
