"""Functionality for loading data directly from TileDB arrays into the Tensorflow Data API."""
import tiledb
import tensorflow as tf
import numpy as np
from typing import List
from functools import partial
from tensorflow.python.framework import constant_op
from tensorflow.python.data.ops import dataset_ops


class TensorflowTileDBSparseDataset(tf.data.Dataset):
    """
    Class that implements all functionality needed to load sparse data from TileDB directly to the
    Tensorflow Data API, by employing generators.
    """

    def __new__(
        cls,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        x_attribute_names: List[str],
        y_attribute_names: List[str],
        batch_size: int,
    ):
        """
        Returns a Tensorflow Dataset object which loads data from TileDB arrays by employing a generator.
        :param x_array: TileDB Sparse Array. Array that contains features.
        :param y_array: TileDB Dense/Sparse Array. Array that contains labels.
        :param x_attribute_names: List of str. A list that contains the attribute names of TileDB array x.
        :param y_attribute_names: List of str. A list that contains the attribute names of TileDB array y.
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
                rows=rows,
                batch_size=batch_size,
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
                rows=rows,
                batch_size=batch_size,
            )

            return dataset_ops.Dataset.from_generator(
                generator=generator_,
                output_signature=x_signature + y_signature,
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
        cls,
        x: tiledb.Array,
        y: tiledb.Array,
        x_attribute_names: List[str],
        y_attribute_names: List[str],
        rows: int,
        batch_size: int,
    ) -> tuple:
        """
        A generator function that yields the next training batch.
        :param x: TileDB Sparse array. An opened TileDB array which contains features.
        :param y: TileDB Sparse array. An opened TileDB array which contains labels.
        :param x_attribute_names: List of str. A list that contains the attribute names of TileDB array x.
        :param y_attribute_names: List of str. A list that contains the attribute names of TileDB array y.
        :param rows: Integer. The number of observations in x, y datasets.
        :param batch_size: Integer. Size of batch, i.e., number of rows returned per call.
        :return: Tuple. Tuple that contains x and y batches.
        """

        x_shape = x.schema.domain.shape[1:]
        y_shape = y.schema.domain.shape[1:]

        # Loop over batches
        # https://github.com/tensorflow/tensorflow/issues/44565
        for offset in range(0, rows, batch_size):
            x_batch = x[offset : offset + batch_size]
            y_batch = y[offset : offset + batch_size]

            # Transform to TF COO format x, y data
            x_coords = []
            for i in range(0, x.schema.domain.ndim):
                dim_name = x.schema.domain.dim(i).name
                x_coords.append(np.array(x_batch[dim_name]))

            y_coords = []
            for i in range(0, y.schema.domain.ndim):
                dim_name = y.schema.domain.dim(i).name
                y_coords.append(np.array(y_batch[dim_name]))

            # Normalise indices for torch.sparse.Tensor We want the coords indices in every iteration
            # to be in the range of [0, self.batch_size] so the torch.sparse.Tensors can be created batch-wise.
            # If we do not normalise the sparse tensor is being created but with a dimension [0, max(coord_index)],
            # which is overkill
            x_coords[0] -= x_coords[0].min()
            y_coords[0] -= y_coords[0].min()

            cls.__check_row_dims(x_coords[0], y_coords[0], sparse=True)

            yield tuple(
                tf.sparse.SparseTensor(
                    indices=constant_op.constant(list(zip(*x_coords)), dtype=tf.int64),
                    values=constant_op.constant(
                        x_batch[attr].ravel(), dtype=x.schema.attr(attr).dtype
                    ),
                    dense_shape=(batch_size, x_shape[0]),
                )
                for attr in x_attribute_names
            ) + tuple(
                tf.sparse.SparseTensor(
                    indices=constant_op.constant(list(zip(*y_coords)), dtype=tf.int64),
                    values=constant_op.constant(
                        y_batch[attr].ravel(), dtype=y.schema.attr(attr).dtype
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
        x_attribute_names: List[str],
        y_attribute_names: List[str],
        rows: int,
        batch_size: int,
    ) -> tuple:
        """
        A generator function that yields the next training batch.
        :param x: TileDB Sparse array. An opened TileDB array which contains features.
        :param y: TileDB Dense array. An opened TileDB array which contains labels.
        :param x_attribute_names: List of str. A list that contains the attribute names of TileDB array x.
        :param y_attribute_names: List of str. A list that contains the attribute names of TileDB array y.
        :param rows: Integer. The number of observations in x, y datasets.
        :param batch_size: Integer. Size of batch, i.e., number of rows returned per call.
        :return: Tuple. Tuple that contains x and y batches.
        """
        x_shape = x.schema.domain.shape[1:]

        # Loop over batches
        # https://github.com/tensorflow/tensorflow/issues/44565
        for offset in range(0, rows, batch_size):
            x_batch = x[offset : offset + batch_size]
            y_batch = y[offset : offset + batch_size]

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
            cls.__check_row_dims(
                x_coords[0], y_batch[y_attribute_names[0]], sparse=False
            )

            yield tuple(
                tf.sparse.SparseTensor(
                    indices=constant_op.constant(list(zip(*x_coords)), dtype=tf.int64),
                    values=constant_op.constant(
                        x_batch[attr].flatten(), dtype=x.schema.attr(attr).dtype
                    ),
                    dense_shape=(batch_size, x_shape[0]),
                )
                for attr in x_attribute_names
            ) + tuple(y_batch[attr] for attr in y_attribute_names)
