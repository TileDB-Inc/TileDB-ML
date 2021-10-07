"""Functionality for loading data directly from TileDB arrays into the Tensorflow Data API."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Iterator, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
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
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
    ) -> tf.data.Dataset:
        """
        Return a Tensorflow Dataset object which loads data from TileDB arrays by
        employing a generator.

        :param x_array: Array that contains features.
        :param y_array: Array that contains labels.
        :param batch_size: The size of the batch that the implemented _generator method
            will return.
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
    def _generator_sparse_sparse(
        cls,
        x: tiledb.Array,
        y: tiledb.Array,
        x_attribute_names: Sequence[str],
        y_attribute_names: Sequence[str],
        rows: int,
        batch_size: int,
    ) -> Iterator[Tuple[tf.sparse.SparseTensor, ...]]:
        """
        Generator for yielding training batches.

        :param x: An opened TileDB array which contains features.
        :param y: An opened TileDB array which contains labels.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        :param rows: The number of observations in x, y datasets.
        :param batch_size: Size of batch, i.e., number of rows returned per call.
        :return: An iterator of x and y batches.
        """
        x_shape = x.schema.domain.shape[1:]
        y_shape = y.schema.domain.shape[1:]

        # Loop over batches
        # https://github.com/tensorflow/tensorflow/issues/44565
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in range(0, rows, batch_size):
                x_batch, y_batch = run_io_tasks_in_parallel(
                    executor, (x, y), batch_size, offset
                )

                # Transform to TF COO format x, y data
                x_coords = []
                for i in range(0, x.schema.domain.ndim):
                    dim_name = x.schema.domain.dim(i).name
                    x_coords.append(np.array(x_batch[dim_name]))

                y_coords = []
                for i in range(0, y.schema.domain.ndim):
                    dim_name = y.schema.domain.dim(i).name
                    y_coords.append(np.array(y_batch[dim_name]))

                # Normalise indices for torch.sparse.Tensor We want the coords indices in
                # every iteration to be in the range of [0, self.batch_size] so the
                # torch.sparse.Tensors can be created batch-wise. If we do not normalise the
                # sparse tensor is being created but with a dimension [0, max(coord_index)],
                # which is overkill
                x_coords[0] -= x_coords[0].min()
                y_coords[0] -= y_coords[0].min()

                cls.__check_row_dims(x_coords[0], y_coords[0], sparse=True)

                yield tuple(
                    tf.sparse.SparseTensor(
                        indices=constant_op.constant(
                            list(zip(*x_coords)), dtype=tf.int64
                        ),
                        values=constant_op.constant(
                            x_batch[attr].ravel(), dtype=x.schema.attr(attr).dtype
                        ),
                        dense_shape=(batch_size, x_shape[0]),
                    )
                    for attr in x_attribute_names
                ) + tuple(
                    tf.sparse.SparseTensor(
                        indices=constant_op.constant(
                            list(zip(*y_coords)), dtype=tf.int64
                        ),
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
        x_attribute_names: Sequence[str],
        y_attribute_names: Sequence[str],
        rows: int,
        batch_size: int,
    ) -> Iterator[Tuple[Union[tf.sparse.SparseTensor, np.ndarray], ...]]:
        """
        Generator for yielding training batches.

        :param x: An opened TileDB array which contains features.
        :param y: An opened TileDB array which contains labels.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        :param rows: The number of observations in x, y datasets.
        :param batch_size: Size of batch, i.e., number of rows returned per call.
        :return: An iterator of x and y batches.
        """
        x_shape = x.schema.domain.shape[1:]

        # Loop over batches
        # https://github.com/tensorflow/tensorflow/issues/44565
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in range(0, rows, batch_size):
                x_batch, y_batch = run_io_tasks_in_parallel(
                    executor, (x, y), batch_size, offset
                )

                x_coords = []
                for i in range(0, x.schema.domain.ndim):
                    dim_name = x.schema.domain.dim(i).name
                    x_coords.append(np.array(x_batch[dim_name]))

                # Normalise indices for torch.sparse.Tensor We want the coords indices in
                # every iteration to be in the range of [0, self.batch_size] so the
                # torch.sparse.Tensors can be created batch-wise. If we do not normalise the
                # sparse tensor is being created but with a dimension [0, max(coord_index)],
                # which is overkill
                x_coords[0] -= x_coords[0].min()

                # for the check slice the row dimension of y dense array
                cls.__check_row_dims(
                    x_coords[0], y_batch[y_attribute_names[0]], sparse=False
                )

                yield tuple(
                    tf.sparse.SparseTensor(
                        indices=constant_op.constant(
                            list(zip(*x_coords)), dtype=tf.int64
                        ),
                        values=constant_op.constant(
                            x_batch[attr].flatten(), dtype=x.schema.attr(attr).dtype
                        ),
                        dense_shape=(batch_size, x_shape[0]),
                    )
                    for attr in x_attribute_names
                ) + tuple(y_batch[attr] for attr in y_attribute_names)
