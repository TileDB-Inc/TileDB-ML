"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from functools import partial
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple, Union

import tensorflow as tf
import wrapt

import tiledb

from . import _tensorflow_generators as tf_gen


def _get_attr_names(array: tiledb.Array) -> Sequence[str]:
    return tuple(array.schema.attr(idx).name for idx in range(array.schema.nattr))


def _get_signature(
    array: tiledb.Array, attrs: Sequence[str]
) -> Tuple[Union[tf.TensorSpec, tf.SparseTensorSpec], ...]:
    cls = (
        tf.SparseTensorSpec if isinstance(array, tiledb.SparseArray) else tf.TensorSpec
    )
    return tuple(
        cls(
            shape=(None, *array.schema.domain.shape[1:]),
            dtype=array.schema.attr(attr).dtype,
        )
        for attr in attrs
    )


class TensorflowTileDBDataset(wrapt.ObjectProxy):
    """Load data from a TileDB array to the Tensorflow Data API."""

    def __init__(
        self,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        batch_size: int,
        buffer_size: Optional[int] = None,
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
        **kwargs: Any,
    ):
        """
        Return a Tensorflow Dataset object which loads data from TileDB arrays.

        :param x_array: Array that contains features.
        :param y_array: Array that contains labels.
        :param batch_size: Size of each batch.
        :param x_attribute_names: Attribute names of x_array.
        :param y_attribute_names: Attribute names of y_array.
        """
        rows: int = x_array.schema.domain.shape[0]

        # Check that x_array and y_array have the same number of rows
        if rows != y_array.schema.domain.shape[0]:
            raise ValueError(
                "x_array and y_array should have the same number of rows, i.e. the "
                "first dimension of x_array and y_array should be of equal domain extent"
            )

        if buffer_size is None:
            buffer_size = batch_size
        elif buffer_size < batch_size:
            raise ValueError("Buffer size should be greater or equal to batch size")

        # If no attribute names are passed explicitly, return all attributes
        if not x_attribute_names:
            x_attribute_names = _get_attr_names(x_array)

        if not y_attribute_names:
            y_attribute_names = _get_attr_names(y_array)

        output_signature = _get_signature(x_array, x_attribute_names)
        output_signature += _get_signature(y_array, y_attribute_names)

        generator: Callable[..., Iterator[Tuple[Any, ...]]]
        if isinstance(x_array, tiledb.DenseArray):
            if not isinstance(y_array, tiledb.DenseArray):
                raise TypeError("if x_array is dense, y_array must be dense too")
            generator = tf_gen.dense_dense_generator
        elif isinstance(y_array, tiledb.DenseArray):
            generator = tf_gen.sparse_dense_generator
        else:
            generator = tf_gen.sparse_sparse_generator

        dataset = tf.data.Dataset.from_generator(
            partial(
                generator,
                x_array=x_array,
                y_array=y_array,
                x_attribute_names=x_attribute_names,
                y_attribute_names=y_attribute_names,
                rows=rows,
                batch_size=batch_size,
                buffer_size=buffer_size,
                **kwargs,
            ),
            output_signature=output_signature,
        )
        super().__init__(dataset)
        self._rows = rows

    def __len__(self) -> int:
        return self._rows


class TensorflowTileDBDenseDataset(TensorflowTileDBDataset):
    """Load data from a dense TileDB array to the Tensorflow Data API."""

    def __init__(
        self,
        x_array: tiledb.DenseArray,
        y_array: tiledb.DenseArray,
        batch_size: int,
        buffer_size: Optional[int] = None,
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
        batch_shuffle: bool = False,
        within_batch_shuffle: bool = False,
    ):
        """
        Return a Tensorflow Dataset object which loads data from TileDB arrays
        by employing a generator.

        For optimal reads from a TileDB array, it is recommended to set the batch size
        equal to the tile extent of the dimension we query (here, we always query the
        first dimension of a TileDB array) in order to get a slice (batch) of the data.
        For example, in case the tile extent of the first dimension of a TileDB array
        (x or y) is equal to 32, it's recommended to set batch_size=32. Any batch size
        will work, but in case it's not equal the tile extent of the first dimension of
        the TileDB array, you won't achieve highest read speed. For more details on tiles,
        tile extent and indices in TileDB, please check here:
        https://docs.tiledb.com/main/how-to/performance/performance-tips/choosing-tiling-and-cell-layout#dense-arrays

        :param x_array: Array that contains features.
        :param y_array: Array that contains labels.
        :param batch_size: The size of the batch that the implemented _generator method
            will return.
        :param batch_shuffle: True if we want to shuffle batches.
        :param within_batch_shuffle: True if we want to shuffle records in each batch.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        """
        if isinstance(x_array, tiledb.SparseArray):
            raise TypeError(
                "TensorflowTileDBDenseDataset accepts tiledb.DenseArray instances only"
            )

        super().__init__(
            x_array=x_array,
            y_array=y_array,
            batch_size=batch_size,
            buffer_size=buffer_size,
            x_attribute_names=x_attribute_names,
            y_attribute_names=y_attribute_names,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        )


class TensorflowTileDBSparseDataset(TensorflowTileDBDataset):
    """Load data from a sparse TileDB array to the Tensorflow Data API."""

    # We have to track the following issues on *working with sparse input*
    # and *convert SparseTensor to Tensor* respectively.
    # TODO: TF https://github.com/tensorflow/tensorflow/issues/47532
    # TODO: TF https://github.com/tensorflow/tensorflow/issues/47931

    def __init__(
        self,
        x_array: tiledb.SparseArray,
        y_array: tiledb.Array,
        batch_size: int,
        buffer_size: Optional[int],
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
        batch_shuffle: bool = False,
    ):
        """
        Return a Tensorflow Dataset object which loads data from TileDB arrays by
        employing a generator.

        :param x_array: Array that contains features.
        :param y_array: Array that contains labels.
        :param batch_size: The size of the batch that the implemented _generator method
            will return.
        :param batch_shuffle: True if we want to shuffle batches.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        """
        if isinstance(x_array, tiledb.DenseArray):
            raise TypeError(
                "TensorflowTileDBSparseDataset accepts tiledb.SparseArray instances only"
            )

        super().__init__(
            x_array=x_array,
            y_array=y_array,
            batch_size=batch_size,
            buffer_size=buffer_size,
            x_attribute_names=x_attribute_names,
            y_attribute_names=y_attribute_names,
            batch_shuffle=batch_shuffle,
        )
