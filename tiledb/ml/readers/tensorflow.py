"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from functools import partial
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple, Union

import tensorflow as tf

import tiledb

from . import _tensorflow_generators as tf_gen

# TODO: We have to track the following issues:
# - https://github.com/tensorflow/tensorflow/issues/47532
# - https://github.com/tensorflow/tensorflow/issues/47931


def TensorflowTileDBDataset(
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    batch_size: int,
    buffer_size: Optional[int] = None,
    x_attribute_names: Sequence[str] = (),
    y_attribute_names: Sequence[str] = (),
    batch_shuffle: bool = False,
    within_batch_shuffle: bool = False,
) -> tf.data.Dataset:
    """Return a tf.data.Dataset for loading data from TileDB arrays.

    :param x_array: TileDB array of the features.
    :param y_array: TileDB array of the labels.
    :param batch_size: Size of each batch.
    :param x_attribute_names: Attribute names of x_array.
    :param y_attribute_names: Attribute names of y_array.
    :param batch_shuffle: True for shuffling batches.
    :param within_batch_shuffle: True for shuffling records in each batch.
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
            offsets=range(0, rows, batch_size),
            batch_size=batch_size,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ),
        output_signature=output_signature,
    )
    # set the cardinality of the dataset to the number of rows in the array
    return dataset.apply(tf.data.experimental.assert_cardinality(rows))


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
