"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from functools import partial
from typing import Iterator, Optional, Sequence, Union

import sparse
import tensorflow as tf

import tiledb

from ._buffer_utils import get_attr_names, get_buffer_size
from ._tensor_gen import TileDBSparseTensorGenerator, tensor_generator

# TODO: We have to track the following issues:
# - https://github.com/tensorflow/tensorflow/issues/47532
# - https://github.com/tensorflow/tensorflow/issues/47931
# - https://github.com/tensorflow/tensorflow/issues/44565


def TensorflowTileDBDataset(
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    *,
    batch_size: int,
    buffer_bytes: Optional[int] = None,
    shuffle_buffer_size: int = 0,
    prefetch: int = tf.data.AUTOTUNE,
    x_attrs: Sequence[str] = (),
    y_attrs: Sequence[str] = (),
) -> tf.data.Dataset:
    """Return a tf.data.Dataset for loading data from TileDB arrays.

    :param x_array: TileDB array of the features.
    :param y_array: TileDB array of the labels.
    :param batch_size: Size of each batch.
    :param buffer_bytes: Maximum size (in bytes) of memory to allocate for reading from
        each array (default=`tiledb.default_ctx().config()["sm.memory_budget"]`).
    :param shuffle_buffer_size: Number of elements from which this dataset will sample.
    :param prefetch: Maximum number of batches that will be buffered when prefetching.
        By default, the buffer size is dynamically tuned.
    :param x_attrs: Attribute names of x_array.
    :param y_attrs: Attribute names of y_array.
    """
    # Check that x_array and y_array have the same number of rows
    rows: int = x_array.shape[0]
    if rows != y_array.shape[0]:
        raise ValueError("X and Y arrays must have the same number of rows")

    if not x_attrs:
        x_attrs = get_attr_names(x_array.schema)
    if not y_attrs:
        y_attrs = get_attr_names(y_array.schema)

    dataset = tf.data.Dataset.from_generator(
        partial(
            tensor_generator,
            x_array=x_array,
            y_array=y_array,
            x_buffer_size=get_buffer_size(x_array, x_attrs, buffer_bytes),
            y_buffer_size=get_buffer_size(y_array, y_attrs, buffer_bytes),
            x_attrs=x_attrs,
            y_attrs=y_attrs,
            sparse_generator_cls=TensorflowSparseTensorGenerator,
        ),
        output_signature=(
            *_iter_tensor_specs(x_array.schema, x_attrs),
            *_iter_tensor_specs(y_array.schema, y_attrs),
        ),
    )
    dataset = dataset.unbatch()
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    return dataset.batch(batch_size).prefetch(prefetch)


def _iter_tensor_specs(
    schema: tiledb.ArraySchema, attrs: Sequence[str]
) -> Iterator[Union[tf.TensorSpec, tf.SparseTensorSpec]]:
    cls = tf.SparseTensorSpec if schema.sparse else tf.TensorSpec
    for attr in attrs:
        yield cls(shape=(None, *schema.shape[1:]), dtype=schema.attr(attr).dtype)


class TensorflowSparseTensorGenerator(TileDBSparseTensorGenerator[tf.SparseTensor]):
    @staticmethod
    def _tensor_from_coo(coo: sparse.COO) -> tf.SparseTensor:
        return tf.SparseTensor(coo.coords.T, coo.data, coo.shape)
