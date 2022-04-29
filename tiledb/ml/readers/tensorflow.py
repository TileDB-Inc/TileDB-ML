"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import sparse
import tensorflow as tf

import tiledb

from ._buffer_utils import get_attr_names, get_buffer_size
from ._tensor_gen import (
    TileDBNumpyGenerator,
    TileDBSparseGenerator,
    TileDBTensorGenerator,
    iter_slices,
)


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
    num_workers: int = 0,
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
    :param num_workers: If greater than zero, create a threadpool of `num_workers` threads
        used to fetch inputs asynchronously and in parallel. Note: yielded batches may
        be shuffled even if `shuffle_buffer_size` is zero when `num_workers` > 1.
    """
    # Check that x_array and y_array have the same number of rows
    rows: int = x_array.shape[0]
    if rows != y_array.shape[0]:
        raise ValueError("X and Y arrays must have the same number of rows")

    if not x_attrs:
        x_attrs = get_attr_names(x_array.schema)
    if not y_attrs:
        y_attrs = get_attr_names(y_array.schema)

    x_gen = _get_tensor_generator(x_array, x_attrs)
    y_gen = _get_tensor_generator(y_array, y_attrs)
    x_buffer_size = get_buffer_size(x_array, x_attrs, buffer_bytes)
    y_buffer_size = get_buffer_size(y_array, y_attrs, buffer_bytes)

    def bounded_dataset(bounds: Union[Tuple[int, int], tf.Tensor]) -> tf.data.Dataset:
        x_dataset = tf.data.Dataset.from_generator(
            x_gen.iter_tensors,
            args=(x_buffer_size, bounds[0], bounds[1]),
            output_signature=_get_tensor_specs(x_array.schema, x_attrs),
        )
        y_dataset = tf.data.Dataset.from_generator(
            y_gen.iter_tensors,
            args=(y_buffer_size, bounds[0], bounds[1]),
            output_signature=_get_tensor_specs(y_array.schema, y_attrs),
        )
        return tf.data.Dataset.zip((x_dataset.unbatch(), y_dataset.unbatch()))

    if num_workers:
        per_worker = int(math.ceil(rows / num_workers))
        offsets = [(s.start, s.stop) for s in iter_slices(0, rows, per_worker)]
        offsets_tensor = tf.convert_to_tensor(offsets, dtype=tf.int64)
        offsets_dataset = tf.data.Dataset.from_tensor_slices(offsets_tensor)
        dataset = offsets_dataset.interleave(
            bounded_dataset, num_parallel_calls=num_workers, deterministic=False
        )
    else:
        dataset = bounded_dataset((0, rows))

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    return dataset.batch(batch_size).prefetch(prefetch)


TensorSpec = Union[tf.TensorSpec, tf.SparseTensorSpec]


def _get_tensor_specs(
    schema: tiledb.ArraySchema, attrs: Sequence[str]
) -> Union[TensorSpec, Sequence[TensorSpec]]:
    cls = tf.SparseTensorSpec if schema.sparse else tf.TensorSpec
    specs = (
        cls(shape=(None, *schema.shape[1:]), dtype=schema.attr(attr).dtype)
        for attr in attrs
    )
    return tuple(specs) if len(attrs) > 1 else next(specs)


def _get_tensor_generator(
    array: tiledb.Array, attrs: Sequence[str]
) -> TileDBTensorGenerator[Union[np.ndarray, tf.SparseTensor]]:
    if not array.schema.sparse:
        return TileDBNumpyGenerator(array, attrs)
    else:
        return TileDBSparseGenerator(array, attrs, _coo_to_sparse_tensor)


def _coo_to_sparse_tensor(coo: sparse.COO) -> tf.SparseTensor:
    return tf.SparseTensor(coo.coords.T, coo.data, coo.shape)
