"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from math import ceil
from typing import Optional, Sequence, Tuple, Union

import sparse
import tensorflow as tf

import tiledb

from ._tensor_gen import TileDBNumpyGenerator, TileDBSparseGenerator
from ._tensor_schema import TensorSchema, iter_slices


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
    x_key_dim: Union[int, str] = 0,
    y_key_dim: Union[int, str] = 0,
    num_workers: int = 0,
) -> tf.data.Dataset:
    """Return a tf.data.Dataset for loading data from TileDB arrays.

    :param x_array: TileDB array of the features.
    :param y_array: TileDB array of the labels.
    :param batch_size: Size of each batch.
    :param buffer_bytes: Maximum size (in bytes) of memory to allocate for reading from
        each array. This is bounded by the `sm.memory_budget` config parameter of the
        array context for dense arrays and `py.init_buffer_bytes` (or 10 MB if unset) for
        sparse arrays. These bounds are also used as the default memory budget.
    :param shuffle_buffer_size: Number of elements from which this dataset will sample.
    :param prefetch: Maximum number of batches that will be buffered when prefetching.
        By default, the buffer size is dynamically tuned.
    :param x_attrs: Attribute and/or dimension names of the x_array to read. Defaults to
        all attributes.
    :param y_attrs: Attribute and/or dimension names of the y_array to read. Defaults to
        all attributes.
    :param x_key_dim: Name or index of the key dimension of x_array.
    :param y_key_dim: Name or index of the key dimension of y_array.
    :param num_workers: If greater than zero, create a threadpool of `num_workers` threads
        used to fetch inputs asynchronously and in parallel. Note: yielded batches may
        be shuffled even if `shuffle_buffer_size` is zero when `num_workers` > 1.
    """
    x_schema = TensorSchema(x_array, x_key_dim, x_attrs)
    y_schema = TensorSchema(y_array, y_key_dim, y_attrs)
    x_schema.ensure_equal_keys(y_schema)

    x_gen = _get_tensor_generator(x_array, x_schema)
    y_gen = _get_tensor_generator(y_array, y_schema)

    def bounded_dataset(bounds: Union[Tuple[int, int], tf.Tensor]) -> tf.data.Dataset:
        x_dataset = tf.data.Dataset.from_generator(
            lambda start, stop: x_gen(
                x_schema.partition_key_dim(buffer_bytes, start, stop)
            ),
            args=(bounds[0], bounds[1]),
            output_signature=_get_tensor_specs(x_array, x_schema),
        )
        y_dataset = tf.data.Dataset.from_generator(
            lambda start, stop: y_gen(
                y_schema.partition_key_dim(buffer_bytes, start, stop)
            ),
            args=(bounds[0], bounds[1]),
            output_signature=_get_tensor_specs(y_array, y_schema),
        )
        return tf.data.Dataset.zip((x_dataset.unbatch(), y_dataset.unbatch()))

    if num_workers:
        per_worker = ceil(x_schema.num_keys / num_workers)
        offsets = [
            (s.start, s.stop)
            for s in iter_slices(x_schema.start_key, x_schema.stop_key, per_worker)
        ]
        offsets_tensor = tf.convert_to_tensor(offsets, dtype=tf.int64)
        offsets_dataset = tf.data.Dataset.from_tensor_slices(offsets_tensor)
        dataset = offsets_dataset.interleave(
            bounded_dataset, num_parallel_calls=num_workers, deterministic=False
        )
    else:
        dataset = bounded_dataset((x_schema.start_key, x_schema.stop_key))

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    return dataset.batch(batch_size).prefetch(prefetch)


TensorSpec = Union[tf.TensorSpec, tf.SparseTensorSpec]


def _get_tensor_specs(
    array: tiledb.Array, schema: TensorSchema
) -> Union[TensorSpec, Sequence[TensorSpec]]:
    cls = tf.SparseTensorSpec if array.schema.sparse else tf.TensorSpec
    shape = (None, *schema.shape[1:])
    specs = tuple(cls(shape=shape, dtype=dtype) for dtype in schema.field_dtypes)
    return specs if len(specs) > 1 else specs[0]


def _get_tensor_generator(
    array: tiledb.Array, schema: TensorSchema
) -> Union[TileDBNumpyGenerator, TileDBSparseGenerator[tf.SparseTensor]]:
    if not array.schema.sparse:
        return TileDBNumpyGenerator(array, schema)
    else:
        return TileDBSparseGenerator(array, schema, from_coo=_coo_to_sparse_tensor)


def _coo_to_sparse_tensor(coo: sparse.COO) -> tf.SparseTensor:
    return tf.SparseTensor(coo.coords.T, coo.data, coo.shape)
