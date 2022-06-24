"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from typing import Sequence, Union

import sparse
import tensorflow as tf

import tiledb

from ._tensor_schema import DenseTensorSchema, SparseTensorSchema, TensorSchema


def TensorflowTileDBDataset(
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    *,
    batch_size: int,
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
        used to fetch inputs asynchronously and in parallel. Note: when `num_workers` > 1
        yielded batches may be shuffled even if `shuffle_buffer_size` is zero.
    """
    x_schema = _get_tensor_schema(x_array, x_key_dim, x_attrs)
    y_schema = _get_tensor_schema(y_array, y_key_dim, y_attrs)
    if not x_schema.key_range.equal_values(y_schema.key_range):
        raise ValueError(
            f"X and Y arrays have different key range: {x_schema.key_range} != {y_schema.key_range}"
        )

    x_max_weight = x_schema.max_partition_weight
    y_max_weight = y_schema.max_partition_weight
    key_ranges = list(x_schema.key_range.partition_by_count(num_workers or 1))

    def key_range_dataset(key_range_idx: int) -> tf.data.Dataset:
        x_dataset = tf.data.Dataset.from_generator(
            lambda i: x_schema.iter_tensors(
                key_ranges[i].partition_by_weight(x_max_weight)
            ),
            args=(key_range_idx,),
            output_signature=_get_tensor_specs(x_schema),
        )
        y_dataset = tf.data.Dataset.from_generator(
            lambda i: y_schema.iter_tensors(
                key_ranges[i].partition_by_weight(y_max_weight)
            ),
            args=(key_range_idx,),
            output_signature=_get_tensor_specs(y_schema),
        )
        return tf.data.Dataset.zip((x_dataset.unbatch(), y_dataset.unbatch()))

    if num_workers:
        dataset = tf.data.Dataset.from_tensor_slices(range(len(key_ranges))).interleave(
            key_range_dataset, num_parallel_calls=num_workers, deterministic=False
        )
    else:
        dataset = key_range_dataset(0)

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    return dataset.batch(batch_size).prefetch(prefetch)


TensorSpec = Union[tf.TensorSpec, tf.SparseTensorSpec]


def _get_tensor_specs(schema: TensorSchema) -> Union[TensorSpec, Sequence[TensorSpec]]:
    cls = tf.SparseTensorSpec if schema.sparse else tf.TensorSpec
    shape = (None, *schema.shape[1:])
    specs = tuple(cls(shape=shape, dtype=dtype) for dtype in schema.field_dtypes)
    return specs if len(specs) > 1 else specs[0]


def _get_tensor_schema(
    array: tiledb.Array,
    key_dim: Union[int, str],
    fields: Sequence[str],
) -> TensorSchema:
    if not array.schema.sparse:
        return DenseTensorSchema(array, key_dim, fields)
    else:
        return SparseTensorSchema(array, key_dim, fields, _coo_to_sparse_tensor)


def _coo_to_sparse_tensor(coo: sparse.COO) -> tf.SparseTensor:
    return tf.SparseTensor(coo.coords.T, coo.data, coo.shape)
