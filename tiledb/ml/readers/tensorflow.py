"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from typing import Sequence, Union

import sparse
import tensorflow as tf

from ._tensor_schema import DenseTensorSchema, SparseTensorSchema, TensorSchema
from .types import ArrayParams


def TensorflowTileDBDataset(
    *array_params: ArrayParams,
    batch_size: int,
    shuffle_buffer_size: int = 0,
    prefetch: int = tf.data.AUTOTUNE,
    num_workers: int = 0,
) -> tf.data.Dataset:
    """Return a tf.data.Dataset for loading data from TileDB arrays.

    :param array_params: One or more `ArrayParams` instances, one per TileDB array.
    :param batch_size: Size of each batch.
    :param shuffle_buffer_size: Number of elements from which this dataset will sample.
    :param prefetch: Maximum number of batches that will be buffered when prefetching.
        By default, the buffer size is dynamically tuned.
    :param num_workers: If greater than zero, create a threadpool of `num_workers` threads
        used to fetch inputs asynchronously and in parallel. Note: when `num_workers` > 1
        yielded batches may be shuffled even if `shuffle_buffer_size` is zero.
    """
    schemas = tuple(map(_get_tensor_schema, array_params))
    key_range = schemas[0].key_range
    if not all(key_range.equal_values(schema.key_range) for schema in schemas[1:]):
        raise ValueError(f"All arrays must have the same key range: {key_range}")

    max_weights = tuple(schema.max_partition_weight for schema in schemas)
    key_subranges = tuple(key_range.partition_by_count(num_workers or 1))

    def key_range_dataset(key_range_idx: int) -> tf.data.Dataset:
        datasets = tuple(
            tf.data.Dataset.from_generator(
                lambda i, schema=schema, max_weight=max_weight: schema.iter_tensors(
                    key_subranges[i].partition_by_weight(max_weight)
                ),
                args=(key_range_idx,),
                output_signature=_get_tensor_specs(schema),
            ).unbatch()
            for schema, max_weight in zip(schemas, max_weights)
        )
        return tf.data.Dataset.zip(datasets) if len(datasets) > 1 else datasets[0]

    if num_workers:
        dataset = tf.data.Dataset.from_tensor_slices(range(len(key_subranges)))
        dataset = dataset.interleave(
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


def _get_tensor_schema(array_params: ArrayParams) -> TensorSchema:
    if not array_params.array.schema.sparse:
        return DenseTensorSchema.from_array_params(array_params)
    else:
        return SparseTensorSchema.from_array_params(array_params, _coo_to_sparse_tensor)


def _coo_to_sparse_tensor(coo: sparse.COO) -> tf.SparseTensor:
    return tf.SparseTensor(coo.coords.T, coo.data, coo.shape)
