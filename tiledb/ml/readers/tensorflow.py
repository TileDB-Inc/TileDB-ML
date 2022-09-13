"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from functools import singledispatch
from typing import Sequence, Union

import numpy as np
import scipy.sparse
import sparse
import tensorflow as tf

from ._tensor_schema import MappedTensorSchema, RaggedArray, SparseArray, TensorSchema
from .types import ArrayParams, TensorKind

Tensor = Union[np.ndarray, tf.SparseTensor]


def TensorflowTileDBDataset(
    *all_array_params: ArrayParams,
    num_workers: int = 0,
) -> tf.data.Dataset:
    """Return a tf.data.Dataset for loading data from TileDB arrays.
    :param all_array_params: One or more `ArrayParams` instances, one per TileDB array.
    :param num_workers: If greater than zero, create a threadpool of `num_workers` threads
        used to fetch inputs asynchronously and in parallel.
    """
    schemas = []
    for array_params in all_array_params:
        schema = array_params.tensor_schema
        if schema.kind is TensorKind.SPARSE_CSR:
            raise NotImplementedError(f"{schema.kind} tensors not supported")
        elif schema.kind is TensorKind.SPARSE_COO:
            schema = MappedTensorSchema(schema, _to_sparse_tensor)
        elif schema.kind is TensorKind.RAGGED:
            schema = MappedTensorSchema(schema, _to_ragged_tensor)
        schemas.append(schema)

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

    return dataset


_tensor_specs = {
    TensorKind.DENSE: tf.TensorSpec,
    TensorKind.SPARSE_COO: tf.SparseTensorSpec,
    TensorKind.RAGGED: tf.RaggedTensorSpec,
}


def _get_tensor_specs(
    schema: TensorSchema[Tensor],
) -> Union[tf.TypeSpec, Sequence[tf.TypeSpec]]:
    spec_cls = _tensor_specs[schema.kind]
    shape = (None, *schema.shape[1:])
    specs = tuple(spec_cls(shape=shape, dtype=dtype) for dtype in schema.field_dtypes)
    return specs if len(specs) > 1 else specs[0]


@singledispatch
def _to_sparse_tensor(sa: SparseArray) -> tf.SparseTensor:
    """Create a tf.SparseTensor from a sparse array"""
    raise NotImplementedError


@_to_sparse_tensor.register(scipy.sparse.csr_matrix)
def _csr_to_sparse_tensor(csr: scipy.sparse.csr_matrix) -> tf.SparseTensor:
    """Create a tf.SparseTensor from a scipy.sparse.csr_matrix instance"""
    coo = csr.tocoo()
    coords = np.array((coo.row, coo.col))
    return tf.SparseTensor(coords.T, coo.data, coo.shape)


@_to_sparse_tensor.register(sparse.COO)
def _coo_to_sparse_tensor(coo: sparse.COO) -> tf.SparseTensor:
    """Create a tf.SparseTensor from a sparse.COO instance"""
    return tf.SparseTensor(coo.coords.T, coo.data, coo.shape)


def _to_ragged_tensor(ra: RaggedArray) -> tf.RaggedTensor:
    return tf.ragged.constant(ra, dtype=ra[0].dtype)
