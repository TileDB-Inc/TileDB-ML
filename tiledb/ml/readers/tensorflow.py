"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np
import tensorflow as tf

from ._tensor_schema import RaggedArray, SparseData, TensorKind, TensorSchema
from .types import ArrayParams

Tensor = Union[np.ndarray, tf.SparseTensor]


def TensorflowTileDBDataset(
    *all_array_params: ArrayParams,
    num_workers: int = 0,
) -> tf.data.Dataset:
    """Return a tf.data.Dataset for loading data from TileDB arrays.
    :param all_array_params: One or more `ArrayParams` instances, one per TileDB array.
    :param num_workers: If greater than zero, create a threadpool of `num_workers` threads
        used to fetch inputs asynchronously and in parallel. Note: when `num_workers` > 1
        yielded batches may be shuffled even if `shuffle_buffer_size` is zero.
    """
    schemas = tuple(
        array_params.to_tensor_schema(_transforms) for array_params in all_array_params
    )
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


def _to_sparse_tensor(sd: SparseData) -> tf.SparseTensor:
    sa = sd.to_sparse_array()
    coords = getattr(sa, "coords", None)
    if coords is None:
        # sa is a scipy.sparse.csr_matrix
        coo = sa.tocoo()
        coords = np.array((coo.row, coo.col))
    return tf.SparseTensor(coords.T, sa.data, sa.shape)


def _to_ragged_tensor(ra: RaggedArray) -> tf.RaggedTensor:
    return tf.ragged.constant(ra, dtype=ra[0].dtype)


_transforms: Mapping[TensorKind, Union[Callable[[Any], Any], bool]] = {
    TensorKind.DENSE: True,
    TensorKind.SPARSE_COO: _to_sparse_tensor,
    TensorKind.SPARSE_CSR: False,
    TensorKind.RAGGED: _to_ragged_tensor,
}
