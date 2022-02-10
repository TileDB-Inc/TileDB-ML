"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from functools import partial
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

import tiledb

from ._batch_utils import (
    BaseDenseBatch,
    BaseSparseBatch,
    get_attr_names,
    tensor_generator,
)

# TODO: We have to track the following issues:
# - https://github.com/tensorflow/tensorflow/issues/47532
# - https://github.com/tensorflow/tensorflow/issues/47931
# - https://github.com/tensorflow/tensorflow/issues/44565


def TensorflowTileDBDataset(
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    batch_size: int,
    buffer_size: Optional[int] = None,
    batch_shuffle: bool = False,
    within_batch_shuffle: bool = False,
    x_attribute_names: Sequence[str] = (),
    y_attribute_names: Sequence[str] = (),
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
    if isinstance(x_array, tiledb.DenseArray):
        if isinstance(y_array, tiledb.SparseArray):
            raise TypeError("Dense x_array and sparse y_array not currently supported")

    # Check that x_array and y_array have the same number of rows
    rows: int = x_array.shape[0]
    if rows != y_array.shape[0]:
        raise ValueError(
            "x_array and y_array should have the same number of rows, i.e. the "
            "first dimension of x_array and y_array should be of equal domain extent"
        )

    output_signature = _get_signature(x_array.schema, x_attribute_names)
    output_signature += _get_signature(y_array.schema, y_attribute_names)
    return tf.data.Dataset.from_generator(
        partial(
            tensor_generator,
            dense_batch_cls=TensorflowDenseBatch,
            sparse_batch_cls=TensorflowSparseBatch,
            x_array=x_array,
            y_array=y_array,
            x_attrs=x_attribute_names,
            y_attrs=y_attribute_names,
            batch_size=batch_size,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ),
        output_signature=output_signature,
    )


def _get_signature(
    schema: tiledb.ArraySchema, attrs: Sequence[str]
) -> Tuple[Union[tf.TensorSpec, tf.SparseTensorSpec], ...]:
    cls = tf.SparseTensorSpec if schema.sparse else tf.TensorSpec
    return tuple(
        cls(shape=(None, *schema.shape[1:]), dtype=schema.attr(attr).dtype)
        for attr in attrs or get_attr_names(schema)
    )


class TensorflowDenseBatch(BaseDenseBatch[tf.Tensor]):
    @staticmethod
    def _tensor_from_numpy(data: np.ndarray) -> tf.Tensor:
        return tf.convert_to_tensor(data)


class TensorflowSparseBatch(BaseSparseBatch[tf.SparseTensor]):
    @staticmethod
    def _tensor_from_coo(
        data: np.ndarray,
        coords: np.ndarray,
        dense_shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> tf.SparseTensor:
        return tf.SparseTensor(
            indices=tf.constant(coords, dtype=tf.int64),
            values=tf.constant(data, dtype=dtype),
            dense_shape=dense_shape,
        )
