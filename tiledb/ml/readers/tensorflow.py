"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from functools import partial
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import tensorflow as tf

import tiledb

from ._batch_utils import (
    BaseDenseBatch,
    BaseSparseBatch,
    get_attr_names,
    get_buffer_size,
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
    x_attrs: Sequence[str] = (),
    y_attrs: Sequence[str] = (),
) -> tf.data.Dataset:
    """Return a tf.data.Dataset for loading data from TileDB arrays.

    :param x_array: TileDB array of the features.
    :param y_array: TileDB array of the labels.
    :param batch_size: Size of each batch.
    :param buffer_size: Size of the buffer used to read the data. If not given,
        it is determined automatically.
    :param x_attrs: Attribute names of x_array.
    :param y_attrs: Attribute names of y_array.
    :param batch_shuffle: True for shuffling batches.
    :param within_batch_shuffle: True for shuffling records in each batch.
    """
    # Check that x_array and y_array have the same number of rows
    rows: int = x_array.shape[0]
    if rows != y_array.shape[0]:
        raise ValueError("X and Y arrays must have the same number of rows")

    return tf.data.Dataset.from_generator(
        partial(
            tensor_generator,
            dense_batch_cls=TensorflowDenseBatch,
            sparse_batch_cls=TensorflowSparseBatch,
            x_array=x_array,
            y_array=y_array,
            x_attrs=x_attrs,
            y_attrs=y_attrs,
            batch_size=batch_size,
            buffer_size=get_buffer_size(buffer_size, batch_size),
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ),
        output_signature=(
            *_iter_tensor_specs(x_array.schema, x_attrs),
            *_iter_tensor_specs(y_array.schema, y_attrs),
        ),
    )


def _iter_tensor_specs(
    schema: tiledb.ArraySchema, attrs: Sequence[str]
) -> Iterator[Union[tf.TensorSpec, tf.SparseTensorSpec]]:
    cls = tf.SparseTensorSpec if schema.sparse else tf.TensorSpec
    for attr in attrs or get_attr_names(schema):
        yield cls(shape=(None, *schema.shape[1:]), dtype=schema.attr(attr).dtype)


class TensorflowDenseBatch(BaseDenseBatch[tf.Tensor]):
    @staticmethod
    def _tensor_from_numpy(data: np.ndarray) -> tf.Tensor:
        return tf.convert_to_tensor(data)


class TensorflowSparseBatch(BaseSparseBatch[tf.SparseTensor]):
    @staticmethod
    def _tensor_from_coo(
        data: np.ndarray,
        coords: np.ndarray,
        dense_shape: Sequence[int],
        dtype: np.dtype,
    ) -> tf.SparseTensor:
        return tf.SparseTensor(
            indices=tf.constant(coords, dtype=tf.int64),
            values=tf.constant(data, dtype=dtype),
            dense_shape=dense_shape,
        )
