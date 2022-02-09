"""Functionality for loading data from TileDB arrays to the Tensorflow Data API."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

import tiledb

from ._batch_utils import BaseDenseBatch, BaseSparseBatch

# TODO: We have to track the following issues:
# - https://github.com/tensorflow/tensorflow/issues/47532
# - https://github.com/tensorflow/tensorflow/issues/47931
# - https://github.com/tensorflow/tensorflow/issues/44565


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
    if isinstance(x_array, tiledb.DenseArray):
        if isinstance(y_array, tiledb.SparseArray):
            raise TypeError("Dense x_array and sparse y_array not currently supported")

    # Check that x_array and y_array have the same number of rows
    rows: int = x_array.schema.domain.shape[0]
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
    return tf.data.Dataset.from_generator(
        partial(
            _generator,
            x_array=x_array,
            y_array=y_array,
            x_attrs=x_attribute_names,
            y_attrs=y_attribute_names,
            offsets=range(0, rows, batch_size),
            batch_size=batch_size,
            buffer_size=buffer_size,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        ),
        output_signature=output_signature,
    )


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


def _generator(
    x_array: tiledb.Array,
    y_array: tiledb.Array,
    x_attrs: Sequence[str],
    y_attrs: Sequence[str],
    offsets: range,
    batch_size: int,
    buffer_size: int,
    batch_shuffle: bool = False,
    within_batch_shuffle: bool = False,
) -> Iterator[Tuple[Union[tf.Tensor, tf.SparseTensor], ...]]:
    x_batch = TensorflowBatch(x_array.schema, x_attrs, batch_size)
    y_batch = TensorflowBatch(y_array.schema, y_attrs, batch_size)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for offset in offsets:
            x_buffer, y_buffer = executor.map(
                lambda array: array[offset : offset + buffer_size],  # type: ignore
                (x_array, y_array),
            )
            x_batch.set_buffer_offset(x_buffer, offset)
            y_batch.set_buffer_offset(y_buffer, offset)

            # Split the buffer_size into batch_size chunks
            batch_offsets = np.arange(0, buffer_size, batch_size)
            if batch_shuffle:
                np.random.shuffle(batch_offsets)

            for batch_offset in batch_offsets:
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                x_batch.set_batch_slice(batch_slice)
                y_batch.set_batch_slice(batch_slice)
                if len(x_batch) != len(y_batch):
                    raise ValueError(
                        "x_array and y_array should have the same number of rows, "
                        "i.e. the first dimension of x_array and y_array should be of "
                        "equal domain extent inside the batch"
                    )
                if x_batch:
                    if within_batch_shuffle:
                        idx = np.arange(len(x_batch))
                        np.random.shuffle(idx)
                    else:
                        idx = Ellipsis

                    yield x_batch.get_tensors(idx) + y_batch.get_tensors(idx)


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


def TensorflowBatch(
    schema: tiledb.ArraySchema, attrs: Sequence[str], batch_size: int
) -> Union[TensorflowDenseBatch, TensorflowSparseBatch]:
    if schema.sparse:
        return TensorflowSparseBatch(attrs, schema, batch_size)
    else:
        return TensorflowDenseBatch(attrs)
