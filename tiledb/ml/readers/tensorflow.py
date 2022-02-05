"""Functionality for loading data from TileDB dense arrays to the Tensorflow Data API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import wrapt

import tiledb

from ._parallel_utils import parallel_slice


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


class TensorflowTileDBDataset(wrapt.ObjectProxy):
    """Load data from a TileDB array to the Tensorflow Data API."""

    def __init__(
        self,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        batch_size: int,
        buffer_size: Optional[int] = None,
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
        **kwargs: Any,
    ):
        """
        Return a Tensorflow Dataset object which loads data from TileDB arrays.

        :param x_array: Array that contains features.
        :param y_array: Array that contains labels.
        :param batch_size: Size of each batch.
        :param x_attribute_names: Attribute names of x_array.
        :param y_attribute_names: Attribute names of y_array.
        """
        rows: int = x_array.schema.domain.shape[0]

        # Check that x and y have the same number of rows
        if rows != y_array.schema.domain.shape[0]:
            raise ValueError(
                "X and Y should have the same number of rows, i.e., the 1st dimension "
                "of TileDB arrays X, Y should be of equal domain extent"
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

        dataset = tf.data.Dataset.from_generator(
            generator=partial(
                self._generator,
                x=x_array,
                y=y_array,
                x_attribute_names=x_attribute_names,
                y_attribute_names=y_attribute_names,
                rows=rows,
                batch_size=batch_size,
                buffer_size=buffer_size,
                **kwargs,
            ),
            output_signature=output_signature,
        )
        super().__init__(dataset)
        self._rows = rows

    def __len__(self) -> int:
        return self._rows

    @classmethod
    def _generator(
        cls,
        x: tiledb.Array,
        y: tiledb.Array,
        x_attribute_names: Sequence[str],
        y_attribute_names: Sequence[str],
        rows: int,
        batch_size: int,
        buffer_size: int,
        **kwargs: Any,
    ) -> Iterator[Tuple[Union[tf.SparseTensor, np.ndarray], ...]]:
        raise NotImplementedError("Abstract method")


class TensorflowTileDBDenseDataset(TensorflowTileDBDataset):
    """Load data from a dense TileDB array to the Tensorflow Data API."""

    def __init__(
        self,
        x_array: tiledb.DenseArray,
        y_array: tiledb.DenseArray,
        batch_size: int,
        buffer_size: Optional[int] = None,
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
        batch_shuffle: bool = False,
        within_batch_shuffle: bool = False,
    ):
        """
        Return a Tensorflow Dataset object which loads data from TileDB arrays
        by employing a generator.

        For optimal reads from a TileDB array, it is recommended to set the batch size
        equal to the tile extent of the dimension we query (here, we always query the
        first dimension of a TileDB array) in order to get a slice (batch) of the data.
        For example, in case the tile extent of the first dimension of a TileDB array
        (x or y) is equal to 32, it's recommended to set batch_size=32. Any batch size
        will work, but in case it's not equal the tile extent of the first dimension of
        the TileDB array, you won't achieve highest read speed. For more details on tiles,
        tile extent and indices in TileDB, please check here:
        https://docs.tiledb.com/main/how-to/performance/performance-tips/choosing-tiling-and-cell-layout#dense-arrays

        :param x_array: Array that contains features.
        :param y_array: Array that contains labels.
        :param batch_size: The size of the batch that the implemented _generator method
            will return.
        :param batch_shuffle: True if we want to shuffle batches.
        :param within_batch_shuffle: True if we want to shuffle records in each batch.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        """
        if isinstance(x_array, tiledb.SparseArray):
            raise TypeError(
                "TensorflowTileDBDenseDataset accepts tiledb.DenseArray instances only"
            )

        super().__init__(
            x_array=x_array,
            y_array=y_array,
            batch_size=batch_size,
            buffer_size=buffer_size,
            x_attribute_names=x_attribute_names,
            y_attribute_names=y_attribute_names,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        )

    @classmethod
    def _generator(
        cls,
        x: tiledb.Array,
        y: tiledb.Array,
        x_attribute_names: Sequence[str],
        y_attribute_names: Sequence[str],
        rows: int,
        batch_size: int,
        buffer_size: int,
        **kwargs: Any,
    ) -> Iterator[Tuple[np.ndarray, ...]]:
        """
        Generator for yielding training batches.

        :param x: An opened TileDB array which contains features.
        :param y: An opened TileDB array which contains labels.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        :param rows: The number of observations in x, y datasets.
        :param batch_size: Size of batch, i.e., number of rows returned per call.
        :param batch_shuffle: True if we want to shuffle batches.
        :param within_batch_shuffle: True if we want to shuffle records in each batch.
        :return: An iterator of x and y batches.
        """
        # Loop over batches
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in range(0, rows, buffer_size):
                x_buffer, y_buffer = parallel_slice(
                    executor,
                    (x, y),
                    buffer_size,
                    offset,
                )

                # Split the buffer_size into batch_size chunks
                batch_offsets = np.arange(0, buffer_size, batch_size)

                # Shuffle offsets in case we need batch shuffling
                if kwargs["batch_shuffle"]:
                    np.random.shuffle(batch_offsets)

                for batch_offset in batch_offsets:
                    x_batch = {
                        attr: data[batch_offset : batch_offset + batch_size]
                        for attr, data in x_buffer.items()
                    }
                    y_batch = {
                        attr: data[batch_offset : batch_offset + batch_size]
                        for attr, data in y_buffer.items()
                    }

                    if kwargs["within_batch_shuffle"]:
                        # We get batch length based on the first attribute
                        # because last batch might be smaller than the batch size
                        rand_permutation = np.arange(
                            x_batch[x_attribute_names[0]].shape[0]
                        )

                        np.random.shuffle(rand_permutation)

                        # Yield the next training batch
                        yield tuple(
                            x_batch[attr][rand_permutation]
                            for attr in x_attribute_names
                        ) + tuple(
                            y_batch[attr][rand_permutation]
                            for attr in y_attribute_names
                        )
                    else:
                        # Yield the next training batch
                        yield tuple(
                            x_batch[attr] for attr in x_attribute_names
                        ) + tuple(y_batch[attr] for attr in y_attribute_names)
