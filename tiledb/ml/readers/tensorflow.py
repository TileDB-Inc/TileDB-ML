"""Functionality for loading data directly from TileDB arrays into the Tensorflow Data API."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Iterator, Optional, Sequence, Tuple, cast

import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import FlatMapDataset

import tiledb
from tiledb.ml._parallel_utils import run_io_tasks_in_parallel


class TensorflowTileDBDenseDataset(FlatMapDataset):
    """
    Class that implements all functionality needed to load data from TileDB directly to the
    Tensorflow Data API, by employing generators.
    """

    def __new__(
        cls,
        x_array: tiledb.DenseArray,
        y_array: tiledb.DenseArray,
        batch_size: int,
        buffer_size: Optional[int] = None,
        batch_shuffle: bool = False,
        within_batch_shuffle: bool = False,
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
    ) -> TensorflowTileDBDenseDataset:
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
        :param batch_size: The size of the batch that the implemented _generator method will return.
        :param batch_shuffle: True if we want to shuffle batches.
        :param within_batch_shuffle: True if we want to shuffle records in each batch.
        :param x_attribute_names: The attribute names of x_array.
        :param y_attribute_names: The attribute names of y_array.
        """

        # Check that x and y have the same number of rows
        if x_array.schema.domain.shape[0] != y_array.schema.domain.shape[0]:
            raise ValueError(
                "X and Y should have the same number of rows, i.e., the 1st dimension "
                "of TileDB arrays X, Y should be of equal domain extent."
            )

        # If a user doesn't pass explicit attribute names to return per batch, we return all attributes.
        if not x_attribute_names:
            x_attribute_names = [
                x_array.schema.attr(idx).name for idx in range(x_array.schema.nattr)
            ]

        if not y_attribute_names:
            y_attribute_names = [
                y_array.schema.attr(idx).name for idx in range(y_array.schema.nattr)
            ]

        # Set the buffer_size appropriately and check its size
        buffer_size_checked = buffer_size or batch_size
        if buffer_size_checked < batch_size:
            raise ValueError("Buffer size should be geq to the batch size.")

        # Get number of observations
        rows = x_array.schema.domain.shape[0]

        # Get x and y shapes
        x_shape = (None,) + x_array.schema.domain.shape[1:]
        y_shape = (None,) + y_array.schema.domain.shape[1:]

        # Signatures for x and y
        x_signature = tuple(
            tf.TensorSpec(shape=x_shape, dtype=x_array.schema.attr(attr).dtype)
            for attr in x_attribute_names
        )
        y_signature = tuple(
            tf.TensorSpec(shape=y_shape, dtype=y_array.schema.attr(attr).dtype)
            for attr in y_attribute_names
        )

        generator_ = partial(
            cls._generator,
            x=x_array,
            y=y_array,
            x_attribute_names=x_attribute_names,
            y_attribute_names=y_attribute_names,
            rows=rows,
            batch_size=batch_size,
            buffer_size=buffer_size_checked,
            batch_shuffle=batch_shuffle,
            within_batch_shuffle=within_batch_shuffle,
        )

        obj = tf.data.Dataset.from_generator(
            generator=generator_,
            output_signature=x_signature + y_signature,
        )

        # Class reassignment in order to be able to override __len__().
        obj.__class__ = cls

        return cast(TensorflowTileDBDenseDataset, obj)

    # We also have to define __init__ to be able to override __len__().
    def __init__(
        self,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        batch_size: int,
        buffer_size: Optional[int] = None,
        batch_shuffle: bool = False,
        within_batch_shuffle: bool = False,
        x_attribute_names: Sequence[str] = (),
        y_attribute_names: Sequence[str] = (),
    ):
        self.length: int = x_array.schema.domain.shape[0]

    @staticmethod
    def _generator(
        x: tiledb.Array,
        y: tiledb.Array,
        x_attribute_names: Sequence[str],
        y_attribute_names: Sequence[str],
        rows: int,
        batch_size: int,
        buffer_size: Optional[int],
        batch_shuffle: bool,
        within_batch_shuffle: bool,
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

        offsets = np.arange(0, rows, buffer_size)

        # Loop over batches
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in offsets:
                x_buffer, y_buffer = run_io_tasks_in_parallel(
                    executor,
                    (x, y),
                    buffer_size,
                    offset,
                )

                # Split the buffer_size into batch_size chunks
                batch_offsets = np.arange(0, buffer_size, batch_size)

                # Shuffle offsets in case we need batch shuffling
                if batch_shuffle:
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

                    if within_batch_shuffle:
                        # We get batch length based on the first attribute, because last batch might be smaller than the
                        # batch size
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

    def __len__(self) -> int:
        return self.length
