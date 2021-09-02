"""Functionality for loading data directly from TileDB arrays into the Tensorflow Data API."""
import tiledb
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import FlatMapDataset
from typing import List, Optional
from functools import partial


class TensorflowTileDBDenseDataset(FlatMapDataset):
    """
    Class that implements all functionality needed to load data from TileDB directly to the
    Tensorflow Data API, by employing generators.
    """

    def __new__(
        cls,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        batch_size: int,
        x_attribute_names: Optional[List[str]] = [],
        y_attribute_names: Optional[List[str]] = [],
    ):
        """
        Returns a Tensorflow Dataset object which loads data from TileDB arrays by employing a generator.
        :param x_array: TileDB Dense Array. Array that contains features.
        :param y_array: TileDB Dense Array. Array that contains labels.
        :param batch_size: Integer. The size of the batch that the implemented _generator method will return.
        :param x_attribute_names: List of str. A list that contains the attribute names of TileDB array x.
        :param y_attribute_names: List of str. A list that contains the attribute names of TileDB array y.
        For optimal reads from a TileDB array, it is recommended to set the batch size equal to the tile extent of the
        dimension we query (here, we always query the first dimension of a TileDB array) in order to get a slice (batch)
        of the data. For example, in case the tile extent of the first dimension of a TileDB array (x or y) is equal to
        32, it's recommended to set batch_size=32. Any batch size will work, but in case it's not equal the tile extent
        of the first dimension of the TileDB array, you won't achieve highest read speed. For more details on tiles,
        tile extent and indices in TileDB, please check here:
        https://docs.tiledb.com/main/solutions/tiledb-embedded/performance-tips/choosing-tiling-and-cell-layout#dense-arrays
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
        )

        obj = tf.data.Dataset.from_generator(
            generator=generator_,
            output_signature=x_signature + y_signature,
        )

        # Class reassignment in order to be able to override __len__().
        obj.__class__ = cls

        return obj

    # We also have to define __init__, in order to be able to override __len__().  We also need the same
    # signature with __new__()
    def __init__(
        self,
        x_array: tiledb.Array,
        y_array: tiledb.Array,
        batch_size: int,
        x_attribute_names: Optional[List[str]] = [],
        y_attribute_names: Optional[List[str]] = [],
    ):
        self.length = x_array.schema.domain.shape[0]

    @staticmethod
    def _generator(
        x: tiledb.Array,
        y: tiledb.Array,
        x_attribute_names: List[str],
        y_attribute_names: List[str],
        rows: int,
        batch_size: int,
    ) -> tuple:
        """
        A generator function that yields the next training batch.
        :param x: TileDB array. An opened TileDB array which contains features.
        :param y: TileDB array. An opened TileDB array which contains labels.
        :param x_attribute_names: List of str. A list that contains the attribute names of TileDB array x.
        :param y_attribute_names: List of str. A list that contains the attribute names of TileDB array y.
        :param rows: Integer. The number of observations in x, y datasets.
        :param batch_size: Integer. Size of batch, i.e., number of rows returned per call.
        :return: Tuple. Tuple that contains x and y batches.
        """
        # Loop over batches
        for offset in range(0, rows, batch_size):
            x_batch = x[offset : offset + batch_size]
            y_batch = y[offset : offset + batch_size]

            # Yield the next training batch
            yield tuple(x_batch[attr] for attr in x_attribute_names) + tuple(
                y_batch[attr] for attr in y_attribute_names
            )

    def __len__(self):
        return self.length
