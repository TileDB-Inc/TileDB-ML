"""Functionality for loading data directly from TileDB arrays into the Tensorflow Data API."""
import tiledb
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import FlatMapDataset


class TensorflowTileDBDenseDataset(FlatMapDataset):
    """
    Class that implements all functionality needed to load data from TileDB directly to the
    Tensorflow Data API, by employing generators.
    """

    def __new__(cls, x_array: tiledb.Array, y_array: tiledb.Array, batch_size: int):
        """
        Returns a Tensorflow Dataset object which loads data from TileDB arrays by employing a generator.
        :param x_array: TileDB Dense Array. Array that contains features.
        :param y_array: TileDB Dense Array. Array that contains labels.
        :param batch_size: Integer. The size of the batch that the implemented _generator method will return.
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

        # Get number of observations
        rows = x_array.schema.domain.shape[0]

        # Get x and y shapes
        x_shape = (None,) + x_array.schema.domain.shape[1:]
        y_shape = (None,) + y_array.schema.domain.shape[1:]

        # Get x and y data types
        x_dtype = x_array.schema.attr(0).dtype
        y_dtype = y_array.schema.attr(0).dtype

        obj = tf.data.Dataset.from_generator(
            generator=cls._generator,
            output_signature=(
                tf.TensorSpec(shape=x_shape, dtype=x_dtype),
                tf.TensorSpec(shape=y_shape, dtype=y_dtype),
            ),
            args=(x_array, y_array, rows, batch_size),
        )

        # Class reassignment in order to be able to override __len__().
        obj.__class__ = cls

        return obj

    # We also have to define __init__, in order to be able to override __len__().  We also need the same
    # signature with __new__()
    def __init__(self, x_array: tiledb.Array, y_array: tiledb.Array, batch_size: int):
        self.length = x_array.schema.domain.shape[0]

    @staticmethod
    def _generator(
        x: tiledb.Array, y: tiledb.Array, rows: int, batch_size: int
    ) -> tuple:
        """
        A generator function that yields the next training batch.
        :param x: TileDB array. An opened TileDB array which contains features.
        :param y: TileDB array. An opened TileDB array which contains labels.
        :param rows: Integer. The number of observations in x, y datasets.
        :param batch_size: Integer. Size of batch, i.e., number of rows returned per call.
        :return: Tuple. Tuple that contains x and y batches.
        """
        # Loop over batches
        for offset in range(0, rows, batch_size):
            # Yield the next training batch
            yield x[offset : offset + batch_size], y[offset : offset + batch_size]

    def __len__(self):
        return self.length
