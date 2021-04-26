"""Functionality for loading data directly from TileDB arrays into the Tensorflow Data API."""
import tiledb
import tensorflow as tf


class TensorflowTileDBDataset(tf.data.Dataset):
    """
    Class that implements all functionality needed to load data from TileDB arrays to the
    Tensorflow Data API, by employing .
    """

    def __new__(cls, x_uri: str, y_uri: str, batch_size: int):
        """
        Returns a Tensorflow Dataset object which loads data by employing a generator.
        :param x_uri: String. URI for a TileDB array that contains features.
        :param y_uri: String. URI for a TileDB array that contains labels.
        :param batch_size: Integer. The size of the batch that the implemented _generator method will return.
        For optimal reads from a TileDB array, it is recommended to set the batch size equal to the tile extent of the
        dimension we query (here, we always query the first dimension of a TileDB array) in order to get a slice (batch)
        of the data. For example, in case the tile extent of the first dimension of a TileDB array (x or y) is equal to
        32, it's recommended to set batch_size=32. Any batch size will work, but in case it's not equal the tile extent
        of the first dimension of the TileDB array, you won't achieve highest read speed. For more details on tiles,
        tile extent and indices in TileDB, please check here:
        https://docs.tiledb.com/main/solutions/tiledb-embedded/performance-tips/choosing-tiling-and-cell-layout#dense-arrays
        """
        # Open TileDB train arrays
        x = tiledb.open(x_uri, mode="r")
        y = tiledb.open(y_uri, mode="r")

        # Get number of observations
        rows = x.schema.domain.dim(0).domain[1] - y.schema.domain.dim(0).domain[0] + 1

        x_dtype = x.schema.attr(0).dtype
        y_dtype = y.schema.attr(0).dtype

        return tf.data.Dataset.from_generator(
            generator=cls._generator,
            output_types=(x_dtype, y_dtype),
            args=(x, y, rows, batch_size),
        )

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
