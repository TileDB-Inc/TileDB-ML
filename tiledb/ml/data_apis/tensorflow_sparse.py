"""Functionality for loading data directly from TileDB arrays into the Tensorflow Data API."""
import dataclasses

import tiledb
import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix

tf.compat.v1.disable_eager_execution()

def pprint_sparse_tensor(st):
  s = "<SparseTensor shape=%s \n values={" % (st.dense_shape.numpy().tolist(),)
  for (index, value) in zip(st.indices, st.values):
    s += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist())
  return s + "}>"

class TensorflowTileDBSparseDataset(tf.data.Dataset):
    """
    Class that implements all functionality needed to load data from TileDB directly to the
    Tensorflow Data API, by employing generators.
    """

    def __new__(cls, x_array: tiledb.Array, y_array: tiledb.Array, batch_size: int):
        """
        Returns a Tensorflow Dataset object which loads data from TileDB arrays by employing a generator.
        :param x_array: TileDB Sparse Array. Array that contains features.
        :param y_array: TileDB Sparse Array. Array that contains labels.
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
            raise Exception(
                "X and Y should have the same number of rows, i.e., the 1st dimension "
                "of TileDB arrays X, Y should be of equal domain extent."
            )

        # Get number of observations
        rows = x_array.schema.domain.shape[0]

        # Get x and y shapes
        x_shape = (1000,) + x_array.schema.domain.shape[1:]
        y_shape = (1000,) + y_array.schema.domain.shape[1:]

        # Get x and y data types
        x_dtype = x_array.schema.attr(0).dtype
        y_dtype = y_array.schema.attr(0).dtype
        #
        # row = np.array([0, 3, 1, 0])
        # col = np.array([0, 3, 1, 2])
        # data = np.array([4, 5, 7, 9])
        # a = coo_matrix((data, (row, col)), shape=(4, 4))
        return tf.data.Dataset.from_generator(
            generator=cls._generator,
            output_signature=(
                tf.TensorSpec(shape=x_shape, dtype=x_dtype),
                tf.TensorSpec(shape=y_shape, dtype=y_dtype),
            ),
            args=(x_array, y_array, rows, batch_size),
        )

        # Loop over batches
        # st_x = []
        # st_y = []
        # for offset in range(0, rows, batch_size):
        #     values_x = x_array[offset: offset + batch_size]["features"],
        #     indices_x = np.array(
        #         list(zip(x_array[offset: offset + batch_size]["dim_0"], x_array[offset: offset + batch_size]["dim_1"])))
        #     values_y = y_array[offset: offset + batch_size]["features"],
        #     indices_y = np.array(
        #         list(zip(y_array[offset: offset + batch_size]["dim_0"], y_array[offset: offset + batch_size]["dim_1"])))
        #     st_x.append(tf.SparseTensor(indices=indices_x, values=values_x[0], dense_shape=x_shape))
        #     st_y.append(tf.SparseTensor(indices=indices_y, values=values_y[0], dense_shape=y_shape))
        #
        # st_concat_x = tf.sparse.concat(axis=0, sp_inputs=st_x)
        # st_concat_y = tf.sparse.concat(axis=0, sp_inputs=st_y)

        # print(pprint_sparse_tensor(st_concat))
        # print(type(st_concat))
        # return (st_concat_x, st_concat_y)
        # return tf.data.Dataset.from_generator(
        #     generator=cls._generator,
        #     output_types=tf.float32,
        #     output_shapes=x_shape,
        #     args=(x_array, y_array, rows, batch_size, x_shape, y_shape)
        # )

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
        #https://github.com/tensorflow/tensorflow/issues/44565
        for offset in range(0, rows, batch_size):
        #     # Yield the next training batch
        #     values_x = x[offset: offset + batch_size]["features"],
        #     indices_x = np.array(
        #         list(zip(x[offset: offset + batch_size]["dim_0"], x[offset: offset + batch_size]["dim_1"])))
            # values_y = y[offset: offset + batch_size]["features"],
            # indices_y = np.array(
            #     list(zip(y[offset: offset + batch_size]["dim_0"], y[offset: offset + batch_size]["dim_1"])))
        # values_a = x.data
        # indices_a = np.array(list(zip(x.row, x.col)))
            yield tf.SparseTensor(indices=indices_a, values=values_a, dense_shape=[4, 4])
            # yield tf.SparseTensor(indices=indices_y, values=values_y[0], dense_shape=y_dshape)
            # yield (indices_x, values_x[0]), (indices_y, values_y[0])