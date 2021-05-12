"""Tests for TileDB integration with Tensorflow Data API."""

import os
import tiledb
import numpy as np
import uuid
from scipy.sparse import random
from scipy import stats

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Dropout,
)
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test

from tiledb.ml.data_apis.tensorflow_sparse import TensorflowTileDBSparseDataset
# from tiledb.ml.utils import pprint_sparse_tensor


# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_eager_execution()

# Test parameters
NUM_OF_CLASSES = 1
BATCH_SIZE = 32
ROWS = 1000

# We test for 2d, 3d, 4d and 5d data
INPUT_SHAPES = [(10,),]


class CustomRandomState(np.random.RandomState):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2

def ingest_in_tiledb(data: np.array, batch_size: int, uri: str):
    dims = [
        tiledb.Dim(
            name="dim_" + str(dim),
            domain=(0, data.shape[dim] - 1),
            tile=data.shape[dim] if dim > 0 else batch_size,
            dtype=np.int32,
        )
        for dim in range(data.ndim)
    ]

    # TileDB schema
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        sparse=False,
        attrs=[tiledb.Attr(name="features", dtype=np.float32)],
    )
    # Create array
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        tiledb_array[:] = data

def ingest_in_tiledb_sparse(data: np.array, batch_size: int, uri: str):
    dims = [
        tiledb.Dim(
            name="dim_" + str(dim),
            domain=(0, data.shape[dim] - 1),
            tile=data.shape[dim] if dim > 0 else batch_size,
            dtype=np.int32,
        )
        for dim in range(data.ndim)
    ]

    # The array will be sparse with a single attribute "a" so each (i,j) cell can store an integer.
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(*dims),
        sparse=True,
        attrs=[tiledb.Attr(name="features", dtype=np.float32)],
    )

    # Create the (empty) array on disk.
    tiledb.SparseArray.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        I,J = data.row, data.col
        data_elem = np.array(data.data)
        tiledb_array[I, J] = data_elem

# def pprint_sparse_tensor(st):
#   x = "<SparseTensor shape=%s \n values={" % (st[0].dense_shape.numpy().tolist(),)
#   for (index, value) in zip(st[0].indices, st[0].values):
#     x += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist()) + "}>"
#   x = x + "}>"
#   y = "<SparseTensor shape=%s \n values={" % (st[1].dense_shape.numpy().tolist(),)
#   for (index, value) in zip(st[1].indices, st[1].values):
#       y += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist()) + "}>"
#   y = y + "}>"
#   return print(x + "|--------------------|" + y)

def create_model(input_shape: tuple, num_of_classes: int):

    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    # model.add(Dense(num_of_classes))
    # # model.add(Dropout(0.5))
    # # model.add(Dense(num_of_classes))
    #
    # # model = Sequential()
    # # model.add(tf.keras.Input(shape=input_shape, sparse=True))
    # # # model.add(Dense(1))
    model.compile(
        # loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["sparse_binary_accuracy"],
    )

    # x = tf.keras.Input(shape=input_shape, name='x')
    # y_pred = tf.keras.layers.Dense(1, name='y_pred')(x)
    # model = tf.keras.Model(inputs=x, outputs=y_pred)
    # loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    # model.compile('Adam', loss=loss)

    # x = tf.keras.Input(shape=(10,), sparse=True)
    # y = tf.keras.layers.Dense(10)(x)
    # model = tf.keras.Model(x, y)

    return model


class TestTileDBTensorflowSparseDataAPI(test.TestCase):
    @testing_utils.run_v2_only
    def test_sparse_tiledb_tf_data_api_with_multiple_dim_data(self):
        for input_shape in INPUT_SHAPES:
            with self.subTest():
                model = create_model(
                    input_shape=input_shape, num_of_classes=NUM_OF_CLASSES
                )

                array_uuid = str(uuid.uuid4())
                tiledb_uri_x = os.path.join(self.get_temp_dir(), "x" + array_uuid)
                tiledb_uri_y = os.path.join(self.get_temp_dir(), "y" + array_uuid)

                dataset_shape_x = (ROWS,) + input_shape
                dataset_shape_y = (ROWS, NUM_OF_CLASSES)

                rs = CustomRandomState()
                rvs = stats.poisson(25, loc=10).rvs

                ingest_in_tiledb_sparse(
                    uri=tiledb_uri_x,
                    data=random(*dataset_shape_x, density=0.25, random_state=rs, data_rvs=rvs),
                    batch_size=BATCH_SIZE,
                )
                ingest_in_tiledb_sparse(
                    uri=tiledb_uri_y,
                    data=random(*dataset_shape_y, density=0.25, random_state=rs, data_rvs=rvs),
                    batch_size=BATCH_SIZE,
                )

                with tiledb.SparseArray(tiledb_uri_x, mode="r") as x, tiledb.SparseArray(
                        tiledb_uri_y, mode="r"
                ) as y:
                    tiledb_dataset = TensorflowTileDBSparseDataset(
                        x_array=x, y_array=y, batch_size=BATCH_SIZE
                    )

                    self.assertIsInstance(tiledb_dataset, tf.data.Dataset)

                    # for item in tiledb_dataset:
                    #     pprint_sparse_tensor(item)
                    # model.fit(tiledb_dataset, epochs=2)

                    model.fit(tiledb_dataset, verbose=0, epochs=1)

    @testing_utils.run_v2_only
    def test_sparse_except_with_diff_number_of_x_y_rows(self):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(self.get_temp_dir(), "x" + array_uuid)
        tiledb_uri_y = os.path.join(self.get_temp_dir(), "y" + array_uuid)

        # Add one extra row on X
        dataset_shape_x = (ROWS + 1,) + INPUT_SHAPES[0]
        dataset_shape_y = (ROWS, NUM_OF_CLASSES)

        rs = CustomRandomState()
        rvs = stats.poisson(25, loc=10).rvs

        ingest_in_tiledb_sparse(
            uri=tiledb_uri_x,
            data=random(*dataset_shape_x, density=0.25, random_state=rs, data_rvs=rvs),
            batch_size=BATCH_SIZE,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(*dataset_shape_y),
            batch_size=BATCH_SIZE,
        )

        with tiledb.SparseArray(tiledb_uri_x, mode="r") as x, tiledb.DenseArray(
                tiledb_uri_y, mode="r"
        ) as y:
            with self.assertRaises(Exception):
                TensorflowTileDBSparseDataset(
                    x_array=x, y_array=y, batch_size=BATCH_SIZE
                )
