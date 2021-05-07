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

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 32
ROWS = 1000

# We test for 2d, 3d, 4d and 5d data
INPUT_SHAPES = [(10,)]


class CustomRandomState(np.random.RandomState):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2


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
        # print(data)
        I,J = data.row, data.col
        data_elem = np.array(data.data)
        tiledb_array[I, J] = data_elem


def create_model(input_shape: tuple, num_of_classes: int):
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape, sparse=True))
    #model.add(Flatten())
    model.add(Dense(64))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_of_classes))
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["binary_accuracy"],
    )

    return model


class TestTileDBTensorflowSparseDataAPI(test.TestCase):
    @testing_utils.run_v2_only
    def test_sparse_tiledb_tf_data_api_with_multiple_dim_data(self):
        for input_shape in INPUT_SHAPES:
            with self.subTest():
                # model = create_model(
                #     input_shape=input_shape, num_of_classes=NUM_OF_CLASSES
                # )

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

                    # model(tiledb_dataset[0])
                    # model.predict(tiledb_dataset[0])

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
        ingest_in_tiledb_sparse(
            uri=tiledb_uri_y,
            data=random(*dataset_shape_y, density=0.25, random_state=rs, data_rvs=rvs),
            batch_size=BATCH_SIZE,
        )

        with tiledb.SparseArray(tiledb_uri_x, mode="r") as x, tiledb.SparseArray(
                tiledb_uri_y, mode="r"
        ) as y:
            with self.assertRaises(Exception):
                TensorflowTileDBSparseDataset(
                    x_array=x, y_array=y, batch_size=BATCH_SIZE
                )
