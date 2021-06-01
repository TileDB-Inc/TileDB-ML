"""Tests for TileDB integration with Tensorflow Data API."""

import os
import tiledb
import numpy as np
import uuid

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Dropout,
)
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test

from tiledb.ml.data_apis.tensorflow import TensorflowTileDBDenseDataset

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 32
ROWS = 1000

# We test for 2d, 3d, 4d and 5d data
INPUT_SHAPES = [(10,), (10, 3), (10, 10, 3), (10, 10, 10, 3)]


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
        tiledb_array[:] = {"features": data}


def create_model(input_shape: tuple, num_of_classes: int):
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes))

    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )

    return model


class TestTileDBTensorflowDataAPI(test.TestCase):
    @testing_utils.run_v2_only
    def test_tiledb_tf_data_api_with_multiple_dim_data(self):
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

                ingest_in_tiledb(
                    uri=tiledb_uri_x,
                    data=np.random.rand(*dataset_shape_x),
                    batch_size=BATCH_SIZE,
                )
                ingest_in_tiledb(
                    uri=tiledb_uri_y,
                    data=np.random.rand(*dataset_shape_y),
                    batch_size=BATCH_SIZE,
                )

                with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:

                    tiledb_dataset = TensorflowTileDBDenseDataset(
                        x_array=x, y_array=y, batch_size=BATCH_SIZE
                    )

                    self.assertIsInstance(tiledb_dataset, tf.data.Dataset)

                    model.fit(tiledb_dataset, verbose=0, epochs=1)

    @testing_utils.run_v2_only
    def test_except_with_diff_number_of_x_y_rows(self):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(self.get_temp_dir(), "x" + array_uuid)
        tiledb_uri_y = os.path.join(self.get_temp_dir(), "y" + array_uuid)

        # Add one extra row on X
        dataset_shape_x = (ROWS + 1,) + INPUT_SHAPES[0]
        dataset_shape_y = (ROWS, NUM_OF_CLASSES)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(*dataset_shape_x),
            batch_size=BATCH_SIZE,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(*dataset_shape_y),
            batch_size=BATCH_SIZE,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with self.assertRaises(Exception):
                TensorflowTileDBDenseDataset(
                    x_array=x, y_array=y, batch_size=BATCH_SIZE
                )

    @testing_utils.run_v2_only
    def test_dataset_length(self):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(self.get_temp_dir(), "x" + array_uuid)
        tiledb_uri_y = os.path.join(self.get_temp_dir(), "y" + array_uuid)

        # Add one extra row on X
        dataset_shape_x = (ROWS,) + INPUT_SHAPES[0]
        dataset_shape_y = (ROWS, NUM_OF_CLASSES)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(*dataset_shape_x),
            batch_size=BATCH_SIZE,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(*dataset_shape_y),
            batch_size=BATCH_SIZE,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBDenseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )

            self.assertEqual(len(tiledb_dataset), ROWS)
