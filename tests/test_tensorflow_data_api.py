"""Tests for TileDB integration with Tensorflow Data API."""

import os
import tiledb
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Dropout,
)
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test

from tiledb.ml.data_apis.tensorflow import TensorflowTileDBDataset

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def ingest_in_tiledb(data: np.array, batch_size: int, uri: str):
    # First dimension
    dims = [
        tiledb.Dim(
            name="dim_1", domain=(0, data.shape[0] - 1), tile=batch_size, dtype=np.int32
        )
    ]

    # Remaining dimensions
    for dim in range(1, data.ndim):
        dims.append(
            tiledb.Dim(
                name="dim" + str(dim),
                domain=(0, data.shape[dim] - 1),
                tile=data.shape[dim],
                dtype=np.int32,
            )
        )

    # TileDB schema
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(dims),
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
    def test_tiledb_tf_data_api_with_2d_data(self):
        num_of_classes = 5
        input_shape = (64, 3)
        batch_size = 32

        model = create_model(input_shape=input_shape, num_of_classes=num_of_classes)

        tiledb_uri_x = os.path.join(self.get_temp_dir(), "x")
        tiledb_uri_y = os.path.join(self.get_temp_dir(), "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x, data=np.random.rand(1000, 64, 3), batch_size=batch_size
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y, data=np.random.rand(1000, 5), batch_size=batch_size
        )

        tiledb_dataset = TensorflowTileDBDataset(
            x_uri=tiledb_uri_x, y_uri=tiledb_uri_y, batch_size=batch_size
        )

        # Assert that dataset is instance of Tensorflow dataset.
        self.assertIsInstance(tiledb_dataset, tf.data.Dataset)

        model.fit(tiledb_dataset, verbose=0, epochs=1)

    @testing_utils.run_v2_only
    def test_tiledb_tf_data_api_with_3d_data(self):
        num_of_classes = 5
        input_shape = (64, 64, 3)
        batch_size = 32

        model = create_model(input_shape=input_shape, num_of_classes=num_of_classes)

        tiledb_uri_x = os.path.join(self.get_temp_dir(), "x")
        tiledb_uri_y = os.path.join(self.get_temp_dir(), "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(1000, 64, 64, 3),
            batch_size=batch_size,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(1000, num_of_classes),
            batch_size=batch_size,
        )

        tiledb_dataset = TensorflowTileDBDataset(
            x_uri=tiledb_uri_x, y_uri=tiledb_uri_y, batch_size=batch_size
        )

        # Assert that dataset is instance of Tensorflow dataset.
        self.assertIsInstance(tiledb_dataset, tf.data.Dataset)

        model.fit(tiledb_dataset, verbose=0, epochs=1)

    @testing_utils.run_v2_only
    def test_tiledb_tf_data_api_with_4d_data(self):
        num_of_classes = 5
        input_shape = (10, 5, 5, 2)
        batch_size = 32

        model = create_model(input_shape=input_shape, num_of_classes=num_of_classes)

        tiledb_uri_x = os.path.join(self.get_temp_dir(), "x")
        tiledb_uri_y = os.path.join(self.get_temp_dir(), "y")

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(1000, 10, 5, 5, 2),
            batch_size=batch_size,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(1000, num_of_classes),
            batch_size=batch_size,
        )

        tiledb_dataset = TensorflowTileDBDataset(
            x_uri=tiledb_uri_x, y_uri=tiledb_uri_y, batch_size=batch_size
        )

        # Assert that dataset is instance of Tensorflow dataset.
        self.assertIsInstance(tiledb_dataset, tf.data.Dataset)

        model.fit(tiledb_dataset, verbose=0, epochs=1)
