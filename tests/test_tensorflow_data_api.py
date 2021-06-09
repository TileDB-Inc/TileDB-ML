"""Tests for TileDB integration with Tensorflow Data API."""

import os
import tiledb
import numpy as np
import uuid
import pytest
import tempfile

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Dropout,
)

from tiledb.ml.data_apis.tensorflow import TensorflowTileDBDenseDataset
from tiledb.ml.utils.helpers import ingest_in_tiledb


# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 32
ROWS = 1000


@pytest.fixture
def temp_uri():
    """
    Returns a temporary directory instance
    """
    return tempfile.TemporaryDirectory()


@pytest.fixture(
    params=[
        {
            "input_shape": (10,),
            "hyperparams": ("categorical_crossentropy", "rmsprop", "accuracy"),
        },
        {
            "input_shape": (10, 3),
            "hyperparams": ("categorical_crossentropy", "rmsprop", "accuracy"),
        },
        {
            "input_shape": (10, 10, 3),
            "hyperparams": ("categorical_crossentropy", "rmsprop", "accuracy"),
        },
    ]
)
def create_model(request):
    model = Sequential()
    model.add(tf.keras.Input(shape=request.param["input_shape"]))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_OF_CLASSES))

    model.compile(
        loss=request.param["hyperparams"][0],
        optimizer=request.param["hyperparams"][1],
        metrics=[request.param["hyperparams"][2]],
    )

    return model


class TestTileDBTensorflowDataAPI:
    def test_tiledb_tf_data_api_with_multiple_dim_data(self, temp_uri, create_model):
        with temp_uri as temp_dir:
            model = create_model
            array_uuid = str(uuid.uuid4())
            tiledb_uri_x = os.path.join(temp_dir, "x" + array_uuid)
            tiledb_uri_y = os.path.join(temp_dir, "y" + array_uuid)

            dataset_shape_x = (ROWS,) + model.input_shape[1:]
            dataset_shape_y = (ROWS, NUM_OF_CLASSES)

            ingest_in_tiledb(
                uri=tiledb_uri_x,
                data=np.random.rand(*dataset_shape_x),
                batch_size=BATCH_SIZE,
                sparse=False,
            )
            ingest_in_tiledb(
                uri=tiledb_uri_y,
                data=np.random.rand(*dataset_shape_y),
                batch_size=BATCH_SIZE,
                sparse=False,
            )

            with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:

                tiledb_dataset = TensorflowTileDBDenseDataset(
                    x_array=x, y_array=y, batch_size=BATCH_SIZE
                )

                assert isinstance(tiledb_dataset, tf.data.Dataset)

                model.fit(tiledb_dataset, verbose=0, epochs=1)

    def test_except_with_diff_number_of_x_y_rows(self, temp_uri, create_model):
        with temp_uri as temp_dir:
            array_uuid = str(uuid.uuid4())
            tiledb_uri_x = os.path.join(temp_dir, "x" + array_uuid)
            tiledb_uri_y = os.path.join(temp_dir, "y" + array_uuid)

            # Add one extra row on X
            dataset_shape_x = (ROWS + 1,) + create_model.input_shape[1:]
            dataset_shape_y = (ROWS, NUM_OF_CLASSES)

            ingest_in_tiledb(
                uri=tiledb_uri_x,
                data=np.random.rand(*dataset_shape_x),
                batch_size=BATCH_SIZE,
                sparse=False,
            )
            ingest_in_tiledb(
                uri=tiledb_uri_y,
                data=np.random.rand(*dataset_shape_y),
                batch_size=BATCH_SIZE,
                sparse=False,
            )

            with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
                with pytest.raises(Exception):
                    TensorflowTileDBDenseDataset(
                        x_array=x, y_array=y, batch_size=BATCH_SIZE
                    )

    def test_dataset_length(self, temp_uri, create_model):
        with temp_uri as temp_dir:
            array_uuid = str(uuid.uuid4())
            tiledb_uri_x = os.path.join(temp_dir, "x" + array_uuid)
            tiledb_uri_y = os.path.join(temp_dir, "y" + array_uuid)

            # Add one extra row on X
            dataset_shape_x = (ROWS,) + create_model.input_shape[1:]
            dataset_shape_y = (ROWS, NUM_OF_CLASSES)

            ingest_in_tiledb(
                uri=tiledb_uri_x,
                data=np.random.rand(*dataset_shape_x),
                batch_size=BATCH_SIZE,
                sparse=False,
            )
            ingest_in_tiledb(
                uri=tiledb_uri_y,
                data=np.random.rand(*dataset_shape_y),
                batch_size=BATCH_SIZE,
                sparse=False,
            )

            with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
                tiledb_dataset = TensorflowTileDBDenseDataset(
                    x_array=x, y_array=y, batch_size=BATCH_SIZE
                )

                assert len(tiledb_dataset) == ROWS
