"""Tests for TileDB integration with Tensorflow Data API."""

import os
import pytest
import tiledb
import numpy as np
import uuid

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tiledb.ml.readers.tensorflow_sparse import TensorflowTileDBSparseDataset
from tiledb.ml._utils import ingest_in_tiledb, create_sparse_array_one_hot_2d

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Test parameters
NUM_OF_CLASSES = 1
BATCH_SIZE = 32
ROWS = 1000


@pytest.fixture(
    params=[
        {
            "input_shape": (10,),
        },
    ]
)
def model(request):
    model = Sequential()
    model.add(tf.keras.Input(shape=request.param["input_shape"], sparse=True))
    # TODO: TF https://github.com/tensorflow/tensorflow/issues/47532
    # TODO: TF https://github.com/tensorflow/tensorflow/issues/47931
    model.compile()

    return model


class TestTileDBTensorflowSparseDataAPI:
    def test_tiledb_tf_sparse_data_api_with_with_sparse_data_sparse_label(
        self, tmpdir, model
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        dataset_shape_x = (ROWS, model.input_shape[1:])
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(dataset_shape_x[0], dataset_shape_x[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(dataset_shape_y[0], dataset_shape_y[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)
            model.fit(tiledb_dataset, verbose=0, epochs=2)

    def test_tiledb_tf_sparse_data_api_with_sparse_data_dense_label(
        self, tmpdir, model
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        dataset_shape_x = (ROWS, model.input_shape[1:])
        dataset_shape_y = (ROWS, NUM_OF_CLASSES)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(dataset_shape_x[0], dataset_shape_x[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)
            model.fit(tiledb_dataset, verbose=0, epochs=2)

    def test_tiledb_tf_sparse_data_api_with_sparse_data_diff_number_of_batch_x_y_rows(
        self, tmpdir, model
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        dataset_shape_x = (ROWS, model.input_shape[1:])
        dataset_shape_y = (ROWS, NUM_OF_CLASSES)

        # Empty one random row
        spoiled_data = create_sparse_array_one_hot_2d(
            dataset_shape_x[0], dataset_shape_x[1]
        )
        spoiled_data[np.nonzero(spoiled_data[0])] = 0

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=spoiled_data,
            batch_size=BATCH_SIZE,
            sparse=True,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )
            with pytest.raises(Exception):
                model.fit(tiledb_dataset, verbose=0, epochs=2)

    def test_sparse_except_with_diff_number_of_x_y_rows(self, tmpdir, model):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        # Add one extra row on X
        dataset_shape_x = (ROWS + 1, model.input_shape[1:])
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(dataset_shape_x[0], dataset_shape_x[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(dataset_shape_y[0], dataset_shape_y[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(Exception):
                TensorflowTileDBSparseDataset(
                    x_array=x, y_array=y, batch_size=BATCH_SIZE
                )

    def test_except_with_diff_number_of_batch_x_y_rows_empty_record(
        self, tmpdir, model
    ):
        # Add one extra row on X
        dataset_shape_x = (ROWS, model.input_shape[1:])
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        spoiled_data = create_sparse_array_one_hot_2d(*dataset_shape_x)
        spoiled_data[np.nonzero(spoiled_data[0])] = 0

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=spoiled_data,
            batch_size=BATCH_SIZE,
            sparse=True,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )
            with pytest.raises(Exception):
                model.fit(tiledb_dataset, verbose=0, epochs=2)

    def test_except_with_multiple_nz_value_record_of_batch_x_y_rows(
        self, tmpdir, model
    ):
        # Add one extra row on X
        dataset_shape_x = (ROWS, model.input_shape[1:])
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        spoiled_data = create_sparse_array_one_hot_2d(*dataset_shape_x)
        spoiled_data[0] += 1

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=spoiled_data,
            batch_size=BATCH_SIZE,
            sparse=True,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x, y_array=y, batch_size=BATCH_SIZE
            )
            model.fit(tiledb_dataset, verbose=0, epochs=2)
