"""Tests for TileDB integration with Tensorflow Data API."""

import os
import tempfile

import tiledb
import numpy as np
import uuid

import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test

from tiledb.ml.data_apis.tensorflow_sparse import TensorflowTileDBSparseDataset

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.compat.v1.enable_eager_execution()

# Test parameters
NUM_OF_CLASSES = 1
BATCH_SIZE = 32
ROWS = 1000

# We test for 2d
INPUT_SHAPES = [
    (10,),
]


def create_sparse_array_one_hot_2d(rows: int, cols: tuple) -> np.ndarray:
    seed = np.random.randint(low=0, high=cols[0], size=(rows,))
    seed[-1] = cols[0] - 1
    b = np.zeros((seed.size, seed.max() + 1))
    b[np.arange(seed.size), seed] = 1
    return b


def get_schema(data: np.array, batch_size: int, sparse: bool) -> tiledb.ArraySchema:
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
        sparse=sparse,
        attrs=[tiledb.Attr(name="features", dtype=np.float32)],
    )

    return schema


def ingest_in_tiledb(data: np.array, batch_size: int, uri: str, sparse: bool):
    schema = get_schema(data, batch_size, sparse)

    # Create the (empty) array on disk.
    tiledb.Array.create(uri, schema)

    # Ingest
    with tiledb.open(uri, "w") as tiledb_array:
        idx = np.nonzero(data) if sparse else slice(None)
        tiledb_array[idx] = {"features": data[idx]}


def create_model(input_shape: tuple, num_of_classes: int) -> object:
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape, sparse=True))
    # TODO: TF https://github.com/tensorflow/tensorflow/issues/47532
    # TODO: TF https://github.com/tensorflow/tensorflow/issues/47931
    model.compile()

    return model


class TestTileDBTensorflowSparseDataAPI(test.TestCase):
    @testing_utils.run_v2_only
    def test_tiledb_tf_sparse_data_api_with_with_sparse_data_sparse_label(self):
        for input_shape in INPUT_SHAPES:
            with self.subTest():
                model = create_model(
                    input_shape=input_shape, num_of_classes=NUM_OF_CLASSES
                )

                array_uuid = str(uuid.uuid4())
                tiledb_uri_x = os.path.join(self.get_temp_dir(), "x" + array_uuid)
                tiledb_uri_y = os.path.join(self.get_temp_dir(), "y" + array_uuid)

                dataset_shape_x = (ROWS, input_shape)
                dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

                ingest_in_tiledb(
                    uri=tiledb_uri_x,
                    data=create_sparse_array_one_hot_2d(
                        dataset_shape_x[0], dataset_shape_x[1]
                    ),
                    batch_size=BATCH_SIZE,
                    sparse=True,
                )
                ingest_in_tiledb(
                    uri=tiledb_uri_y,
                    data=create_sparse_array_one_hot_2d(
                        dataset_shape_y[0], dataset_shape_y[1]
                    ),
                    batch_size=BATCH_SIZE,
                    sparse=True,
                )

                with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
                    tiledb_dataset = TensorflowTileDBSparseDataset(
                        x_array=x, y_array=y, batch_size=BATCH_SIZE
                    )

                    self.assertIsInstance(tiledb_dataset, tf.data.Dataset)
                    model.fit(tiledb_dataset, verbose=0, epochs=2)

    @testing_utils.run_v2_only
    def test_tiledb_tf_sparse_data_api_with_sparse_data_dense_label(self):
        for input_shape in INPUT_SHAPES:
            with self.subTest():
                model = create_model(
                    input_shape=input_shape, num_of_classes=NUM_OF_CLASSES
                )

                array_uuid = str(uuid.uuid4())
                tiledb_uri_x = os.path.join(self.get_temp_dir(), "x" + array_uuid)
                tiledb_uri_y = os.path.join(self.get_temp_dir(), "y" + array_uuid)

                dataset_shape_x = (ROWS, input_shape)
                dataset_shape_y = (ROWS, NUM_OF_CLASSES)

                ingest_in_tiledb(
                    uri=tiledb_uri_x,
                    data=create_sparse_array_one_hot_2d(
                        dataset_shape_x[0], dataset_shape_x[1]
                    ),
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

                    self.assertIsInstance(tiledb_dataset, tf.data.Dataset)
                    model.fit(tiledb_dataset, verbose=0, epochs=2)

    @testing_utils.run_v2_only
    def test_tiledb_tf_sparse_data_api_with_sparse_data_diff_number_of_batch_x_y_rows(
        self,
    ):
        for input_shape in INPUT_SHAPES:
            with self.subTest():
                model = create_model(
                    input_shape=input_shape, num_of_classes=NUM_OF_CLASSES
                )
                array_uuid = str(uuid.uuid4())
                tiledb_uri_x = os.path.join(self.get_temp_dir(), "x" + array_uuid)
                tiledb_uri_y = os.path.join(self.get_temp_dir(), "y" + array_uuid)

                dataset_shape_x = (ROWS, input_shape)
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
                    with self.assertRaises(Exception):
                        model.fit(tiledb_dataset, verbose=0, epochs=2)

    @testing_utils.run_v2_only
    def test_sparse_except_with_diff_number_of_x_y_rows(self):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(self.get_temp_dir(), "x" + array_uuid)
        tiledb_uri_y = os.path.join(self.get_temp_dir(), "y" + array_uuid)

        # Add one extra row on X
        dataset_shape_x = (ROWS + 1, INPUT_SHAPES[0])
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
            with self.assertRaises(Exception):
                TensorflowTileDBSparseDataset(
                    x_array=x, y_array=y, batch_size=BATCH_SIZE
                )

    def test_except_with_diff_number_of_batch_x_y_rows_empty_record(self):
        for input_shape in INPUT_SHAPES:
            with self.subTest():
                with tempfile.TemporaryDirectory() as tiledb_uri_x, tempfile.TemporaryDirectory() as tiledb_uri_y:
                    model = create_model(
                        input_shape=input_shape, num_of_classes=NUM_OF_CLASSES
                    )
                    # Add one extra row on X
                    dataset_shape_x = (ROWS, INPUT_SHAPES[0])
                    dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

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
                        with self.assertRaises(Exception):
                            model.fit(tiledb_dataset, verbose=0, epochs=2)

    def test_except_with_multiple_nz_value_record_of_batch_x_y_rows(self):
        for input_shape in INPUT_SHAPES:
            with self.subTest():
                with tempfile.TemporaryDirectory() as tiledb_uri_x, tempfile.TemporaryDirectory() as tiledb_uri_y:
                    model = create_model(
                        input_shape=input_shape, num_of_classes=NUM_OF_CLASSES
                    )
                    # Add one extra row on X
                    dataset_shape_x = (ROWS, input_shape)
                    dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

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
