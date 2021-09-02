"""Tests for TileDB integration with Tensorflow Data API."""

import os
import pytest
import tiledb
import numpy as np
import uuid
import tensorflow as tf

from tiledb.ml.readers.tensorflow_sparse import TensorflowTileDBSparseDataset
from tiledb.ml._utils import ingest_in_tiledb, create_sparse_array_one_hot_2d

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Test parameters
NUM_OF_CLASSES = 1
BATCH_SIZE = 32
ROWS = 1000


@pytest.mark.parametrize(
    "input_shape",
    [
        (10,),
    ],
)
# We test for single and multiple attributes
@pytest.mark.parametrize(
    "num_of_attributes",
    [1, 2, 3],
)
class TestTileDBTensorflowSparseDataAPI:
    def test_tiledb_tf_sparse_data_api_with_sparse_data_sparse_label(
        self, tmpdir, input_shape, num_of_attributes
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        dataset_shape_x = (ROWS, input_shape)
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(dataset_shape_x[0], dataset_shape_x[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(dataset_shape_y[0], dataset_shape_y[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x,
                y_array=y,
                x_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                y_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                batch_size=BATCH_SIZE,
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)

    def test_tiledb_tf_sparse_data_api_with_with_dense_data_except(
        self, tmpdir, input_shape, num_of_attributes
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        dataset_shape_x = (ROWS,) + input_shape
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(*dataset_shape_x),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(dataset_shape_y[0], dataset_shape_y[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(TypeError):
                TensorflowTileDBSparseDataset(
                    x_array=x,
                    y_array=y,
                    x_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    y_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    batch_size=BATCH_SIZE,
                )

        # Same test without attribute names explicitly provided by the user
        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(TypeError):
                TensorflowTileDBSparseDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                )

    def test_tiledb_tf_sparse_data_api_with_sparse_data_dense_label(
        self, tmpdir, input_shape, num_of_attributes
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        dataset_shape_x = (ROWS, input_shape)
        dataset_shape_y = (ROWS, NUM_OF_CLASSES)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(dataset_shape_x[0], dataset_shape_x[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x,
                y_array=y,
                x_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                y_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                batch_size=BATCH_SIZE,
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)

    def test_tiledb_tf_sparse_data_api_with_sparse_data_diff_number_of_batch_x_y_rows(
        self, tmpdir, input_shape, num_of_attributes
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

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
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x,
                y_array=y,
                x_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                y_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                batch_size=BATCH_SIZE,
            )
            with pytest.raises(Exception):
                next(tiledb_dataset)

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
            )
            with pytest.raises(Exception):
                next(tiledb_dataset)

    def test_sparse_except_with_diff_number_of_x_y_rows(
        self, tmpdir, input_shape, num_of_attributes
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        # Add one extra row on X
        dataset_shape_x = (ROWS + 1, input_shape)
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(dataset_shape_x[0], dataset_shape_x[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(dataset_shape_y[0], dataset_shape_y[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(Exception):
                TensorflowTileDBSparseDataset(
                    x_array=x,
                    y_array=y,
                    x_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    y_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    batch_size=BATCH_SIZE,
                )

        # Same test without attribute names explicitly provided by the user
        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(Exception):
                TensorflowTileDBSparseDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                )

    def test_except_with_diff_number_of_batch_x_y_rows_empty_record(
        self, tmpdir, input_shape, num_of_attributes
    ):
        # Add one extra row on X
        dataset_shape_x = (ROWS, input_shape)
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
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x,
                y_array=y,
                x_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                y_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                batch_size=BATCH_SIZE,
            )
            with pytest.raises(Exception):
                next(tiledb_dataset)

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = TensorflowTileDBSparseDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
            )
            with pytest.raises(Exception):
                next(tiledb_dataset)

    def test_generator_sparse_x_dense_y_batch_output(
        self, tmpdir, input_shape, num_of_attributes
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        dataset_shape_x = (ROWS, input_shape)
        dataset_shape_y = (ROWS, NUM_OF_CLASSES)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(dataset_shape_x[0], dataset_shape_x[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(*dataset_shape_y),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            attribute_names = [
                "features_" + str(attr) for attr in range(num_of_attributes)
            ]

            generated_data = next(
                TensorflowTileDBSparseDataset._generator_sparse_dense(
                    x=x,
                    y=y,
                    x_attribute_names=attribute_names,
                    y_attribute_names=attribute_names,
                    rows=ROWS,
                    batch_size=BATCH_SIZE,
                )
            )

            assert len(generated_data) == 2 * num_of_attributes

            for attr in range(num_of_attributes):
                assert isinstance(generated_data[attr], tf.sparse.SparseTensor)
                assert isinstance(generated_data[attr + num_of_attributes], np.ndarray)

                # Coords should be equal to batch for both x and y
                assert generated_data[attr].indices.shape[0] == BATCH_SIZE

                assert generated_data[attr + num_of_attributes].shape == (
                    BATCH_SIZE,
                    NUM_OF_CLASSES,
                )

    def test_generator_sparse_x_sparse_y_batch_output(
        self, tmpdir, input_shape, num_of_attributes
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        # Add one extra row on X
        dataset_shape_x = (ROWS + 1, input_shape)
        dataset_shape_y = (ROWS, (NUM_OF_CLASSES,))

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(dataset_shape_x[0], dataset_shape_x[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(dataset_shape_y[0], dataset_shape_y[1]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            attribute_names = [
                "features_" + str(attr) for attr in range(num_of_attributes)
            ]

            generated_data = next(
                TensorflowTileDBSparseDataset._generator_sparse_sparse(
                    x=x,
                    y=y,
                    x_attribute_names=attribute_names,
                    y_attribute_names=attribute_names,
                    rows=ROWS,
                    batch_size=BATCH_SIZE,
                )
            )

            assert len(generated_data) == 2 * num_of_attributes

            for attr in range(num_of_attributes):
                assert isinstance(generated_data[attr], tf.sparse.SparseTensor)
                assert isinstance(
                    generated_data[attr + num_of_attributes], tf.sparse.SparseTensor
                )

                # Coords should be equal to batch for both x and y
                assert generated_data[attr].indices.shape[0] == BATCH_SIZE

                assert generated_data[attr + num_of_attributes].shape == (
                    BATCH_SIZE,
                    NUM_OF_CLASSES,
                )
