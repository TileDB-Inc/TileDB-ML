"""Tests for TileDB integration with Tensorflow Data API."""

import os
import uuid

import numpy as np
import pytest
import tensorflow as tf

import tiledb
from tiledb.ml.readers._batch_utils import tensor_generator
from tiledb.ml.readers.tensorflow import (
    TensorflowDenseBatch,
    TensorflowSparseBatch,
    TensorflowTileDBDataset,
)

from .utils import create_sparse_array_one_hot_2d, ingest_in_tiledb

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
    [
        1,
    ],
)
@pytest.mark.parametrize("batch_shuffle", [True, False])
@pytest.mark.parametrize(
    "buffer_size",
    [50, None],
)
class TestTileDBTensorflowSparseDataAPI:
    def test_tiledb_tf_sparse_data_api_with_sparse_data_sparse_label(
        self, tmpdir, input_shape, num_of_attributes, batch_shuffle, buffer_size
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(ROWS, input_shape[0]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                x_attrs=["features_" + str(attr) for attr in range(num_of_attributes)],
                y_attrs=["features_" + str(attr) for attr in range(num_of_attributes)],
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = TensorflowTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)

    def test_tiledb_tf_sparse_data_api_with_dense_data_sparse_label_except(
        self, tmpdir, input_shape, num_of_attributes, batch_shuffle, buffer_size
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(ROWS, *input_shape),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(TypeError):
                TensorflowTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    x_attrs=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    y_attrs=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                )

        # Same test without attribute names explicitly provided by the user
        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(TypeError):
                TensorflowTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                )

    def test_tiledb_tf_sparse_data_api_with_sparse_data_dense_label(
        self, tmpdir, input_shape, num_of_attributes, batch_shuffle, buffer_size
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(ROWS, input_shape[0]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                x_attrs=["features_" + str(attr) for attr in range(num_of_attributes)],
                y_attrs=["features_" + str(attr) for attr in range(num_of_attributes)],
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = TensorflowTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)

    def test_tiledb_tf_sparse_data_api_with_sparse_data_diff_number_of_batch_x_y_rows(
        self, tmpdir, input_shape, num_of_attributes, batch_shuffle, buffer_size
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        # Empty one random row
        spoiled_data = create_sparse_array_one_hot_2d(ROWS, input_shape[0])
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
            data=np.random.rand(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                x_attrs=["features_" + str(attr) for attr in range(num_of_attributes)],
                y_attrs=["features_" + str(attr) for attr in range(num_of_attributes)],
            )

            with pytest.raises(Exception):
                for _ in tiledb_dataset:
                    pass

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = TensorflowTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
            )

            with pytest.raises(Exception):
                for _ in tiledb_dataset:
                    pass

    def test_sparse_except_with_diff_number_of_x_y_rows(
        self, tmpdir, input_shape, num_of_attributes, batch_shuffle, buffer_size
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            # Add one extra row on X
            data=create_sparse_array_one_hot_2d(ROWS + 1, input_shape[0]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(Exception):
                TensorflowTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    x_attrs=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    y_attrs=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                )

        # Same test without attribute names explicitly provided by the user
        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(Exception):
                TensorflowTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                )

    def test_except_with_diff_number_of_batch_x_y_rows_empty_record(
        self, tmpdir, input_shape, num_of_attributes, batch_shuffle, buffer_size
    ):
        tiledb_uri_x = os.path.join(tmpdir, "x")
        tiledb_uri_y = os.path.join(tmpdir, "y")

        spoiled_data = create_sparse_array_one_hot_2d(ROWS, input_shape[0])
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
            data=create_sparse_array_one_hot_2d(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            tiledb_dataset = TensorflowTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
                x_attrs=["features_" + str(attr) for attr in range(num_of_attributes)],
                y_attrs=["features_" + str(attr) for attr in range(num_of_attributes)],
            )
            with pytest.raises(Exception):
                for _ in tiledb_dataset:
                    pass

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = TensorflowTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                buffer_size=buffer_size,
                batch_shuffle=batch_shuffle,
            )
            with pytest.raises(Exception):
                for _ in tiledb_dataset:
                    pass

    def test_generator_sparse_x_dense_y_batch_output(
        self, tmpdir, input_shape, num_of_attributes, batch_shuffle, buffer_size
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(ROWS, input_shape[0]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            attrs = ["features_" + str(attr) for attr in range(num_of_attributes)]
            kwargs = dict(
                x_array=x,
                y_array=y,
                x_attrs=attrs,
                y_attrs=attrs,
                buffer_size=buffer_size,
                batch_size=BATCH_SIZE,
                batch_shuffle=batch_shuffle,
            )
            # Test the generator twice: once with the public api (TensorflowTileDBDataset)
            # and once with calling tensor_generator directly. Although the former calls
            # the latter internally, it is not reported as covered by the coverage report
            # due to https://github.com/tensorflow/tensorflow/issues/33759
            generators = [
                iter(TensorflowTileDBDataset(**kwargs)),
                tensor_generator(
                    dense_batch_cls=TensorflowDenseBatch,
                    sparse_batch_cls=TensorflowSparseBatch,
                    **kwargs
                ),
            ]
            for generator in generators:
                generated_data = next(generator)
                assert len(generated_data) == 2 * num_of_attributes

                for attr in range(num_of_attributes):
                    assert isinstance(generated_data[attr], tf.SparseTensor)
                    assert isinstance(
                        generated_data[attr + num_of_attributes], tf.Tensor
                    )

                    # Coords should be equal to batch for both x and y
                    assert generated_data[attr].indices.shape[0] <= BATCH_SIZE
                    assert tuple(generated_data[attr + num_of_attributes].shape) <= (
                        BATCH_SIZE,
                        NUM_OF_CLASSES,
                    )

    def test_generator_sparse_x_sparse_y_batch_output(
        self, tmpdir, input_shape, num_of_attributes, batch_shuffle, buffer_size
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=create_sparse_array_one_hot_2d(ROWS, input_shape[0]),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=create_sparse_array_one_hot_2d(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=True,
            num_of_attributes=num_of_attributes,
        )

        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            attrs = ["features_" + str(attr) for attr in range(num_of_attributes)]
            kwargs = dict(
                x_array=x,
                y_array=y,
                x_attrs=attrs,
                y_attrs=attrs,
                buffer_size=buffer_size,
                batch_size=BATCH_SIZE,
                batch_shuffle=batch_shuffle,
            )
            # Test the generator twice: once with the public api (TensorflowTileDBDataset)
            # and once with calling tensor_generator directly. Although the former calls
            # the latter internally, it is not reported as covered by the coverage report
            # due to https://github.com/tensorflow/tensorflow/issues/33759
            generators = [
                iter(TensorflowTileDBDataset(**kwargs)),
                tensor_generator(
                    dense_batch_cls=TensorflowDenseBatch,
                    sparse_batch_cls=TensorflowSparseBatch,
                    **kwargs
                ),
            ]
            for generator in generators:
                generated_data = next(generator)
                assert len(generated_data) == 2 * num_of_attributes

                for attr in range(num_of_attributes):
                    assert isinstance(generated_data[attr], tf.SparseTensor)
                    assert isinstance(
                        generated_data[attr + num_of_attributes], tf.SparseTensor
                    )

                    # Coords should be equal to batch for both x and y
                    assert generated_data[attr].indices.shape[0] <= BATCH_SIZE

                    assert tuple(
                        generated_data[attr + num_of_attributes].shape.dims
                    ) <= (
                        BATCH_SIZE,
                        NUM_OF_CLASSES,
                    )
