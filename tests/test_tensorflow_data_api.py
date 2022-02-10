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

from .utils import ingest_in_tiledb

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Test parameters
NUM_OF_CLASSES = 5
BATCH_SIZE = 32
ROWS = 1000


@pytest.mark.parametrize(
    "input_shape",
    [
        (10,),
        (10, 3),
        (10, 10, 3),
    ],
)
# We test for single and multiple attributes
@pytest.mark.parametrize(
    "num_of_attributes",
    [1, 2, 3],
)
@pytest.mark.parametrize(
    "batch_shuffle",
    [True, False],
)
@pytest.mark.parametrize(
    "within_batch_shuffle",
    [True, False],
)
@pytest.mark.parametrize(
    "buffer_size",
    [50, None],
)
class TestTileDBTensorflowDataAPI:
    def test_tiledb_tf_data_api_with_multiple_dim_data(
        self,
        tmpdir,
        input_shape,
        num_of_attributes,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(ROWS, *input_shape[1:]),
            batch_size=BATCH_SIZE,
            sparse=False,
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
                batch_shuffle=batch_shuffle,
                buffer_size=buffer_size,
                within_batch_shuffle=within_batch_shuffle,
                x_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
                y_attribute_names=[
                    "features_" + str(attr) for attr in range(num_of_attributes)
                ],
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)

            # Same test without attribute names explicitly provided by the user
            tiledb_dataset = TensorflowTileDBDataset(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                batch_shuffle=batch_shuffle,
                buffer_size=buffer_size,
                within_batch_shuffle=within_batch_shuffle,
            )

            assert isinstance(tiledb_dataset, tf.data.Dataset)

    def test_except_with_diff_number_of_x_y_rows(
        self,
        tmpdir,
        input_shape,
        num_of_attributes,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            # Add one extra row on X
            data=np.random.rand(ROWS + 1, *input_shape[1:]),
            batch_size=BATCH_SIZE,
            sparse=False,
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
            with pytest.raises(Exception):
                TensorflowTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    batch_shuffle=batch_shuffle,
                    buffer_size=buffer_size,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    y_attribute_names=[
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
                    batch_shuffle=batch_shuffle,
                    buffer_size=buffer_size,
                    within_batch_shuffle=within_batch_shuffle,
                )

    def test_dataset_generator_batch_output(
        self,
        tmpdir,
        input_shape,
        num_of_attributes,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            data=np.random.rand(ROWS, *input_shape[1:]),
            batch_size=BATCH_SIZE,
            sparse=False,
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
            attribute_names = [
                "features_" + str(attr) for attr in range(num_of_attributes)
            ]
            kwargs = dict(
                x_array=x,
                y_array=y,
                batch_size=BATCH_SIZE,
                batch_shuffle=batch_shuffle,
                within_batch_shuffle=within_batch_shuffle,
            )
            # Test the generator twice: once with the public api (TensorflowTileDBDataset)
            # and once with calling tensor_generator directly. Although the former calls
            # the latter internally, it is not reported as covered by the coverage report
            # due to https://github.com/tensorflow/tensorflow/issues/33759
            generators = [
                iter(
                    TensorflowTileDBDataset(
                        x_attribute_names=attribute_names,
                        y_attribute_names=attribute_names,
                        buffer_size=buffer_size,
                        **kwargs
                    )
                ),
                tensor_generator(
                    dense_batch_cls=TensorflowDenseBatch,
                    sparse_batch_cls=TensorflowSparseBatch,
                    x_attrs=attribute_names,
                    y_attrs=attribute_names,
                    buffer_size=buffer_size or BATCH_SIZE,
                    **kwargs
                ),
            ]
            for generator in generators:
                generated_data = next(generator)
                assert len(generated_data) == 2 * num_of_attributes
                for attr in range(num_of_attributes):
                    assert tuple(generated_data[attr].shape) <= (
                        BATCH_SIZE,
                        *input_shape[1:],
                    )
                    assert tuple(generated_data[num_of_attributes + attr].shape) <= (
                        BATCH_SIZE,
                        NUM_OF_CLASSES,
                    )

    def test_buffer_size_geq_batch_size_exception(
        self,
        tmpdir,
        input_shape,
        num_of_attributes,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        array_uuid = str(uuid.uuid4())
        tiledb_uri_x = os.path.join(tmpdir, "x" + array_uuid)
        tiledb_uri_y = os.path.join(tmpdir, "y" + array_uuid)

        ingest_in_tiledb(
            uri=tiledb_uri_x,
            # Add one extra row on X
            data=np.random.rand(ROWS + 1, *input_shape[1:]),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )
        ingest_in_tiledb(
            uri=tiledb_uri_y,
            data=np.random.rand(ROWS, NUM_OF_CLASSES),
            batch_size=BATCH_SIZE,
            sparse=False,
            num_of_attributes=num_of_attributes,
        )

        # Set the buffer_size less than the batch_size
        buffer_size = 10
        with tiledb.open(tiledb_uri_x) as x, tiledb.open(tiledb_uri_y) as y:
            with pytest.raises(ValueError):
                TensorflowTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    batch_shuffle=batch_shuffle,
                    buffer_size=buffer_size,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                    y_attribute_names=[
                        "features_" + str(attr) for attr in range(num_of_attributes)
                    ],
                )
