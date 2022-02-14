"""Tests for TileDB integration with Tensorflow Data API."""

import os

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


@pytest.mark.parametrize("input_shape", [(10,), (10, 3), (10, 10, 3)])
@pytest.mark.parametrize("num_attrs", [1, 2, 3])
@pytest.mark.parametrize("batch_shuffle", [True, False])
@pytest.mark.parametrize("within_batch_shuffle", [True, False])
@pytest.mark.parametrize("buffer_size", [50, None])
class TestTileDBTensorflowDataAPI:
    def test_data_api_with_multiple_dim_data(
        self,
        tmpdir,
        input_shape,
        num_attrs,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=np.random.rand(ROWS, *input_shape[1:]),
            data_y=np.random.rand(ROWS, NUM_OF_CLASSES),
            sparse_x=False,
            sparse_y=False,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            for pass_attrs in True, False:
                dataset = TensorflowTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    batch_shuffle=batch_shuffle,
                    buffer_size=buffer_size,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )
                assert isinstance(dataset, tf.data.Dataset)

    def test_except_with_diff_number_of_x_y_rows(
        self,
        tmpdir,
        input_shape,
        num_attrs,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            # Add one extra row on X
            data_x=np.random.rand(ROWS + 1, *input_shape[1:]),
            sparse_x=False,
            data_y=np.random.rand(ROWS, NUM_OF_CLASSES),
            sparse_y=False,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            for pass_attrs in True, False:
                with pytest.raises(Exception):
                    TensorflowTileDBDataset(
                        x_array=x,
                        y_array=y,
                        batch_size=BATCH_SIZE,
                        batch_shuffle=batch_shuffle,
                        buffer_size=buffer_size,
                        within_batch_shuffle=within_batch_shuffle,
                        x_attrs=attrs if pass_attrs else [],
                        y_attrs=attrs if pass_attrs else [],
                    )

    def test_dataset_generator_batch_output(
        self,
        tmpdir,
        input_shape,
        num_attrs,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=np.random.rand(ROWS, *input_shape[1:]),
            data_y=np.random.rand(ROWS, NUM_OF_CLASSES),
            sparse_x=False,
            sparse_y=False,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            for pass_attrs in True, False:
                kwargs = dict(
                    x_array=x,
                    y_array=y,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                    buffer_size=buffer_size,
                    batch_size=BATCH_SIZE,
                    batch_shuffle=batch_shuffle,
                    within_batch_shuffle=within_batch_shuffle,
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
                        **kwargs,
                    ),
                ]
                for generator in generators:
                    generated_data = next(generator)
                    assert len(generated_data) == 2 * num_attrs
                    for attr in range(num_attrs):
                        assert tuple(generated_data[attr].shape) <= (
                            BATCH_SIZE,
                            *input_shape[1:],
                        )
                        assert tuple(generated_data[num_attrs + attr].shape) <= (
                            BATCH_SIZE,
                            NUM_OF_CLASSES,
                        )

    def test_buffer_size_geq_batch_size_exception(
        self,
        tmpdir,
        input_shape,
        num_attrs,
        batch_shuffle,
        within_batch_shuffle,
        buffer_size,
    ):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=np.random.rand(ROWS, *input_shape[1:]),
            sparse_x=False,
            data_y=np.random.rand(ROWS, NUM_OF_CLASSES),
            sparse_y=False,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            for pass_attrs in True, False:
                dataset = TensorflowTileDBDataset(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    # Set the buffer_size less than the batch_size
                    buffer_size=BATCH_SIZE - 1,
                    batch_shuffle=batch_shuffle,
                    within_batch_shuffle=within_batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )
                with pytest.raises(Exception) as excinfo:
                    next(iter(dataset))
                assert str(excinfo.value).startswith(
                    "ValueError: Buffer size should be greater or equal to batch size"
                )
