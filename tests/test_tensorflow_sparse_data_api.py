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

from .utils import create_rand_labels, ingest_in_tiledb

# Suppress all Tensorflow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Test parameters
NUM_OF_FEATURES = 10
NUM_OF_CLASSES = 5
BATCH_SIZE = 32
ROWS = 1000


@pytest.mark.parametrize("num_attrs", [1])
@pytest.mark.parametrize("batch_shuffle", [False, True])
@pytest.mark.parametrize("buffer_size", [50, None])
class TestTensorflowTileDBDatasetSparse:
    def test_sparse_x_sparse_y(self, tmpdir, num_attrs, batch_shuffle, buffer_size):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=create_rand_labels(ROWS, NUM_OF_FEATURES, one_hot=True),
            data_y=create_rand_labels(ROWS, NUM_OF_CLASSES, one_hot=True),
            sparse_x=True,
            sparse_y=True,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            for pass_attrs in True, False:
                kwargs = dict(
                    x_array=x,
                    y_array=y,
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )
                dataset = TensorflowTileDBDataset(**kwargs)
                assert isinstance(dataset, tf.data.Dataset)
                # Test the generator twice: once with the public api (TensorflowTileDBDataset)
                # and once with calling tensor_generator directly. Although the former calls
                # the latter internally, it is not reported as covered by the coverage report
                # due to https://github.com/tensorflow/tensorflow/issues/33759
                generators = [
                    iter(dataset),
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
                        assert isinstance(generated_data[attr], tf.SparseTensor)
                        assert isinstance(
                            generated_data[attr + num_attrs], tf.SparseTensor
                        )

                        # Coords should be equal to batch for both x and y
                        assert generated_data[attr].indices.shape[0] <= BATCH_SIZE

                        assert tuple(generated_data[attr + num_attrs].shape.dims) <= (
                            BATCH_SIZE,
                            NUM_OF_CLASSES,
                        )

    def test_sparse_x_dense_y(self, tmpdir, num_attrs, batch_shuffle, buffer_size):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=create_rand_labels(ROWS, NUM_OF_FEATURES, one_hot=True),
            data_y=np.random.rand(ROWS, NUM_OF_CLASSES),
            sparse_x=True,
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
                    batch_size=BATCH_SIZE,
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )
                dataset = TensorflowTileDBDataset(**kwargs)
                assert isinstance(dataset, tf.data.Dataset)
                # Test the generator twice: once with the public api (TensorflowTileDBDataset)
                # and once with calling tensor_generator directly. Although the former calls
                # the latter internally, it is not reported as covered by the coverage report
                # due to https://github.com/tensorflow/tensorflow/issues/33759
                generators = [
                    iter(dataset),
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
                        assert isinstance(generated_data[attr], tf.SparseTensor)
                        assert isinstance(generated_data[attr + num_attrs], tf.Tensor)

                        # Coords should be equal to batch for both x and y
                        assert generated_data[attr].indices.shape[0] <= BATCH_SIZE
                        assert tuple(generated_data[attr + num_attrs].shape) <= (
                            BATCH_SIZE,
                            NUM_OF_CLASSES,
                        )

    def test_unequal_num_rows(self, tmpdir, num_attrs, batch_shuffle, buffer_size):
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            # Add one extra row on X
            data_x=create_rand_labels(ROWS + 1, NUM_OF_FEATURES, one_hot=True),
            data_y=create_rand_labels(ROWS, NUM_OF_CLASSES, one_hot=True),
            sparse_x=True,
            sparse_y=True,
            batch_size=BATCH_SIZE,
            num_attrs=num_attrs,
        )
        attrs = [f"features_{attr}" for attr in range(num_attrs)]
        with tiledb.open(uri_x) as x, tiledb.open(uri_y) as y:
            for pass_attrs in True, False:
                with pytest.raises(ValueError):
                    TensorflowTileDBDataset(
                        x_array=x,
                        y_array=y,
                        batch_size=BATCH_SIZE,
                        buffer_size=buffer_size,
                        batch_shuffle=batch_shuffle,
                        x_attrs=attrs if pass_attrs else [],
                        y_attrs=attrs if pass_attrs else [],
                    )

    def test_unequal_num_rows_in_batch(
        self, tmpdir, num_attrs, batch_shuffle, buffer_size
    ):
        spoiled_data = create_rand_labels(ROWS, NUM_OF_FEATURES, one_hot=True)
        spoiled_data[np.nonzero(spoiled_data[0])] = 0
        uri_x, uri_y = ingest_in_tiledb(
            tmpdir,
            data_x=spoiled_data,
            data_y=create_rand_labels(ROWS, NUM_OF_CLASSES, one_hot=True),
            sparse_x=True,
            sparse_y=True,
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
                    buffer_size=buffer_size,
                    batch_shuffle=batch_shuffle,
                    x_attrs=attrs if pass_attrs else [],
                    y_attrs=attrs if pass_attrs else [],
                )
                with pytest.raises(Exception):
                    for _ in dataset:
                        pass
